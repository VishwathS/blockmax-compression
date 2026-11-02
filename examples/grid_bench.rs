//! Real-data loader: runs every registry codec on the block-max grids written by
//! Parker's `export_data` (`<prefix>._blocks.bin` / `._superblocks.bin`).
//!
//! Two modes:
//!
//! - `--mode size`: encodes EVERY row of the grid with every codec, checks every round
//!   trip, and reports exact whole-grid totals overall and per frequency stratum, plus
//!   the Raw baseline and a "nonzero entries × 1 byte" reference. Streams one row at a
//!   time (RSS stays at a few MB). Single-threaded.
//! - `--mode speed`: times encode and full-row decode on a sample of rows: a fixed
//!   number per stratum, the most frequent terms, and terms drawn from the dev queries
//!   (weighted by how many queries use them). Median of `--reps` timed runs after one
//!   warm-up, per row and codec. Run it pinned to one P-core thread:
//!   `taskset -c 2 nice -n 10 target/release/examples/grid_bench --mode speed ...`
//!
//! Strata: each term's stratum comes from its number of nonzero entries in the
//! SUPERBLOCKS grid (so a term is in the same stratum whichever grid is measured):
//! S0 = no nonzero entry, S1-S4 = quartiles of the nonzero count among the other terms.
//!
//! Outputs (never overwritten; the program refuses to start if any exists):
//!   size:  <out>_size_totals.csv, <out>_size_per_term.csv, <out>_size.md
//!   speed: <out>_speed_rows.csv, <out>_speed_summary.csv, <out>_speed.md
//!
//! Build: `cargo build --release --example grid_bench`
//! (timings from a debug build are meaningless; the program warns if it is one).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::File;
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use codecs::Codec;
use codecs::grid::{
    Grid, N_STRATA, QueryTerms, RowSize, STRATUM_NAMES, SizeScan, assign_strata, nonzero_per_row,
    size_scan,
};
use codecs::simdbp::Kernel;

const USAGE: &str = "\
grid_bench --mode size|speed --grid <._blocks.bin|._superblocks.bin> --out <prefix> [options]

  --strata-grid <path>   grid used to assign strata (default: the ._superblocks.bin
                         next to --grid)
  --queries <path>       ._queries.bin (default: next to --grid; speed mode only)
  --reps <n>             timed repetitions per row and codec, after 1 warm-up (default 5)
  --per-stratum <n>      rows sampled per stratum (default 64)
  --top <n>              most frequent terms added to the sample (default 32)
  --query-sample <n>     query terms added to the sample, weighted by use (default 128)
  --seed <n>             sampling seed (default 42)
";

/// Command-line options, after defaults are filled in.
#[derive(Debug, Clone)]
struct Opts {
    mode: Mode,
    grid: PathBuf,
    strata_grid: PathBuf,
    queries: PathBuf,
    out: PathBuf,
    reps: usize,
    per_stratum: usize,
    top: usize,
    query_sample: usize,
    seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Size,
    Speed,
}

/// Replaces the `._blocks.bin` / `._superblocks.bin` ending of a grid path with
/// `suffix` (e.g. `._queries.bin`). `None` if the path has neither ending.
fn sibling(grid: &Path, suffix: &str) -> Option<PathBuf> {
    let s = grid.to_str()?;
    let stem = s
        .strip_suffix("._blocks.bin")
        .or_else(|| s.strip_suffix("._superblocks.bin"))?;
    Some(PathBuf::from(format!("{stem}{suffix}")))
}

fn parse_args(args: &[String]) -> Result<Opts, String> {
    // Collect `--key value` pairs; every option takes exactly one value.
    let mut kv: BTreeMap<String, String> = BTreeMap::new();
    let mut it = args.iter();
    while let Some(key) = it.next() {
        if key == "--help" || key == "-h" {
            return Err(String::new());
        }
        let Some(name) = key.strip_prefix("--") else {
            return Err(format!("unexpected argument {key:?}"));
        };
        let value = it.next().ok_or(format!("--{name} needs a value"))?;
        if kv.insert(name.to_string(), value.clone()).is_some() {
            return Err(format!("--{name} given twice"));
        }
    }
    let mut take = |k: &str| kv.remove(k);
    let num = |v: Option<String>, k: &str, default: u64| -> Result<u64, String> {
        v.map_or(Ok(default), |s| {
            s.parse().map_err(|_| format!("--{k}: not a number: {s}"))
        })
    };

    let mode = match take("mode").as_deref() {
        Some("size") => Mode::Size,
        Some("speed") => Mode::Speed,
        Some(other) => return Err(format!("--mode must be size or speed, not {other:?}")),
        None => return Err("--mode is required".into()),
    };
    let grid = PathBuf::from(take("grid").ok_or("--grid is required")?);
    let out = PathBuf::from(take("out").ok_or("--out is required")?);
    let strata_grid = match take("strata-grid") {
        Some(p) => PathBuf::from(p),
        None => sibling(&grid, "._superblocks.bin")
            .ok_or("--grid doesn't end in ._blocks.bin/._superblocks.bin; pass --strata-grid")?,
    };
    let queries = match take("queries") {
        Some(p) => PathBuf::from(p),
        None => sibling(&grid, "._queries.bin")
            .ok_or("--grid doesn't end in ._blocks.bin/._superblocks.bin; pass --queries")?,
    };
    let opts = Opts {
        mode,
        grid,
        strata_grid,
        queries,
        out,
        reps: num(take("reps"), "reps", 5)? as usize,
        per_stratum: num(take("per-stratum"), "per-stratum", 64)? as usize,
        top: num(take("top"), "top", 32)? as usize,
        query_sample: num(take("query-sample"), "query-sample", 128)? as usize,
        seed: num(take("seed"), "seed", 42)?,
    };
    if let Some(unknown) = kv.keys().next() {
        return Err(format!("unknown option --{unknown}"));
    }
    if opts.reps == 0 {
        return Err("--reps must be at least 1".into());
    }
    Ok(opts)
}

/// Creates a NEW file for writing; fails if it already exists (results are never
/// overwritten).
fn create_new(path: &Path) -> Result<BufWriter<File>, String> {
    File::create_new(path).map(BufWriter::new).map_err(|e| {
        format!(
            "can't create {} (it must not exist yet): {e}",
            path.display()
        )
    })
}

/// `<out>` + suffix, e.g. `results/splade_blocks` + `_size.md`.
fn out_path(out: &Path, suffix: &str) -> PathBuf {
    PathBuf::from(format!("{}{suffix}", out.display()))
}

/// Fails before any work if an output file already exists or its folder doesn't.
fn check_outputs(paths: &[PathBuf]) -> Result<(), String> {
    for p in paths {
        if p.exists() {
            return Err(format!(
                "{} already exists; refusing to overwrite results",
                p.display()
            ));
        }
        let dir = p
            .parent()
            .filter(|d| !d.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        if !dir.is_dir() {
            return Err(format!("output folder {} doesn't exist", dir.display()));
        }
    }
    Ok(())
}

/// Facts about the run, printed at the top of every report so a number is never
/// separated from how it was produced.
fn run_info(opts: &Opts, grid: &Grid) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "- grid: `{}` (n_terms = {}, n_blocks = {})",
        opts.grid.display(),
        grid.n_terms(),
        grid.n_blocks()
    );
    let _ = writeln!(s, "- strata from: `{}`", opts.strata_grid.display());
    let _ = writeln!(
        s,
        "- build: {}",
        if cfg!(debug_assertions) {
            "DEBUG (timings meaningless)"
        } else {
            "release"
        }
    );
    let _ = writeln!(
        s,
        "- SIMDBP kernels chosen at run time: SimdBp128 → {:?}, SimdBp256 → {:?}, SimdBp512 → {:?}",
        Kernel::best(8),
        Kernel::best(16),
        Kernel::best(32)
    );
    let _ = writeln!(s, "- CPU: {}", cpu_model());
    let _ = writeln!(
        s,
        "- CPUs allowed (taskset): {}   nice: {}",
        proc_status("Cpus_allowed_list"),
        nice_value()
    );
    let _ = writeln!(
        s,
        "- command: `{}`",
        std::env::args().collect::<Vec<_>>().join(" ")
    );
    s
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|m| m.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

fn proc_status(key: &str) -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split(':').nth(1))
                .map(|v| v.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

/// This process's nice value: field 19 of /proc/self/stat (`man 5 proc`). The
/// command name (field 2) can contain spaces, so count fields after its closing ')'.
fn nice_value() -> String {
    std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|t| {
            let after = &t[t.rfind(')')? + 1..];
            after.split_whitespace().nth(16).map(str::to_string)
        })
        .unwrap_or_else(|| "unknown".into())
}

/// What `load_strata` returns: stratum per term, nonzero superblocks per term, and the
/// three quartile thresholds.
type Strata = (Vec<u8>, Vec<u32>, [u32; 3]);

/// Per-term nonzero counts in the strata grid, then strata (see `assign_strata`).
fn load_strata(opts: &Opts, n_terms: usize) -> Result<Strata, String> {
    let sgrid = Grid::open(&opts.strata_grid).map_err(|e| e.to_string())?;
    if sgrid.n_terms() != n_terms {
        return Err(format!(
            "strata grid has {} terms but the grid has {n_terms}",
            sgrid.n_terms()
        ));
    }
    eprintln!(
        "counting nonzero entries per term in {} ...",
        opts.strata_grid.display()
    );
    let nonzero = nonzero_per_row(&sgrid);
    let (strata, q) = assign_strata(&nonzero);
    Ok((strata, nonzero, q))
}

fn strata_info(strata: &[u8], q: &[u32; 3]) -> String {
    let mut counts = [0usize; N_STRATA];
    for &s in strata {
        counts[usize::from(s)] += 1;
    }
    format!(
        "- strata (nonzero superblocks per term): S0 = 0 [{} terms], S1 ≤ {} [{}], S2 ≤ {} [{}], S3 ≤ {} [{}], S4 > {} [{}]\n",
        counts[0], q[0], counts[1], q[1], counts[2], q[2], counts[3], q[2], counts[4]
    )
}

fn mb(bytes: u64) -> String {
    format!("{:.2}", bytes as f64 / 1e6)
}

// ------------------------------------------------------------------------------------
// size mode
// ------------------------------------------------------------------------------------

fn run_size(opts: &Opts) -> Result<(), String> {
    let totals_csv = out_path(&opts.out, "_size_totals.csv");
    let per_term_csv = out_path(&opts.out, "_size_per_term.csv");
    let md = out_path(&opts.out, "_size.md");
    check_outputs(&[totals_csv.clone(), per_term_csv.clone(), md.clone()])?;

    let grid = Grid::open(&opts.grid).map_err(|e| e.to_string())?;
    let (strata, _sb_nonzero, q) = load_strata(opts, grid.n_terms())?;
    let codecs = Codec::ALL;

    let mut per_term = create_new(&per_term_csv)?;
    let names: Vec<&str> = codecs.iter().map(|c| c.name()).collect();
    writeln!(
        per_term,
        "term,stratum,n_values,nonzero,max,{}",
        names.join(",")
    )
    .map_err(|e| e.to_string())?;

    let start = Instant::now();
    let n_terms = grid.n_terms();
    let mut io_error = None;
    let scan: SizeScan = size_scan(&grid, &strata, &codecs, |r: &RowSize| {
        let line = format!(
            "{},{},{},{},{},{}",
            r.term,
            r.stratum,
            r.n_values,
            r.nonzero,
            r.max,
            r.bytes
                .iter()
                .map(|b| b.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        if let Err(e) = writeln!(per_term, "{line}") {
            io_error.get_or_insert(e);
        }
        let done = r.term + 1;
        if done.is_multiple_of(1000) || done == n_terms {
            let secs = start.elapsed().as_secs_f64();
            let eta = secs / done as f64 * (n_terms - done) as f64;
            eprintln!("  {done}/{n_terms} rows, {secs:.0} s elapsed, ~{eta:.0} s left");
        }
    });
    if let Some(e) = io_error {
        return Err(format!("writing {}: {e}", per_term_csv.display()));
    }
    per_term.flush().map_err(|e| e.to_string())?;
    let secs = start.elapsed().as_secs_f64();

    // Totals CSV: one line per (group, codec), plus the reference line.
    let mut tc = create_new(&totals_csv)?;
    let w = |tc: &mut BufWriter<File>, s: String| writeln!(tc, "{s}").map_err(|e| e.to_string());
    w(
        &mut tc,
        "group,rows,values,nonzero,codec,bytes,bytes_per_value,ratio_vs_raw".into(),
    )?;
    let mut groups: Vec<(String, &codecs::grid::SizeTotals)> = vec![("ALL".into(), &scan.overall)];
    for (i, t) in scan.per_stratum.iter().enumerate() {
        groups.push((STRATUM_NAMES[i].into(), t));
    }
    for (name, t) in &groups {
        let raw = t.values.max(1) as f64; // Raw = 1 byte per value
        for (c, &b) in codecs.iter().zip(&t.bytes) {
            w(
                &mut tc,
                format!(
                    "{name},{},{},{},{},{b},{:.6},{:.6}",
                    t.rows,
                    t.values,
                    t.nonzero,
                    c.name(),
                    b as f64 / raw,
                    b as f64 / raw
                ),
            )?;
        }
        w(
            &mut tc,
            format!(
                "{name},{},{},{},nonzero_x_1B,{},{:.6},{:.6}",
                t.rows,
                t.values,
                t.nonzero,
                t.nonzero,
                t.nonzero as f64 / raw,
                t.nonzero as f64 / raw
            ),
        )?;
    }
    tc.flush().map_err(|e| e.to_string())?;

    // Markdown report.
    let mut m = String::new();
    let _ = writeln!(m, "# Size scan: every row of `{}`\n", opts.grid.display());
    m.push_str(&run_info(opts, &grid));
    m.push_str(&strata_info(&strata, &q));
    let _ = writeln!(
        m,
        "- every round trip checked (decode == input, all bytes consumed): yes, {} rows × {} codecs",
        scan.overall.rows,
        codecs.len()
    );
    let _ = writeln!(m, "- scan time: {secs:.1} s (single thread)\n");
    let o = &scan.overall;
    let _ = writeln!(
        m,
        "## Whole grid ({} rows × {} values = {} values; {} nonzero = {:.3}%)\n",
        o.rows,
        grid.n_blocks(),
        o.values,
        o.nonzero,
        100.0 * o.nonzero as f64 / o.values.max(1) as f64
    );
    let _ = writeln!(
        m,
        "| codec | bytes | MB | bytes/value | bits/value | vs Raw |"
    );
    let _ = writeln!(m, "|---|---:|---:|---:|---:|---:|");
    for (c, &b) in codecs.iter().zip(&o.bytes) {
        let bpv = b as f64 / o.values.max(1) as f64;
        let _ = writeln!(
            m,
            "| {} | {b} | {} | {bpv:.4} | {:.3} | {:.2}% |",
            c.name(),
            mb(b),
            8.0 * bpv,
            100.0 * bpv
        );
    }
    let nzv = o.nonzero as f64 / o.values.max(1) as f64;
    let _ = writeln!(
        m,
        "| *nonzero × 1 B (reference)* | {} | {} | {nzv:.4} | {:.3} | {:.2}% |",
        o.nonzero,
        mb(o.nonzero),
        8.0 * nzv,
        100.0 * nzv
    );
    let _ = writeln!(
        m,
        "\n\"nonzero × 1 B\" is a reference point (one byte per nonzero entry, positions free), not a strict lower bound: a bit-packing codec can store a value in fewer than 8 bits.\n"
    );
    let _ = writeln!(m, "## Per stratum: total MB (bytes/value)\n");
    let _ = writeln!(
        m,
        "| stratum | rows | nonzero % | {} | nonzero × 1 B |",
        names.join(" | ")
    );
    let _ = writeln!(m, "|---|---:|---:|{}---:|", "---:|".repeat(codecs.len()));
    for (i, t) in scan.per_stratum.iter().enumerate() {
        let v = t.values.max(1) as f64;
        let cells: Vec<String> = t
            .bytes
            .iter()
            .map(|&b| format!("{} ({:.4})", mb(b), b as f64 / v))
            .collect();
        let _ = writeln!(
            m,
            "| {} | {} | {:.3} | {} | {} ({:.4}) |",
            STRATUM_NAMES[i],
            t.rows,
            100.0 * t.nonzero as f64 / v,
            cells.join(" | "),
            mb(t.nonzero),
            t.nonzero as f64 / v
        );
    }
    let mut f = create_new(&md)?;
    f.write_all(m.as_bytes()).map_err(|e| e.to_string())?;
    f.flush().map_err(|e| e.to_string())?;
    print!("{m}");
    eprintln!(
        "wrote {}, {}, {}",
        md.display(),
        totals_csv.display(),
        per_term_csv.display()
    );
    Ok(())
}

// ------------------------------------------------------------------------------------
// speed mode
// ------------------------------------------------------------------------------------

/// xorshift64: tiny deterministic PRNG, so the same seed always picks the same rows.
struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // Mix the seed so small seeds (0, 1, 2) still give well-spread states; never 0.
        XorShift64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform-ish in 0..n (n > 0). The modulo bias is negligible for n << 2^64.
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Picks up to `k` items from `items` uniformly without replacement (partial
/// Fisher-Yates shuffle), returned in their original order.
fn sample_uniform(items: &[usize], k: usize, rng: &mut XorShift64) -> Vec<usize> {
    let mut v = items.to_vec();
    let k = k.min(v.len());
    for i in 0..k {
        let j = i + rng.below(v.len() - i);
        v.swap(i, j);
    }
    let mut picked = v[..k].to_vec();
    picked.sort_unstable();
    picked
}

/// Picks up to `k` distinct items without replacement, each draw proportional to its
/// weight (here: how many queries use the term), so terms that queries hit more
/// often are more likely to be timed.
fn sample_weighted(items: &[(usize, u64)], k: usize, rng: &mut XorShift64) -> Vec<usize> {
    let mut pool: Vec<(usize, u64)> = items.iter().copied().filter(|&(_, w)| w > 0).collect();
    let mut picked = Vec::new();
    while picked.len() < k && !pool.is_empty() {
        let total: u64 = pool.iter().map(|&(_, w)| w).sum();
        let mut r = rng.next() % total;
        let idx = pool
            .iter()
            .position(|&(_, w)| {
                if r < w {
                    true
                } else {
                    r -= w;
                    false
                }
            })
            .expect("r < total, so some item is chosen");
        picked.push(pool.swap_remove(idx).0);
    }
    picked.sort_unstable();
    picked
}

fn median(xs: &mut [u64]) -> u64 {
    xs.sort_unstable();
    xs[xs.len() / 2] // reps is small and usually odd; upper median for even counts
}

/// One timed (row, codec) measurement.
struct Timed {
    term: usize,
    codec: usize,
    n_values: usize,
    nonzero: usize,
    bytes: usize,
    enc_ns: u64,
    dec_ns: u64,
}

fn run_speed(opts: &Opts) -> Result<(), String> {
    let rows_csv = out_path(&opts.out, "_speed_rows.csv");
    let summary_csv = out_path(&opts.out, "_speed_summary.csv");
    let md = out_path(&opts.out, "_speed.md");
    check_outputs(&[rows_csv.clone(), summary_csv.clone(), md.clone()])?;

    let grid = Grid::open(&opts.grid).map_err(|e| e.to_string())?;
    let n_terms = grid.n_terms();
    let (strata, sb_nonzero, q) = load_strata(opts, n_terms)?;
    let queries = QueryTerms::read(&opts.queries).map_err(|e| e.to_string())?;
    // Term ids are sorted (BTreeMap), so checking the largest checks them all.
    if let Some((&t, _)) = queries.occurrences.iter().next_back()
        && t as usize >= n_terms
    {
        return Err(format!("query term id {t} >= n_terms {n_terms}"));
    }

    // ---- choose the rows; a term can belong to several groups ----
    let mut rng = XorShift64::new(opts.seed);
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for (s, name) in STRATUM_NAMES.iter().enumerate() {
        let members: Vec<usize> = (0..n_terms)
            .filter(|&t| usize::from(strata[t]) == s)
            .collect();
        groups.push((
            name.to_string(),
            sample_uniform(&members, opts.per_stratum, &mut rng),
        ));
    }
    let mut by_freq: Vec<usize> = (0..n_terms).collect();
    by_freq.sort_by_key(|&t| (std::cmp::Reverse(sb_nonzero[t]), t));
    let mut top: Vec<usize> = by_freq[..opts.top.min(n_terms)].to_vec();
    top.sort_unstable();
    groups.push((format!("top {} frequent", top.len()), top));
    let weighted: Vec<(usize, u64)> = queries
        .occurrences
        .iter()
        .map(|(&t, &c)| (t as usize, u64::from(c)))
        .collect();
    let qs = sample_weighted(&weighted, opts.query_sample, &mut rng);
    groups.push((format!("query terms ({} sampled)", qs.len()), qs));

    let mut selected: Vec<usize> = groups
        .iter()
        .flat_map(|(_, ts)| ts.iter().copied())
        .collect();
    selected.sort_unstable();
    selected.dedup();

    // ---- time each selected row (ascending term order = sequential file access) ----
    let codecs = Codec::ALL;
    let n = grid.n_blocks();
    let mut values = vec![0u16; n];
    let mut decoded = vec![0u16; n];
    let mut encoded: Vec<u8> = Vec::with_capacity(2 * n + 4096);
    let mut enc_times = vec![0u64; opts.reps];
    let mut dec_times = vec![0u64; opts.reps];
    let mut results: Vec<Timed> = Vec::new();
    let start = Instant::now();
    for (i, &term) in selected.iter().enumerate() {
        let row = grid.row(term);
        let mut nonzero = 0usize;
        for (v, &b) in values.iter_mut().zip(row) {
            *v = u16::from(b);
            nonzero += usize::from(b != 0);
        }
        grid.release_row(term);

        for (ci, &codec) in codecs.iter().enumerate() {
            // Correctness first: a timing for a wrong codec is meaningless.
            encoded.clear();
            codec.encode(&values, &mut encoded);
            let used = codec.decode(&encoded, &mut decoded);
            if used != encoded.len() || decoded != values {
                return Err(format!("{}: term {term}: round trip FAILED", codec.name()));
            }
            // Encode: 1 warm-up + reps timed runs. `black_box` stops the compiler from
            // optimizing the work away or moving it outside the timed region.
            for r in 0..=opts.reps {
                encoded.clear();
                let t0 = Instant::now();
                codec.encode(black_box(&values), &mut encoded);
                black_box(&encoded);
                let dt = t0.elapsed().as_nanos() as u64;
                if r > 0 {
                    enc_times[r - 1] = dt;
                }
            }
            // Decode the whole row: 1 warm-up + reps timed runs.
            for r in 0..=opts.reps {
                let t0 = Instant::now();
                let used = codec.decode(black_box(&encoded), &mut decoded);
                black_box(&decoded);
                black_box(used);
                let dt = t0.elapsed().as_nanos() as u64;
                if r > 0 {
                    dec_times[r - 1] = dt;
                }
            }
            results.push(Timed {
                term,
                codec: ci,
                n_values: n,
                nonzero,
                bytes: encoded.len(),
                enc_ns: median(&mut enc_times),
                dec_ns: median(&mut dec_times),
            });
        }
        if (i + 1) % 50 == 0 || i + 1 == selected.len() {
            eprintln!(
                "  {}/{} rows timed, {:.0} s",
                i + 1,
                selected.len(),
                start.elapsed().as_secs_f64()
            );
        }
    }
    let secs = start.elapsed().as_secs_f64();

    // ---- per-row CSV ----
    let mut group_of: BTreeMap<usize, Vec<&str>> = BTreeMap::new();
    for (name, ts) in &groups {
        for &t in ts {
            group_of.entry(t).or_default().push(name);
        }
    }
    let mut rc = create_new(&rows_csv)?;
    writeln!(
        rc,
        "term,groups,stratum,n_values,nonzero,codec,bytes,enc_ns_median,dec_ns_median"
    )
    .map_err(|e| e.to_string())?;
    for t in &results {
        writeln!(
            rc,
            "{},{},{},{},{},{},{},{},{}",
            t.term,
            group_of[&t.term].join(";"),
            strata[t.term],
            t.n_values,
            t.nonzero,
            codecs[t.codec].name(),
            t.bytes,
            t.enc_ns,
            t.dec_ns
        )
        .map_err(|e| e.to_string())?;
    }
    rc.flush().map_err(|e| e.to_string())?;

    // ---- summary per (group, codec): sums over rows, so long rows weigh more ----
    let mut all_group = groups.clone();
    all_group.push((
        format!("ALL sampled ({} unique rows)", selected.len()),
        selected.clone(),
    ));
    let mut sc = create_new(&summary_csv)?;
    writeln!(sc, "group,codec,rows,values,bytes,bytes_per_value,enc_ns_per_value,dec_ns_per_value,dec_mvalues_per_s").map_err(|e| e.to_string())?;
    let mut m = String::new();
    let _ = writeln!(m, "# Speed: sampled rows of `{}`\n", opts.grid.display());
    m.push_str(&run_info(opts, &grid));
    m.push_str(&strata_info(&strata, &q));
    let _ = writeln!(
        m,
        "- dev queries: {} queries, {} (term, weight) pairs, **{} unique term ids**",
        queries.n_queries,
        queries.n_pairs,
        queries.occurrences.len()
    );
    let _ = writeln!(
        m,
        "- sample (seed {}): {} per stratum, top {} by nonzero superblocks, {} query terms weighted by #queries using them → {} unique rows",
        opts.seed,
        opts.per_stratum,
        opts.top,
        opts.query_sample,
        selected.len()
    );
    let _ = writeln!(
        m,
        "- timing: per row and codec, 1 warm-up then median of {} runs; encode = one `encode` of the whole row, decode = one `decode` of the whole row; round trip checked before timing; total {secs:.1} s\n",
        opts.reps
    );
    let _ = writeln!(
        m,
        "Each cell: **decode ns/value** (encode ns/value) — bytes/value. Per group, times and bytes are summed over its rows and divided by its values.\n"
    );
    let names: Vec<&str> = codecs.iter().map(|c| c.name()).collect();
    let _ = writeln!(m, "| group | rows | {} |", names.join(" | "));
    let _ = writeln!(m, "|---|---:|{}", "---|".repeat(codecs.len()));
    for (gname, ts) in &all_group {
        let set: std::collections::BTreeSet<usize> = ts.iter().copied().collect();
        let mut cells = Vec::new();
        for (ci, c) in codecs.iter().enumerate() {
            let (mut vals, mut bytes, mut enc, mut dec) = (0u64, 0u64, 0u64, 0u64);
            for t in results
                .iter()
                .filter(|t| t.codec == ci && set.contains(&t.term))
            {
                vals += t.n_values as u64;
                bytes += t.bytes as u64;
                enc += t.enc_ns;
                dec += t.dec_ns;
            }
            let v = vals.max(1) as f64;
            let (bpv, enc_v, dec_v) = (bytes as f64 / v, enc as f64 / v, dec as f64 / v);
            let mvps = if dec_v > 0.0 { 1e3 / dec_v } else { 0.0 };
            writeln!(
                sc,
                "{gname},{},{},{vals},{bytes},{bpv:.6},{enc_v:.4},{dec_v:.4},{mvps:.1}",
                c.name(),
                set.len()
            )
            .map_err(|e| e.to_string())?;
            cells.push(format!("**{dec_v:.3}** ({enc_v:.3}) — {bpv:.4}"));
        }
        let _ = writeln!(m, "| {gname} | {} | {} |", set.len(), cells.join(" | "));
    }
    sc.flush().map_err(|e| e.to_string())?;
    let mut f = create_new(&md)?;
    f.write_all(m.as_bytes()).map_err(|e| e.to_string())?;
    f.flush().map_err(|e| e.to_string())?;
    print!("{m}");
    eprintln!(
        "wrote {}, {}, {}",
        md.display(),
        summary_csv.display(),
        rows_csv.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = match parse_args(&args) {
        Ok(o) => o,
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("error: {msg}\n");
            }
            eprint!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    if cfg!(debug_assertions) {
        eprintln!("WARNING: debug build; sizes are valid but timings are not. Use --release.");
    }
    let result = match opts.mode {
        Mode::Size => run_size(&opts),
        Mode::Speed => run_speed(&opts),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn defaults_derive_sibling_files() {
        let o = parse_args(&args("--mode speed --grid /d/x._blocks.bin --out /r/x")).unwrap();
        assert_eq!(o.strata_grid, PathBuf::from("/d/x._superblocks.bin"));
        assert_eq!(o.queries, PathBuf::from("/d/x._queries.bin"));
        assert_eq!(
            (o.reps, o.per_stratum, o.top, o.query_sample, o.seed),
            (5, 64, 32, 128, 42)
        );
    }

    #[test]
    fn bad_args_rejected() {
        assert!(parse_args(&args("--grid a._blocks.bin --out o")).is_err()); // no mode
        assert!(parse_args(&args("--mode fast --grid a._blocks.bin --out o")).is_err());
        assert!(parse_args(&args("--mode size --grid a.bin --out o")).is_err()); // can't derive
        assert!(parse_args(&args("--mode size --grid a._blocks.bin --out o --bogus 1")).is_err());
        assert!(parse_args(&args("--mode size --grid a._blocks.bin --out o --reps 0")).is_err());
        assert!(parse_args(&args("--mode size --grid a._blocks.bin --out o --reps")).is_err());
    }

    #[test]
    fn uniform_sample_is_distinct_and_deterministic() {
        let items: Vec<usize> = (0..100).collect();
        let a = sample_uniform(&items, 10, &mut XorShift64::new(7));
        let b = sample_uniform(&items, 10, &mut XorShift64::new(7));
        assert_eq!(a, b);
        let mut d = a.clone();
        d.dedup();
        assert_eq!(d.len(), 10);
        assert_eq!(
            sample_uniform(&items[..3], 10, &mut XorShift64::new(7)).len(),
            3
        );
    }

    #[test]
    fn weighted_sample_skips_zero_weight_and_is_distinct() {
        let items = vec![(1, 0), (2, 5), (3, 1), (4, 0), (5, 100)];
        let s = sample_weighted(&items, 10, &mut XorShift64::new(1));
        assert_eq!(s, vec![2, 3, 5]);
    }

    #[test]
    fn median_odd_and_even() {
        assert_eq!(median(&mut [5, 1, 3]), 3);
        assert_eq!(median(&mut [4, 1, 3, 2]), 3);
    }
}
