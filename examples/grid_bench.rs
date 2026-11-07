//! Real-data loader: runs every registry codec on the block-max grids written by
//! Parker's `export_data` (`<prefix>._blocks.bin` / `._superblocks.bin`).
//!
//! Three modes:
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
//! - `--mode stats`: distribution statistics of EVERY row (value histogram, zero runs,
//!   gaps between nonzeros, nonzeros per 128/256/512 group) and analytic size
//!   ESTIMATES for sparse formats (see `codecs::stats`). Streams one row at a time;
//!   the SimdBp128 numbers of every row are checked against the real encoder.
//!
//! Strata: each term's stratum comes from its number of nonzero entries in the
//! SUPERBLOCKS grid (so a term is in the same stratum whichever grid is measured):
//! S0 = no nonzero entry, S1-S4 = quartiles of the nonzero count among the other terms.
//!
//! Outputs (never overwritten; the program refuses to start if any exists):
//!   size:  <out>_size_totals.csv, <out>_size_per_term.csv, <out>_size.md
//!   speed: <out>_speed_rows.csv, <out>_speed_summary.csv, <out>_speed.md
//!   stats: <out>_stats.md, <out>_stats_{values,runs,groups,estimates,per_term}.csv
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
use codecs::stats::{Stats, StatsScan, bucket_range, stats_scan};

const USAGE: &str = "\
grid_bench --mode size|speed|stats --grid <._blocks.bin|._superblocks.bin> --out <prefix> [options]

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
    Stats,
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
        Some("stats") => Mode::Stats,
        Some(other) => {
            return Err(format!(
                "--mode must be size, speed or stats, not {other:?}"
            ));
        }
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

// ------------------------------------------------------------------------------------
// stats mode
// ------------------------------------------------------------------------------------

/// `100 × a / b`, or 0 when `b` is 0.
fn pct(a: f64, b: f64) -> f64 {
    if b > 0.0 { 100.0 * a / b } else { 0.0 }
}

/// Median nonzero value (lower median) from a value histogram; 0 if there is none.
fn median_nonzero(hist: &[u64; 256]) -> u8 {
    let total: u64 = hist[1..].iter().sum();
    if total == 0 {
        return 0;
    }
    let mut seen = 0;
    for (v, &c) in hist.iter().enumerate().skip(1) {
        seen += c;
        if 2 * seen >= total {
            return v as u8;
        }
    }
    255
}

/// The bucket holding the median of a power-of-2 histogram, as "lo–hi".
fn median_bucket(hist: &[u64]) -> String {
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return "n/a".into();
    }
    let mut seen = 0;
    for (k, &c) in hist.iter().enumerate() {
        seen += c;
        if 2 * seen >= total {
            let (lo, hi) = bucket_range(k);
            return if lo == hi {
                format!("{lo}")
            } else {
                format!("{lo}–{hi}")
            };
        }
    }
    "n/a".into()
}

/// The named size items reported per group of rows: (name, bytes as f64).
/// Header parts are counted in bits (4 per group), so they can be half bytes.
fn estimate_items(s: &Stats) -> Vec<(&'static str, f64)> {
    let e = &s.est;
    vec![
        ("SimdBp128 (actual codec)", e.sb128_bytes as f64),
        (
            "  headers: fields of all-zero groups",
            e.sb128_header_bits_zero_groups as f64 / 8.0,
        ),
        (
            "  headers: fields of non-empty groups",
            e.sb128_header_bits_nonempty as f64 / 8.0,
        ),
        (
            "  headers: unused slots (last header)",
            e.sb128_header_bits_unused as f64 / 8.0,
        ),
        ("  data: all non-empty groups", e.sb128_data_bytes as f64),
        (
            "  data: groups with 1 nonzero",
            e.sb128_data_bytes_by_nnz[1] as f64,
        ),
        (
            "  data: groups with 2 nonzeros",
            e.sb128_data_bytes_by_nnz[2] as f64,
        ),
        ("ESTIMATE (ii) position + value", e.posval_bytes as f64),
        (
            "ESTIMATE (iii) zero-group bitmap + SimdBp128",
            e.bitmap_bytes() as f64,
        ),
        ("  bitmap part", e.bitmap_bitmap_bytes as f64),
        (
            "  SimdBp128 on non-empty groups",
            e.bitmap_simdbp_bytes as f64,
        ),
        ("nonzero × 1 B (reference)", s.nonzero as f64),
    ]
}

fn run_stats(opts: &Opts) -> Result<(), String> {
    let md = out_path(&opts.out, "_stats.md");
    let values_csv = out_path(&opts.out, "_stats_values.csv");
    let runs_csv = out_path(&opts.out, "_stats_runs.csv");
    let groups_csv = out_path(&opts.out, "_stats_groups.csv");
    let est_csv = out_path(&opts.out, "_stats_estimates.csv");
    let per_term_csv = out_path(&opts.out, "_stats_per_term.csv");
    check_outputs(&[
        md.clone(),
        values_csv.clone(),
        runs_csv.clone(),
        groups_csv.clone(),
        est_csv.clone(),
        per_term_csv.clone(),
    ])?;

    let grid = Grid::open(&opts.grid).map_err(|e| e.to_string())?;
    let (strata, _sb_nonzero, q) = load_strata(opts, grid.n_terms())?;

    // ---- the scan: one row at a time, per-term line written as we go ----
    let mut per_term = create_new(&per_term_csv)?;
    writeln!(per_term, "term,stratum,n_values,nonzero,max,groups_128,zero_groups_128,groups_128_nnz1,groups_128_nnz2,simdbp128_bytes,posval_bytes,bitmap_bytes")
        .map_err(|e| e.to_string())?;
    let start = Instant::now();
    let n_terms = grid.n_terms();
    let mut io_error = None;
    let scan: StatsScan = stats_scan(&grid, &strata, |term, stratum, r| {
        if let Err(e) = writeln!(
            per_term,
            "{term},{stratum},{},{},{},{},{},{},{},{},{},{}",
            r.n_values,
            r.nonzero,
            r.max,
            r.groups_128,
            r.zero_groups_128,
            r.groups_128_nnz1,
            r.groups_128_nnz2,
            r.sb128_bytes,
            r.posval_bytes,
            r.bitmap_bitmap_bytes + r.bitmap_simdbp_bytes
        ) {
            io_error.get_or_insert(e);
        }
        let done = term + 1;
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

    // Groups of rows reported everywhere: the whole grid, then each stratum.
    let mut groups: Vec<(&str, &Stats)> = vec![("ALL", &scan.overall)];
    for (i, s) in scan.per_stratum.iter().enumerate() {
        groups.push((STRATUM_NAMES[i], s));
    }
    let werr = |e: std::io::Error| e.to_string();

    // ---- CSVs (long format: one line per number) ----
    let mut c = create_new(&values_csv)?;
    writeln!(c, "group,value,count").map_err(werr)?;
    for (name, s) in &groups {
        for (v, &n) in s.value_hist.iter().enumerate() {
            writeln!(c, "{name},{v},{n}").map_err(werr)?;
        }
    }
    c.flush().map_err(werr)?;

    let mut c = create_new(&runs_csv)?;
    writeln!(c, "group,kind,bucket,len_lo,len_hi,count").map_err(werr)?;
    for (name, s) in &groups {
        for (kind, hist) in [("zero_run", &s.zero_run_hist), ("gap", &s.gap_hist)] {
            for (k, &n) in hist.iter().enumerate() {
                if n > 0 {
                    let (lo, hi) = bucket_range(k);
                    writeln!(c, "{name},{kind},{k},{lo},{hi},{n}").map_err(werr)?;
                }
            }
        }
    }
    c.flush().map_err(werr)?;

    let mut c = create_new(&groups_csv)?;
    writeln!(c, "group,group_len,nonzeros_in_group,count").map_err(werr)?;
    for (name, s) in &groups {
        for g in &s.groups {
            for (k, &n) in g.nnz_hist.iter().enumerate() {
                if n > 0 {
                    writeln!(c, "{name},{},{k},{n}", g.group_len).map_err(werr)?;
                }
            }
        }
    }
    c.flush().map_err(werr)?;

    let mut c = create_new(&est_csv)?;
    writeln!(c, "group,item,bytes,pct_of_raw,bits_per_nonzero").map_err(werr)?;
    for (name, s) in &groups {
        for (item, b) in estimate_items(s) {
            writeln!(
                c,
                "{name},{},{b:.1},{:.4},{:.3}",
                item.trim(),
                pct(b, s.values as f64),
                if s.nonzero > 0 {
                    8.0 * b / s.nonzero as f64
                } else {
                    0.0
                }
            )
            .map_err(werr)?;
        }
    }
    c.flush().map_err(werr)?;

    // ---- Markdown report ----
    let mut m = String::new();
    let o = &scan.overall;
    let _ = writeln!(
        m,
        "# Distribution stats: every row of `{}`\n",
        opts.grid.display()
    );
    m.push_str(&run_info(opts, &grid));
    m.push_str(&strata_info(&strata, &q));
    let _ = writeln!(
        m,
        "- every row's SimdBp128 bytes (and the bitmap estimate's SimdBp128 part) checked against the real encoder: yes, {} rows",
        o.rows
    );
    let _ = writeln!(m, "- scan time: {secs:.1} s (single thread)");
    let _ = writeln!(
        m,
        "- whole grid: {} rows × {} values = {} values, {} nonzero ({:.3}%)\n",
        o.rows,
        grid.n_blocks(),
        o.values,
        o.nonzero,
        pct(o.nonzero as f64, o.values as f64)
    );

    let _ = writeln!(m, "## 1. Nonzero values\n");
    let _ = writeln!(
        m,
        "Bit width = bits needed to store the value (1 → 1, 2–3 → 2, …, 128–255 → 8). Columns w1–w8: % of nonzero entries.\n"
    );
    let _ = writeln!(
        m,
        "| group | rows | nonzero % | mean | median | w1 | w2 | w3 | w4 | w5 | w6 | w7 | w8 |"
    );
    let _ = writeln!(
        m,
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    );
    for (name, s) in &groups {
        let nz = s.nonzero as f64;
        let sum: f64 = s
            .value_hist
            .iter()
            .enumerate()
            .map(|(v, &c)| v as f64 * c as f64)
            .sum();
        let mut widths = [0u64; 9];
        for (v, &c) in s.value_hist.iter().enumerate().skip(1) {
            widths[codecs::stats::bit_width(v as u8) as usize] += c;
        }
        let w: Vec<String> = (1..=8)
            .map(|k| format!("{:.1}", pct(widths[k] as f64, nz)))
            .collect();
        let _ = writeln!(
            m,
            "| {name} | {} | {:.3} | {:.1} | {} | {} |",
            s.rows,
            pct(nz, s.values as f64),
            if nz > 0.0 { sum / nz } else { 0.0 },
            median_nonzero(&s.value_hist),
            w.join(" | ")
        );
    }

    let _ = writeln!(m, "\n## 2. Zero runs and gaps\n");
    let _ = writeln!(
        m,
        "Zero run = maximal stretch of zeros (including at the start/end of a row; an all-zero row is one run). Gap = distance between consecutive nonzeros (adjacent = 1).\n"
    );
    let _ = writeln!(m, "Whole grid, by length:\n");
    let _ = writeln!(m, "| length | zero runs | % of runs | gaps | % of gaps |");
    let _ = writeln!(m, "|---|---:|---:|---:|---:|");
    let (runs, gaps) = (o.zero_runs() as f64, o.gaps() as f64);
    for k in 0..o.zero_run_hist.len() {
        let (r, g) = (o.zero_run_hist[k], o.gap_hist[k]);
        if r == 0 && g == 0 {
            continue;
        }
        let (lo, hi) = bucket_range(k);
        let label = if lo == hi {
            format!("{lo}")
        } else {
            format!("{lo}–{hi}")
        };
        let _ = writeln!(
            m,
            "| {label} | {r} | {:.2} | {g} | {:.2} |",
            pct(r as f64, runs),
            pct(g as f64, gaps)
        );
    }
    let _ = writeln!(m, "\nPer stratum:\n");
    let _ = writeln!(
        m,
        "| group | zero runs | mean run length | median run | gaps | gap = 1 % | gap ≤ 7 % | median gap |"
    );
    let _ = writeln!(m, "|---|---:|---:|---|---:|---:|---:|---|");
    for (name, s) in &groups {
        let (r, g) = (s.zero_runs(), s.gaps());
        let le7: u64 = s.gap_hist[..3].iter().sum();
        let _ = writeln!(
            m,
            "| {name} | {r} | {:.1} | {} | {g} | {:.2} | {:.2} | {} |",
            if r > 0 {
                s.zero_run_total as f64 / r as f64
            } else {
                0.0
            },
            median_bucket(&s.zero_run_hist),
            pct(s.gap_hist[0] as f64, g as f64),
            pct(le7 as f64, g as f64),
            median_bucket(&s.gap_hist)
        );
    }

    let _ = writeln!(m, "\n## 3. Groups of 128 / 256 / 512 values\n");
    let _ = writeln!(
        m,
        "Groups are aligned at 0 (the last one in a row may be shorter). The nz columns are % of NON-EMPTY groups.\n"
    );
    let _ = writeln!(
        m,
        "| group | G | groups | all-zero % | non-empty | 1 nz % | 2 nz % | 1–2 nz % | ≤ 4 nz % | ≤ 16 nz % | mean nz / non-empty |"
    );
    let _ = writeln!(m, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for (name, s) in &groups {
        for g in &s.groups {
            let ne = g.nonempty_groups() as f64;
            let _ = writeln!(
                m,
                "| {name} | {} | {} | {:.2} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |",
                g.group_len,
                g.groups(),
                pct(g.zero_groups() as f64, g.groups() as f64),
                g.nonempty_groups(),
                pct(g.nonempty_with(1, 1) as f64, ne),
                pct(g.nonempty_with(2, 2) as f64, ne),
                pct(g.nonempty_with(1, 2) as f64, ne),
                pct(g.nonempty_with(1, 4) as f64, ne),
                pct(g.nonempty_with(1, 16) as f64, ne),
                if ne > 0.0 { s.nonzero as f64 / ne } else { 0.0 }
            );
        }
    }

    let _ = writeln!(
        m,
        "\n## 4. Sizes: SimdBp128 breakdown and ESTIMATES (not implemented codecs)\n"
    );
    let _ = writeln!(
        m,
        "SimdBp128 is the real codec (its total equals the encoder's output, checked per row); its split is computed from the group widths. \
         (ii) and (iii) are ESTIMATES: byte counts of formats that do not exist as code (no decoder, no speed). \
         (ii) = per row Elias-γ(nonzero+1) + per nonzero Elias-γ(gap; first = position+1) + 8 bits, rounded up to bytes per row. \
         (iii) = per row ⌈groups/8⌉ bytes of 1-bit-per-128-group bitmap + SimdBp128 of the non-empty groups only.\n"
    );
    let _ = writeln!(m, "Whole grid:\n");
    let _ = writeln!(m, "| item | bytes | % of Raw | bits / nonzero |");
    let _ = writeln!(m, "|---|---:|---:|---:|");
    for (item, b) in estimate_items(o) {
        let _ = writeln!(
            m,
            "| {} | {b:.1} | {:.3}% | {:.2} |",
            item.replace("  ", "↳ "),
            pct(b, o.values as f64),
            if o.nonzero > 0 {
                8.0 * b / o.nonzero as f64
            } else {
                0.0
            }
        );
    }
    let _ = writeln!(m, "\nPer stratum: % of Raw (bits / nonzero)\n");
    let _ = writeln!(
        m,
        "| group | SimdBp128 | (ii) position + value | (iii) bitmap + SimdBp128 | nonzero × 1 B | SimdBp128 data in 1–2 nz groups, % of its bytes |"
    );
    let _ = writeln!(m, "|---|---:|---:|---:|---:|---:|");
    for (name, s) in &groups {
        let (v, nz) = (s.values as f64, s.nonzero as f64);
        let cell = |b: f64| {
            format!(
                "{:.3}% ({:.1})",
                pct(b, v),
                if nz > 0.0 { 8.0 * b / nz } else { 0.0 }
            )
        };
        let sparse = (s.est.sb128_data_bytes_by_nnz[1] + s.est.sb128_data_bytes_by_nnz[2]) as f64;
        let _ = writeln!(
            m,
            "| {name} | {} | {} | {} | {} | {:.1}% |",
            cell(s.est.sb128_bytes as f64),
            cell(s.est.posval_bytes as f64),
            cell(s.est.bitmap_bytes() as f64),
            cell(nz),
            pct(sparse, s.est.sb128_bytes as f64)
        );
    }

    let mut f = create_new(&md)?;
    f.write_all(m.as_bytes()).map_err(werr)?;
    f.flush().map_err(werr)?;
    print!("{m}");
    eprintln!(
        "wrote {}, {}, {}, {}, {}, {}",
        md.display(),
        values_csv.display(),
        runs_csv.display(),
        groups_csv.display(),
        est_csv.display(),
        per_term_csv.display()
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
        Mode::Stats => run_stats(&opts),
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
    fn stats_mode_parses() {
        let o = parse_args(&args("--mode stats --grid /d/x._blocks.bin --out /r/x")).unwrap();
        assert_eq!(o.mode, Mode::Stats);
        assert_eq!(o.strata_grid, PathBuf::from("/d/x._superblocks.bin"));
    }

    #[test]
    fn stats_medians() {
        let mut h = [0u64; 256];
        assert_eq!(median_nonzero(&h), 0); // no nonzero values
        h[0] = 1000; // zeros are ignored
        h[3] = 2;
        h[10] = 1;
        h[200] = 2;
        assert_eq!(median_nonzero(&h), 10); // 3,3,10,200,200
        h[10] = 0;
        assert_eq!(median_nonzero(&h), 3); // lower median of 3,3,200,200
        // power-of-2 buckets: 1 in bucket 0, 3 in bucket 2 (4..=7)
        assert_eq!(median_bucket(&[1, 0, 3]), "4–7");
        assert_eq!(median_bucket(&[3, 0, 1]), "1");
        assert_eq!(median_bucket(&[0, 0]), "n/a");
        assert_eq!(pct(1.0, 0.0), 0.0);
    }

    #[test]
    fn median_odd_and_even() {
        assert_eq!(median(&mut [5, 1, 3]), 3);
        assert_eq!(median(&mut [4, 1, 3, 2]), 3);
    }
}
