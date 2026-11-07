//! Tests for `codecs::stats` (grid_bench --mode stats).
//!
//! 1. Hand-computed rows: every histogram bucket and byte count worked out on paper
//!    (the arithmetic is in the comments).
//! 2. Random rows: the analytic SimdBp128 numbers checked against the real encoder, and
//!    runs / gaps / position+value bits checked against an independent brute-force model.
//! 3. Merging strata = one scan over all rows; `stats_scan` on a synthetic grid file.
//! 4. End-to-end on the smoke grid (SKIPPED if missing; `GRID_SMOKE_REQUIRE=1` makes it fail).

use std::path::PathBuf;

use codecs::grid::{Grid, N_STRATA, assign_strata, nonzero_per_row};
use codecs::stats::{
    RowSummary, Stats, bit_width, bucket_range, check_row_estimates, gamma_bits, log2_bucket,
    stats_scan,
};

fn stats_of(row: &[u8]) -> (Stats, RowSummary) {
    let mut s = Stats::new();
    let r = s.add_row(row);
    (s, r)
}

/// Asserts the analytic SimdBp128 numbers of `row` equal the real encoder's output.
fn assert_matches_encoder(row: &[u8], r: &RowSummary) {
    let (mut v, mut o) = (Vec::new(), Vec::new());
    check_row_estimates(row, r, &mut v, &mut o).unwrap();
}

#[test]
fn helpers() {
    assert_eq!(log2_bucket(1), 0);
    assert_eq!(log2_bucket(2), 1);
    assert_eq!(log2_bucket(3), 1);
    assert_eq!(log2_bucket(4), 2);
    assert_eq!(log2_bucket(1 << 20), 20);
    assert_eq!(bucket_range(0), (1, 1));
    assert_eq!(bucket_range(3), (8, 15));
    // Elias-γ: 1 → "1" (1 bit), 2 → "010" (3), 4 → "00100" (5), 130 → 2·7+1 = 15
    assert_eq!([1, 2, 3, 4, 130].map(gamma_bits), [1, 3, 3, 5, 15]);
    assert_eq!(
        [0u8, 1, 2, 3, 7, 8, 127, 128, 255].map(bit_width),
        [0, 1, 2, 2, 3, 4, 7, 8, 8]
    );
}

#[test]
fn small_row_by_hand() {
    // positions:  0  1  2  3  4  5  6  7  8  9
    let row = [0u8, 5, 0, 0, 7, 7, 0, 0, 0, 1];
    let (s, r) = stats_of(&row);
    assert_eq!((s.rows, s.values, s.nonzero), (1, 10, 4));
    assert_eq!(s.value_hist[0], 6);
    assert_eq!(
        (s.value_hist[1], s.value_hist[5], s.value_hist[7]),
        (1, 1, 2)
    );
    assert_eq!(s.value_hist.iter().sum::<u64>(), 10);
    // zero runs: [0] len 1 → bucket 0; [2,3] len 2 and [6..8] len 3 → bucket 1; none at the end
    assert_eq!(&s.zero_run_hist[..3], &[1, 2, 0]);
    assert_eq!(s.zero_runs(), 3);
    assert_eq!(s.zero_run_total, 6);
    // gaps: 4-1 = 3 (bucket 1), 5-4 = 1 (bucket 0), 9-5 = 4 (bucket 2)
    assert_eq!(&s.gap_hist[..4], &[1, 1, 1, 0]);
    // position+value: γ(4+1)=5; gaps γ(1+1)=3, γ(3)=3, γ(1)=1, γ(4)=5 → 12; values 4·8 = 32
    // total 49 bits → 7 bytes
    assert_eq!(s.est.posval_bits, 49);
    assert_eq!(r.posval_bytes, 7);
    // one (partial) group of every size, with 4 nonzeros
    for g in &s.groups {
        assert_eq!((g.groups(), g.zero_groups(), g.nnz_hist[4]), (1, 0, 1));
    }
    // SimdBp128: 1 header (64 B) + 1 group at width 3 (max 7) = 16·3 = 48 → 112 B
    assert_eq!(r.sb128_bytes, 112);
    assert_eq!(s.est.sb128_data_bytes, 48);
    assert_eq!(s.est.sb128_data_bytes_by_nnz[4], 48);
    assert_eq!(s.est.sb128_header_bits_nonempty, 4);
    assert_eq!(s.est.sb128_header_bits_zero_groups, 0);
    assert_eq!(s.est.sb128_header_bits_unused, 512 - 4);
    // bitmap: ⌈1/8⌉ = 1 B + SimdBp128 of the one non-empty group (112) = 113
    assert_eq!((r.bitmap_bitmap_bytes, r.bitmap_simdbp_bytes), (1, 112));
    assert_eq!(s.est.bitmap_bytes(), 113);
    assert_matches_encoder(&row, &r);
}

#[test]
fn all_zero_row() {
    let row = vec![0u8; 300];
    let (s, r) = stats_of(&row);
    assert_eq!((s.nonzero, s.value_hist[0]), (0, 300));
    // one zero run of 300 → bucket 8 (256..=511); no gaps
    assert_eq!(s.zero_run_hist[8], 1);
    assert_eq!(s.zero_runs(), 1);
    assert_eq!(s.gaps(), 0);
    // groups: 128 → 3 (128, 128, 44), 256 → 2, 512 → 1; all empty
    let zero: Vec<(u64, u64)> = s
        .groups
        .iter()
        .map(|g| (g.groups(), g.zero_groups()))
        .collect();
    assert_eq!(zero, vec![(3, 3), (2, 2), (1, 1)]);
    // SimdBp128: just one header (64 B); 3 zero-group fields = 12 bits, 500 unused
    assert_eq!(r.sb128_bytes, 64);
    assert_eq!(s.est.sb128_header_bits_zero_groups, 12);
    assert_eq!(s.est.sb128_header_bits_unused, 500);
    // position+value: only γ(0+1) = 1 bit → 1 byte
    assert_eq!((s.est.posval_bits, r.posval_bytes), (1, 1));
    // bitmap: ⌈3/8⌉ = 1 B and nothing to encode
    assert_eq!((r.bitmap_bitmap_bytes, r.bitmap_simdbp_bytes), (1, 0));
    assert_matches_encoder(&row, &r);
}

#[test]
fn sparse_groups_by_hand() {
    // n = 600: 255 at 0, then 1 at 130 and 2 at 131
    let mut row = vec![0u8; 600];
    row[0] = 255;
    row[130] = 1;
    row[131] = 2;
    let (s, r) = stats_of(&row);
    // zero runs: 1..=129 (129, bucket 7), 132..=599 (468, bucket 8)
    assert_eq!(
        (s.zero_run_hist[7], s.zero_run_hist[8], s.zero_runs()),
        (1, 1, 2)
    );
    assert_eq!(s.zero_run_total, 597);
    // gaps: 130 (bucket 7), 1 (bucket 0)
    assert_eq!((s.gap_hist[7], s.gap_hist[0], s.gaps()), (1, 1, 2));
    // position+value: γ(4)=5 + γ(0+1)=1 + γ(130)=15 + γ(1)=1 + 3·8 = 46 bits → 6 B
    assert_eq!((s.est.posval_bits, r.posval_bytes), (46, 6));
    // 128-groups: [0,128) 1 nz, [128,256) 2 nz, then 3 empty (the last one is 88 long)
    let g128 = &s.groups[0];
    assert_eq!(
        (g128.nnz_hist[0], g128.nnz_hist[1], g128.nnz_hist[2]),
        (3, 1, 1)
    );
    assert_eq!(
        (
            g128.nonempty_with(1, 1),
            g128.nonempty_with(1, 2),
            g128.nonempty_with(3, 128)
        ),
        (1, 2, 0)
    );
    // 256-groups: [0,256) 3 nz, two empty; 512-groups: [0,512) 3 nz, one empty
    assert_eq!((s.groups[1].nnz_hist[0], s.groups[1].nnz_hist[3]), (2, 1));
    assert_eq!((s.groups[2].nnz_hist[0], s.groups[2].nnz_hist[3]), (1, 1));
    assert_eq!(
        (
            r.groups_128,
            r.zero_groups_128,
            r.groups_128_nnz1,
            r.groups_128_nnz2
        ),
        (5, 3, 1, 1)
    );
    // SimdBp128: 64 + 16·8 (max 255) + 16·2 (max 2) = 224 B; the 1-nonzero group alone costs 128 B
    assert_eq!(r.sb128_bytes, 224);
    assert_eq!(
        (
            s.est.sb128_data_bytes_by_nnz[1],
            s.est.sb128_data_bytes_by_nnz[2]
        ),
        (128, 32)
    );
    // bitmap: ⌈5/8⌉ = 1 + (64 + 160) = 225: dropping all-zero groups saves nothing here,
    // because in SIMDBP an all-zero group only costs its 4-bit header field
    assert_eq!(s.est.bitmap_bytes(), 225);
    assert_matches_encoder(&row, &r);
}

#[test]
fn two_metablocks() {
    // 128·128 + 1 values = 129 groups → 2 headers (128 B) + one width-1 group (16 B)
    let mut row = vec![0u8; 128 * 128 + 1];
    *row.last_mut().unwrap() = 1;
    let (s, r) = stats_of(&row);
    assert_eq!(r.sb128_bytes, 144);
    assert_eq!(s.est.sb128_header_bits_unused, 2 * 512 - 4 * 129);
    // bitmap: ⌈129/8⌉ = 17 + one header for the single non-empty group + 16 = 97
    assert_eq!(s.est.bitmap_bytes(), 17 + 64 + 16);
    assert_matches_encoder(&row, &r);
}

#[test]
fn empty_row() {
    let (s, r) = stats_of(&[]);
    assert_eq!((s.rows, s.values, s.zero_runs(), s.gaps()), (1, 0, 0, 0));
    assert_eq!(
        (
            r.sb128_bytes,
            r.posval_bytes,
            r.bitmap_bitmap_bytes,
            r.bitmap_simdbp_bytes
        ),
        (0, 0, 0, 0)
    );
    assert!(s.groups.iter().all(|g| g.groups() == 0));
    assert_matches_encoder(&[], &r);
}

/// Small deterministic PRNG for the random tests.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// Rows of many lengths and densities, including clustered nonzeros.
fn random_rows() -> Vec<Vec<u8>> {
    let mut rng = Lcg(12345);
    let mut rows = Vec::new();
    for &n in &[
        1usize, 2, 127, 128, 129, 300, 1000, 16383, 16384, 16385, 40000,
    ] {
        for &per_mille in &[0u64, 1, 10, 100, 500, 1000] {
            let row: Vec<u8> = (0..n)
                .map(|_| {
                    if rng.next() % 1000 < per_mille {
                        (rng.next() % 255 + 1) as u8
                    } else {
                        0
                    }
                })
                .collect();
            rows.push(row);
        }
        // clustered: dense bursts of 20 every ~3000 values
        let row: Vec<u8> = (0..n)
            .map(|i| {
                if i % 3000 < 20 {
                    (i % 200 + 1) as u8
                } else {
                    0
                }
            })
            .collect();
        rows.push(row);
    }
    rows
}

/// Independent model: positions of the nonzeros first, then runs and gaps from them.
fn brute_force(row: &[u8]) -> (Vec<u64>, Vec<u64>, u64) {
    let pos: Vec<usize> = (0..row.len()).filter(|&i| row[i] != 0).collect();
    // zero runs = stretches between "fences" at -1, each nonzero position, and n
    let mut fences: Vec<i64> = vec![-1];
    fences.extend(pos.iter().map(|&p| p as i64));
    fences.push(row.len() as i64);
    let runs: Vec<u64> = fences
        .windows(2)
        .map(|w| (w[1] - w[0] - 1) as u64)
        .filter(|&l| l > 0)
        .collect();
    let gaps: Vec<u64> = pos.windows(2).map(|w| (w[1] - w[0]) as u64).collect();
    let mut bits = 0u64;
    if !row.is_empty() {
        bits += gamma_bits(pos.len() as u64 + 1);
        let mut prev: i64 = -1;
        for &p in &pos {
            bits += gamma_bits((p as i64 - prev) as u64) + 8;
            prev = p as i64;
        }
    }
    (runs, gaps, bits)
}

#[test]
fn random_rows_match_encoder_and_model() {
    for row in random_rows() {
        let (s, r) = stats_of(&row);
        assert_matches_encoder(&row, &r);
        let (runs, gaps, bits) = brute_force(&row);
        let mut run_hist = [0u64; 41];
        runs.iter().for_each(|&l| run_hist[log2_bucket(l)] += 1);
        let mut gap_hist = [0u64; 41];
        gaps.iter().for_each(|&g| gap_hist[log2_bucket(g)] += 1);
        assert_eq!(s.zero_run_hist, run_hist, "n={}", row.len());
        assert_eq!(s.gap_hist, gap_hist, "n={}", row.len());
        assert_eq!(s.zero_run_total, runs.iter().sum::<u64>());
        assert_eq!(s.est.posval_bits, bits);
        assert_eq!(s.est.posval_bytes, bits.div_ceil(8));
        // every group size: groups and nonzeros add up
        for g in &s.groups {
            assert_eq!(g.groups() as usize, row.len().div_ceil(g.group_len));
            let nz: u64 = g
                .nnz_hist
                .iter()
                .enumerate()
                .map(|(c, &k)| c as u64 * k)
                .sum();
            assert_eq!(nz, s.nonzero);
        }
        // SimdBp128 split adds up: header bits = 8 × header bytes, data by nnz = data
        let e = &s.est;
        let header_bits = e.sb128_header_bits_zero_groups
            + e.sb128_header_bits_nonempty
            + e.sb128_header_bits_unused;
        assert_eq!(header_bits % 8, 0);
        assert_eq!(header_bits / 8 + e.sb128_data_bytes, e.sb128_bytes);
        assert_eq!(
            e.sb128_data_bytes_by_nnz.iter().sum::<u64>(),
            e.sb128_data_bytes
        );
    }
}

#[test]
fn merge_equals_one_scan() {
    let rows = random_rows();
    let mut all = Stats::new();
    let mut a = Stats::new();
    let mut b = Stats::new();
    for (i, row) in rows.iter().enumerate() {
        all.add_row(row);
        if i % 3 == 0 {
            a.add_row(row)
        } else {
            b.add_row(row)
        };
    }
    let mut merged = Stats::new();
    merged.merge(&a);
    merged.merge(&b);
    assert_eq!(merged, all);
}

fn write_grid(name: &str, rows: &[Vec<u8>]) -> PathBuf {
    let n_blocks = rows.first().map_or(0, Vec::len);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(n_blocks as u32).to_le_bytes());
    for r in rows {
        bytes.extend_from_slice(r);
    }
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn stats_scan_on_a_grid_file() {
    // 40 rows of 1000 values, with the same mix as random_rows() but one length
    let rows: Vec<Vec<u8>> = random_rows()
        .into_iter()
        .filter(|r| r.len() == 1000)
        .cycle()
        .take(40)
        .collect();
    let path = write_grid("stats_scan_test._blocks.bin", &rows);
    let grid = Grid::open(&path).unwrap();
    let (strata, _) = assign_strata(&nonzero_per_row(&grid));
    let mut seen = 0;
    let scan = stats_scan(&grid, &strata, |term, stratum, r| {
        assert_eq!(stratum, strata[term]);
        assert_eq!(r.nonzero, rows[term].iter().filter(|&&v| v != 0).count());
        seen += 1;
    });
    assert_eq!(seen, 40);
    assert_eq!(scan.per_stratum.len(), N_STRATA);
    let mut direct = Stats::new();
    rows.iter().for_each(|r| {
        direct.add_row(r);
    });
    assert_eq!(scan.overall, direct);
    let rows_by_stratum: u64 = scan.per_stratum.iter().map(|s| s.rows).sum();
    assert_eq!(rows_by_stratum, 40);
}

#[test]
fn smoke_grid_end_to_end() {
    let prefix = std::env::var("GRID_SMOKE_PREFIX").unwrap_or_else(|_| {
        format!(
            "{}/data/smoke_128_8",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let blocks = PathBuf::from(format!("{prefix}._blocks.bin"));
    let supers = PathBuf::from(format!("{prefix}._superblocks.bin"));
    if !blocks.exists() || !supers.exists() {
        assert!(
            std::env::var("GRID_SMOKE_REQUIRE").is_err(),
            "GRID_SMOKE_REQUIRE is set but {prefix}._*.bin is missing"
        );
        eprintln!("SKIPPED: smoke grid {prefix}._blocks.bin not found");
        return;
    }
    let grid = Grid::open(&blocks).unwrap();
    let (strata, _) = assign_strata(&nonzero_per_row(&Grid::open(&supers).unwrap()));
    let scan = stats_scan(&grid, &strata, |_, _, _| {}); // checks every row against the encoder
    let o = &scan.overall;
    let values = (grid.n_terms() * grid.n_blocks()) as u64;
    assert_eq!(o.values, values);
    assert_eq!(o.value_hist.iter().sum::<u64>(), values);
    assert_eq!(o.zero_run_total, values - o.nonzero);
    let rows_with_nz: u64 = (0..grid.n_terms())
        .filter(|&t| grid.row(t).iter().any(|&v| v != 0))
        .count() as u64;
    assert_eq!(o.gaps(), o.nonzero - rows_with_nz);
    let direct_sb128: u64 = (0..grid.n_terms())
        .map(|t| {
            let v: Vec<u16> = grid.row(t).iter().map(|&b| u16::from(b)).collect();
            let mut out = Vec::new();
            codecs::Codec::SimdBp128.encode(&v, &mut out);
            out.len() as u64
        })
        .sum();
    assert_eq!(o.est.sb128_bytes, direct_sb128);
}
