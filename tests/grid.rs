//! Tests for the real-data loader (`codecs::grid`).
//!
//! 1. Synthetic grids written to cargo's per-crate temp folder (`target/tmp`), so the
//!    expected answers are known exactly.
//! 2. End-to-end on the SMOKE grid produced by `~/lsp-work/scripts/smoke_test.sh`
//!    (nfcorpus: 15,572 terms, 3,633 docs, `--bsize 128 8`), default prefix
//!    `$HOME/data/smoke_128_8`, override with `GRID_SMOKE_PREFIX`. If the files are
//!    missing the test prints SKIPPED; set `GRID_SMOKE_REQUIRE=1` to make that a failure.

use std::path::PathBuf;

use codecs::Codec;
use codecs::grid::{Grid, N_STRATA, QueryTerms, assign_strata, nonzero_per_row, size_scan};

/// Writes a grid file (header + rows) into cargo's temp folder and returns its path.
fn write_grid(name: &str, rows: &[Vec<u8>]) -> PathBuf {
    let n_blocks = rows.first().map_or(0, Vec::len);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(n_blocks as u32).to_le_bytes());
    for r in rows {
        assert_eq!(r.len(), n_blocks);
        bytes.extend_from_slice(r);
    }
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

/// A few rows with different shapes: all zero, sparse, dense, max values, a ramp.
fn synthetic_rows(n_blocks: usize) -> Vec<Vec<u8>> {
    let mut rows = vec![vec![0u8; n_blocks]];
    rows.push(
        (0..n_blocks)
            .map(|i| if i % 97 == 0 { (i % 251) as u8 + 1 } else { 0 })
            .collect(),
    );
    rows.push((0..n_blocks).map(|i| (i * 31 % 256) as u8).collect());
    rows.push(vec![255u8; n_blocks]);
    rows.push((0..n_blocks).map(|i| (i % 64) as u8).collect());
    rows
}

#[test]
fn grid_open_reads_rows_exactly() {
    let rows = synthetic_rows(1000);
    let path = write_grid("grid_open.bin", &rows);
    let g = Grid::open(&path).unwrap();
    assert_eq!((g.n_terms(), g.n_blocks()), (5, 1000));
    for (t, r) in rows.iter().enumerate() {
        assert_eq!(g.row(t), &r[..], "row {t}");
        g.release_row(t);
        assert_eq!(
            g.row(t),
            &r[..],
            "row {t} after release_row (must re-read the same bytes)"
        );
    }
}

#[test]
fn grid_open_rejects_wrong_length() {
    let path = write_grid("grid_bad.bin", &synthetic_rows(100));
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.pop(); // one byte short
    let short = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("grid_short.bin");
    std::fs::write(&short, &bytes).unwrap();
    let e = Grid::open(&short)
        .err()
        .expect("truncated grid must be rejected")
        .to_string();
    assert!(e.contains("should be"), "{e}");
    assert!(
        Grid::open(&PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("no_such_grid.bin")).is_err()
    );
}

#[test]
#[should_panic(expected = "out of range")]
fn grid_row_out_of_range_panics() {
    let g = Grid::open(&write_grid("grid_oob.bin", &synthetic_rows(10))).unwrap();
    let _ = g.row(5);
}

#[test]
fn size_scan_matches_direct_encoding() {
    // 70,000 values per row: more than one SIMDBP512 metablock (512 × 512 = 262,144 is
    // one metablock, so also test a long row below) and a partial last group for all.
    for (name, n) in [("scan_a.bin", 70_000usize), ("scan_b.bin", 300_001)] {
        let rows = synthetic_rows(n);
        let g = Grid::open(&write_grid(name, &rows)).unwrap();
        let strata = vec![0u8, 1, 2, 3, 4];
        let mut seen = Vec::new();
        let scan = size_scan(&g, &strata, &Codec::ALL, |r| seen.push(r.clone()));
        assert_eq!(seen.len(), rows.len());
        for (t, r) in rows.iter().enumerate() {
            let values: Vec<u16> = r.iter().map(|&b| u16::from(b)).collect();
            for (ci, codec) in Codec::ALL.iter().enumerate() {
                let mut enc = Vec::new();
                codec.encode(&values, &mut enc);
                assert_eq!(seen[t].bytes[ci], enc.len(), "{} row {t}", codec.name());
            }
            assert_eq!(seen[t].nonzero, r.iter().filter(|&&b| b != 0).count());
            assert_eq!(seen[t].max, *r.iter().max().unwrap());
        }
        // Raw = 1 byte per value, and the per-stratum totals add up to the overall one.
        let raw = Codec::ALL.iter().position(|&c| c == Codec::Raw).unwrap();
        assert_eq!(scan.overall.bytes[raw], (rows.len() * n) as u64);
        for ci in 0..Codec::ALL.len() {
            let sum: u64 = scan.per_stratum.iter().map(|s| s.bytes[ci]).sum();
            assert_eq!(sum, scan.overall.bytes[ci]);
        }
        assert_eq!(
            scan.per_stratum.iter().map(|s| s.rows).sum::<u64>(),
            rows.len() as u64
        );
    }
}

// ------------------------------------------------------------------------------------
// End-to-end on the smoke grid
// ------------------------------------------------------------------------------------

fn smoke_prefix() -> Option<String> {
    let prefix = std::env::var("GRID_SMOKE_PREFIX").unwrap_or_else(|_| {
        format!(
            "{}/data/smoke_128_8",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let present = ["._blocks.bin", "._superblocks.bin", "._queries.bin"]
        .iter()
        .all(|s| std::path::Path::new(&format!("{prefix}{s}")).exists());
    if present {
        return Some(prefix);
    }
    if std::env::var("GRID_SMOKE_REQUIRE").as_deref() == Ok("1") {
        panic!("smoke grid {prefix}.* not found and GRID_SMOKE_REQUIRE=1");
    }
    eprintln!(
        "SKIPPED: smoke grid {prefix}.* not found (run smoke_test.sh, or set GRID_SMOKE_PREFIX)"
    );
    None
}

#[test]
fn smoke_grid_end_to_end() {
    let Some(prefix) = smoke_prefix() else { return };
    let blocks = Grid::open(&PathBuf::from(format!("{prefix}._blocks.bin"))).unwrap();
    let supers = Grid::open(&PathBuf::from(format!("{prefix}._superblocks.bin"))).unwrap();

    // Shapes from NOTES.md: 15,572 terms; ceil(3633/8) = 455; ceil(3633/128) = 29.
    assert_eq!((blocks.n_terms(), blocks.n_blocks()), (15_572, 455));
    assert_eq!((supers.n_terms(), supers.n_blocks()), (15_572, 29));

    // Consistency between the two levels, independent of any codec: a superblock covers
    // 128 / 8 = 16 blocks, so its max must equal the max of those 16 block maxes.
    let per = 128 / 8;
    let mut nonempty = 0;
    for t in 0..blocks.n_terms() {
        let (b, s) = (blocks.row(t), supers.row(t));
        for (si, &smax) in s.iter().enumerate() {
            let lo = si * per;
            let hi = ((si + 1) * per).min(b.len());
            let bmax = b[lo..hi].iter().copied().max().unwrap_or(0);
            assert_eq!(smax, bmax, "term {t} superblock {si}");
        }
        nonempty += usize::from(s.iter().any(|&x| x != 0));
    }
    assert!(nonempty > 0, "every row is zero: not a real grid");

    // Strata from the superblocks; the size scan of the blocks grid round-trips every
    // row through every codec (size_scan panics on any mismatch).
    let sb_nonzero = nonzero_per_row(&supers);
    let (strata, q) = assign_strata(&sb_nonzero);
    assert!(q[0] <= q[1] && q[1] <= q[2]);
    let scan = size_scan(&blocks, &strata, &Codec::ALL, |_| {});
    assert_eq!(scan.overall.rows, 15_572);
    assert_eq!(scan.overall.values, 15_572 * 455);
    let raw = Codec::ALL.iter().position(|&c| c == Codec::Raw).unwrap();
    assert_eq!(scan.overall.bytes[raw], 15_572 * 455);
    // S0 terms have no nonzero superblock, so (by the consistency check) no nonzero block.
    assert_eq!(scan.per_stratum[0].nonzero, 0);
    assert_eq!(scan.per_stratum.len(), N_STRATA);
    for ci in 0..Codec::ALL.len() {
        let sum: u64 = scan.per_stratum.iter().map(|s| s.bytes[ci]).sum();
        assert_eq!(sum, scan.overall.bytes[ci]);
    }

    // The query file parses exactly and every term id is a valid row.
    let qt = QueryTerms::read(&PathBuf::from(format!("{prefix}._queries.bin"))).unwrap();
    assert_eq!(qt.n_queries, 6_980);
    assert!(
        qt.occurrences
            .keys()
            .all(|&t| (t as usize) < blocks.n_terms())
    );
    eprintln!(
        "smoke: {nonempty} nonempty terms, strata thresholds {q:?}, {} unique query terms",
        qt.occurrences.len()
    );
}
