//! Reading the dense block-max grids written by Parker's `export_data`, and the
//! whole-grid size scan.
//!
//! # File formats (from `bin/export_data.rs` of hierarchical_pruning @ fa0ede5)
//!
//! `<prefix>._blocks.bin` and `<prefix>._superblocks.bin` (a "grid"):
//! `u32 LE n_terms`, `u32 LE n_blocks`, then `n_terms × n_blocks` bytes, term-major:
//! row `t` (one term's block maxes, one `u8` per block) starts at byte
//! `8 + t × n_blocks`. Most entries are 0 (the term has no posting in that block).
//!
//! `<prefix>._queries.bin`: `u32 LE n_queries`, then per query `u16 q_len`, `u32 qid`,
//! and `q_len × (u16 term_id, u16 weight)`.
//!
//! [`Grid`] memory-maps a grid read-only, so a row is a borrowed `&[u8]` with no
//! copy. The mapping only costs RAM for pages actually touched, and
//! [`Grid::release_row`] hands a finished row's pages back to the kernel, so a full
//! pass over a 31 GB grid keeps RSS at a few MB.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::path::Path;

use memmap2::{Advice, Mmap, UncheckedAdvice};

use crate::Codec;

/// Bytes before the first row of a grid: `n_terms` and `n_blocks`, each a `u32`.
pub const GRID_HEADER_BYTES: usize = 8;

/// How far before a row [`Grid::release_row`] also releases (see there). 2 MiB is far
/// more than Linux's default fault-around window (64 KiB); releasing pages that are
/// already unmapped is cheap.
pub const RELEASE_BEHIND: usize = 2 << 20;

/// Anything that can go wrong opening or parsing an input file. The message names
/// the file and what was wrong with it.
#[derive(Debug)]
pub struct GridError(pub String);

impl fmt::Display for GridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GridError {}

fn err<T>(msg: String) -> Result<T, GridError> {
    Err(GridError(msg))
}

/// Parses and checks a grid header: `bytes` is (at least) the first 8 bytes of the
/// file and `file_len` its full length. Returns `(n_terms, n_blocks)`.
///
/// Refuses a file whose length isn't exactly `8 + n_terms × n_blocks`: that catches a
/// truncated copy, a wrong file, or a different layout before any row is read.
pub fn parse_grid_header(bytes: &[u8], file_len: u64) -> Result<(usize, usize), GridError> {
    if bytes.len() < GRID_HEADER_BYTES || file_len < GRID_HEADER_BYTES as u64 {
        return err(format!(
            "grid file is {file_len} bytes, shorter than its 8-byte header"
        ));
    }
    let n_terms = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as u64;
    let n_blocks = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as u64;
    let expected = GRID_HEADER_BYTES as u64 + n_terms * n_blocks;
    if file_len != expected {
        return err(format!(
            "grid header says n_terms={n_terms}, n_blocks={n_blocks}, so the file should be \
             8 + {n_terms}×{n_blocks} = {expected} bytes, but it is {file_len} bytes"
        ));
    }
    Ok((n_terms as usize, n_blocks as usize))
}

/// A read-only, memory-mapped grid file.
pub struct Grid {
    map: Mmap,
    n_terms: usize,
    n_blocks: usize,
}

impl Grid {
    /// Opens and memory-maps `path` read-only, and checks the header against the file
    /// length (see [`parse_grid_header`]).
    pub fn open(path: &Path) -> Result<Grid, GridError> {
        let file = File::open(path)
            .map_err(|e| GridError(format!("can't open grid {}: {e}", path.display())))?;
        // SAFETY: the map is read-only. It's only unsound if another process truncates or
        // rewrites the file while we hold the map; the grids are written once by
        // export_data and never modified afterwards.
        let map = unsafe { Mmap::map(&file) }
            .map_err(|e| GridError(format!("can't mmap {}: {e}", path.display())))?;
        let (n_terms, n_blocks) = parse_grid_header(&map, map.len() as u64)
            .map_err(|e| GridError(format!("{}: {e}", path.display())))?;
        Ok(Grid {
            map,
            n_terms,
            n_blocks,
        })
    }

    /// Number of rows (terms).
    pub fn n_terms(&self) -> usize {
        self.n_terms
    }

    /// Values per row (blocks or superblocks).
    pub fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    /// Row `term`: that term's block maxes, `n_blocks` bytes, borrowed from the map.
    ///
    /// # Panics
    ///
    /// Panics if `term >= n_terms`.
    pub fn row(&self, term: usize) -> &[u8] {
        assert!(
            term < self.n_terms,
            "term {term} out of range (n_terms = {})",
            self.n_terms
        );
        let start = GRID_HEADER_BYTES + term * self.n_blocks;
        &self.map[start..start + self.n_blocks]
    }

    /// Tells the kernel the file will be read front to back (more read-ahead). A hint
    /// only; errors are ignored.
    pub fn advise_sequential(&self) {
        let _ = self.map.advise(Advice::Sequential);
    }

    /// Drops this process's mapping of row `term`'s pages, plus up to
    /// [`RELEASE_BEHIND`] bytes before the row (`MADV_DONTNEED`), so pages already
    /// processed stop counting toward RSS. The data stays in the file (and usually in
    /// the page cache); touching the row again just faults it back in.
    /// A hint only; errors are ignored.
    ///
    /// Why the extra bytes before the row: on a page fault, Linux maps a whole aligned
    /// window of pages around the faulting one ("fault-around", 64 KiB by default), so
    /// reading row `t` also maps a few pages at the END of row `t - 1`, which was
    /// already released. Measured on the 2 GB superblocks grid: releasing only the row
    /// itself left ~40% of the file mapped (RSS grew to 860 MB).
    pub fn release_row(&self, term: usize) {
        let row_start = GRID_HEADER_BYTES + term * self.n_blocks;
        let start = row_start.saturating_sub(RELEASE_BEHIND);
        let len = row_start + self.n_blocks - start;
        // SAFETY: the map is MAP_SHARED and read-only (memmap2's `Mmap::map`), backed by
        // a file. MADV_DONTNEED on such a mapping does not change what the memory reads
        // as: later reads re-fault the same file bytes. (It zeroes pages only for
        // private or anonymous maps, which this is not.) So no `&[u8]` returned by `row`
        // can observe a different value afterwards.
        let _ = unsafe {
            self.map
                .unchecked_advise_range(UncheckedAdvice::DontNeed, start, len)
        };
    }
}

/// Term ids used by the dev queries, from a `_queries.bin` file.
#[derive(Debug, Clone)]
pub struct QueryTerms {
    /// Number of queries in the file.
    pub n_queries: usize,
    /// Total `(term_id, weight)` pairs over all queries.
    pub n_pairs: usize,
    /// For each distinct term id: in how many queries it appears.
    pub occurrences: BTreeMap<u32, u32>,
}

impl QueryTerms {
    /// Reads and parses a `_queries.bin` file (small: under 1 MB for 6,980 queries).
    pub fn read(path: &Path) -> Result<QueryTerms, GridError> {
        let bytes = std::fs::read(path)
            .map_err(|e| GridError(format!("can't read {}: {e}", path.display())))?;
        QueryTerms::parse(&bytes).map_err(|e| GridError(format!("{}: {e}", path.display())))
    }

    /// Parses the bytes of a `_queries.bin` file. The whole input must be consumed
    /// exactly; anything left over or missing is an error.
    pub fn parse(bytes: &[u8]) -> Result<QueryTerms, GridError> {
        let mut pos = 0usize;
        let mut take = |n: usize| -> Result<&[u8], GridError> {
            if pos + n > bytes.len() {
                return err(format!("truncated at byte {pos} (wanted {n} more bytes)"));
            }
            let s = &bytes[pos..pos + n];
            pos += n;
            Ok(s)
        };
        let n_queries = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let mut n_pairs = 0usize;
        let mut occurrences = BTreeMap::new();
        for _ in 0..n_queries {
            let q_len = u16::from_le_bytes(take(2)?.try_into().unwrap()) as usize;
            let _qid = take(4)?;
            for _ in 0..q_len {
                let term = u16::from_le_bytes(take(2)?.try_into().unwrap()) as u32;
                let _weight = take(2)?;
                *occurrences.entry(term).or_insert(0) += 1;
                n_pairs += 1;
            }
        }
        if pos != bytes.len() {
            return err(format!(
                "{} trailing bytes after {n_queries} queries",
                bytes.len() - pos
            ));
        }
        Ok(QueryTerms {
            n_queries,
            n_pairs,
            occurrences,
        })
    }
}

/// Frequency strata. Every term gets one, from its number of nonzero entries in a
/// reference grid (normally the superblocks grid, so a term lands in the same stratum
/// whichever grid is measured):
/// stratum 0 = no nonzero entry at all; strata 1-4 = quartiles of the nonzero count
/// among the remaining terms (1 = rarest quarter, 4 = most frequent quarter).
pub const N_STRATA: usize = 5;

/// Human-readable stratum names, indexed by stratum.
pub const STRATUM_NAMES: [&str; N_STRATA] = ["S0 empty", "S1 rare", "S2", "S3", "S4 frequent"];

/// Assigns strata from per-term nonzero counts (see [`N_STRATA`]). Returns
/// `(stratum per term, [q1, q2, q3])`, where a nonempty term with count `c` is in
/// stratum 1 if `c <= q1`, 2 if `c <= q2`, 3 if `c <= q3`, else 4. The thresholds are
/// the 25th/50th/75th percentile counts (nearest rank) of the nonempty terms.
pub fn assign_strata(nonzero: &[u32]) -> (Vec<u8>, [u32; 3]) {
    let mut sorted: Vec<u32> = nonzero.iter().copied().filter(|&c| c > 0).collect();
    sorted.sort_unstable();
    let q = |p: usize| -> u32 {
        if sorted.is_empty() {
            return 0;
        }
        // nearest-rank percentile: ceil(p/100 × n), 1-based
        let rank = (p * sorted.len()).div_ceil(100).max(1);
        sorted[rank - 1]
    };
    let thresholds = [q(25), q(50), q(75)];
    let strata = nonzero
        .iter()
        .map(|&c| match c {
            0 => 0,
            c if c <= thresholds[0] => 1,
            c if c <= thresholds[1] => 2,
            c if c <= thresholds[2] => 3,
            _ => 4,
        })
        .collect();
    (strata, thresholds)
}

/// Number of nonzero bytes in each row of `grid`, streaming one row at a time.
pub fn nonzero_per_row(grid: &Grid) -> Vec<u32> {
    grid.advise_sequential();
    (0..grid.n_terms())
        .map(|t| {
            let n = grid.row(t).iter().filter(|&&b| b != 0).count() as u32;
            grid.release_row(t);
            n
        })
        .collect()
}

/// One row's result from [`size_scan`].
#[derive(Debug, Clone)]
pub struct RowSize {
    pub term: usize,
    pub stratum: u8,
    pub n_values: usize,
    pub nonzero: usize,
    pub max: u8,
    /// Encoded bytes per codec, in the order of the `codecs` argument.
    pub bytes: Vec<usize>,
}

/// Totals for one group of rows (the whole grid, or one stratum).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SizeTotals {
    pub rows: u64,
    pub values: u64,
    pub nonzero: u64,
    /// Encoded bytes per codec, in the order of the `codecs` argument.
    pub bytes: Vec<u64>,
}

impl SizeTotals {
    fn new(n_codecs: usize) -> SizeTotals {
        SizeTotals {
            bytes: vec![0; n_codecs],
            ..SizeTotals::default()
        }
    }

    fn add(&mut self, row: &RowSize) {
        self.rows += 1;
        self.values += row.n_values as u64;
        self.nonzero += row.nonzero as u64;
        for (total, &b) in self.bytes.iter_mut().zip(&row.bytes) {
            *total += b as u64;
        }
    }
}

/// Result of a whole-grid size scan.
#[derive(Debug, Clone)]
pub struct SizeScan {
    pub overall: SizeTotals,
    pub per_stratum: [SizeTotals; N_STRATA],
}

/// Encodes EVERY row of `grid` with each codec in `codecs` (one row = one `encode`
/// call, the way one term's level would be stored), decodes it back and checks the
/// round trip exactly, and adds up the encoded sizes overall and per stratum.
///
/// Streams one row at a time and releases each row's pages afterwards, so memory
/// stays at a few row-sized buffers whatever the grid size. `on_row` is called after
/// each row (for per-term output and progress).
///
/// # Panics
///
/// Panics if any round trip fails or `decode` doesn't consume exactly the encoded
/// bytes: a size number for a wrong codec would be meaningless.
pub fn size_scan(
    grid: &Grid,
    strata: &[u8],
    codecs: &[Codec],
    mut on_row: impl FnMut(&RowSize),
) -> SizeScan {
    assert_eq!(strata.len(), grid.n_terms(), "need one stratum per term");
    let n = grid.n_blocks();
    let mut values = vec![0u16; n];
    let mut decoded = vec![0u16; n];
    let mut encoded: Vec<u8> = Vec::with_capacity(2 * n + 4096);
    let mut overall = SizeTotals::new(codecs.len());
    let mut per_stratum: [SizeTotals; N_STRATA] =
        std::array::from_fn(|_| SizeTotals::new(codecs.len()));

    grid.advise_sequential();
    for term in 0..grid.n_terms() {
        let row = grid.row(term);
        let mut nonzero = 0usize;
        let mut max = 0u8;
        for (v, &b) in values.iter_mut().zip(row) {
            *v = u16::from(b);
            nonzero += usize::from(b != 0);
            max = max.max(b);
        }
        grid.release_row(term);

        let mut bytes = Vec::with_capacity(codecs.len());
        for &codec in codecs {
            encoded.clear();
            codec.encode(&values, &mut encoded);
            let used = codec.decode(&encoded, &mut decoded);
            assert_eq!(
                used,
                encoded.len(),
                "{}: term {term}: decode consumed {used} of {} bytes",
                codec.name(),
                encoded.len()
            );
            assert!(
                decoded == values,
                "{}: term {term}: round trip FAILED",
                codec.name()
            );
            bytes.push(encoded.len());
        }

        let result = RowSize {
            term,
            stratum: strata[term],
            n_values: n,
            nonzero,
            max,
            bytes,
        };
        overall.add(&result);
        per_stratum[usize::from(strata[term])].add(&result);
        on_row(&result);
    }
    SizeScan {
        overall,
        per_stratum,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_ok_and_length_checked() {
        let mut h = Vec::new();
        h.extend_from_slice(&3u32.to_le_bytes());
        h.extend_from_slice(&5u32.to_le_bytes());
        assert_eq!(parse_grid_header(&h, 8 + 15).unwrap(), (3, 5));
        assert!(parse_grid_header(&h, 8 + 14).is_err());
        assert!(parse_grid_header(&h, 8 + 16).is_err());
        assert!(parse_grid_header(&h[..7], 7).is_err());
    }

    #[test]
    fn strata_quartiles() {
        // 0 is empty; 1..=8 are the nonempty counts: q1=2, q2=4, q3=6.
        let counts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0];
        let (s, q) = assign_strata(&counts);
        assert_eq!(q, [2, 4, 6]);
        assert_eq!(s, vec![0, 1, 1, 2, 2, 3, 3, 4, 4, 0]);
    }

    #[test]
    fn strata_all_empty() {
        let (s, q) = assign_strata(&[0, 0]);
        assert_eq!(s, vec![0, 0]);
        assert_eq!(q, [0, 0, 0]);
    }

    fn query_bytes(queries: &[(u32, &[(u16, u16)])]) -> Vec<u8> {
        let mut b = (queries.len() as u32).to_le_bytes().to_vec();
        for (qid, terms) in queries {
            b.extend_from_slice(&(terms.len() as u16).to_le_bytes());
            b.extend_from_slice(&qid.to_le_bytes());
            for (t, w) in *terms {
                b.extend_from_slice(&t.to_le_bytes());
                b.extend_from_slice(&w.to_le_bytes());
            }
        }
        b
    }

    #[test]
    fn queries_parse_and_count() {
        let b = query_bytes(&[(7, &[(3, 10), (5, 2)]), (9, &[(3, 1)]), (11, &[])]);
        let q = QueryTerms::parse(&b).unwrap();
        assert_eq!(q.n_queries, 3);
        assert_eq!(q.n_pairs, 3);
        assert_eq!(
            q.occurrences.into_iter().collect::<Vec<_>>(),
            vec![(3, 2), (5, 1)]
        );
    }

    #[test]
    fn queries_reject_truncated_and_trailing() {
        let b = query_bytes(&[(7, &[(3, 10)])]);
        assert!(QueryTerms::parse(&b[..b.len() - 1]).is_err());
        let mut extra = b.clone();
        extra.push(0);
        assert!(QueryTerms::parse(&extra).is_err());
    }
}
