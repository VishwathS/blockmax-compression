//! Distribution statistics of block-max rows, and analytic size ESTIMATES for sparse
//! formats. Used by `grid_bench --mode stats`.
//!
//! A row is one term's block maxes (`&[u8]`, one byte per block, mostly 0). For every
//! row, [`Stats::add_row`] records:
//!
//! - a histogram of the values (index 0 = zeros, 1..=255 = the nonzero values);
//! - **zero runs**: every maximal run of consecutive zeros, including a run at the start
//!   or end of the row (an all-zero row is one run of length n), in power-of-2 buckets;
//! - **gaps**: distance between consecutive nonzero positions (`p[j] - p[j-1]`, so
//!   adjacent nonzeros have gap 1), in power-of-2 buckets. Bucket `k` holds lengths
//!   `2^k ..= 2^(k+1) - 1` (bucket 0 = exactly 1);
//! - **groups** of 128 / 256 / 512 consecutive values (aligned at 0, the last one may be
//!   shorter): how many nonzeros each group has (0 = an all-zero group).
//!
//! and three byte counts:
//!
//! - (i) **SimdBp128**, the real codec, computed from the group widths and split into
//!   header bits for all-zero groups, header bits for non-empty groups, unused header
//!   slots, and group data. [`check_row_estimates`] proves the total equals the encoder's
//!   output (the stats scan runs it on every row).
//! - (ii) **position + value** (ESTIMATE, no codec exists): per row, Elias-γ of
//!   `nonzero + 1` (so the decoder knows the count), then for each nonzero the Elias-γ
//!   code of its gap (the first gap = position + 1) and 8 bits for its value; the row is
//!   rounded up to whole bytes. Elias-γ of `x >= 1` takes `2 ⌊log2 x⌋ + 1` bits.
//! - (iii) **zero-group bitmap** (ESTIMATE): 1 bit per 128-value group (rounded up to
//!   whole bytes per row) saying whether it is all zero, plus SimdBp128 of the non-empty
//!   groups only, concatenated in order. [`check_row_estimates`] checks the SimdBp128
//!   part against the real encoder on the compacted row.
//! - (iv) **hybrid** (ESTIMATE, like Roaring bitmaps' array vs bitmap containers): keeps
//!   SimdBp128's group structure, so a group's byte offset still comes from the headers
//!   alone (random access). Per row: the same metablock headers (64 B per 128 groups,
//!   4 bits per group) plus 1 flag bit per group (⌈groups/8⌉ bytes). Each non-empty
//!   128-group is stored either
//!   - **dense**: bit-packed at its width w, exactly as SimdBp128 (16·w bytes), or
//!   - **sparse**: its k nonzeros as (7-bit position in the group, 8-bit value) pairs,
//!     ⌈15·k / 8⌉ bytes (byte-aligned per group). Its 4-bit header field then holds k
//!     instead of w, so sparse needs k ≤ [`HYBRID_MAX_SPARSE`] (15).
//!
//!   A group is sparse when that is allowed and strictly smaller; all-zero groups cost
//!   0 data bytes either way.
//!
//! Everything is integer counting, so the totals are exact; merging strata
//! ([`Stats::merge`]) gives the same numbers as one scan over all rows.

use crate::grid::{Grid, N_STRATA};
use crate::{BlockCodec, SimdBp128};

/// Group sizes reported. Must be multiples of [`SB128_GROUP`] (they are built from
/// 128-value chunks).
pub const GROUP_LENS: [usize; 3] = [128, 256, 512];

/// Number of power-of-2 buckets; bucket 40 would hold lengths around 10^12, far above
/// any row length.
pub const N_LOG2_BUCKETS: usize = 41;

/// SimdBp128 group length (16 × STRIDE 8), which is also the number of groups that
/// share one metablock header.
pub const SB128_GROUP: usize = 128;

/// (iv) hybrid: most nonzeros a sparse group can hold (its count must fit the 4-bit
/// header field).
pub const HYBRID_MAX_SPARSE: usize = 15;

/// (iv) hybrid: bits per sparse entry = 7-bit position within a 128-group + 8-bit value.
pub const HYBRID_ENTRY_BITS: u64 = 7 + 8;

/// (iv) hybrid: data bytes of one non-empty 128-group with `nnz` nonzeros and largest
/// value `max`, and whether it is stored sparse.
pub fn hybrid_group_bytes(nnz: usize, max: u8) -> (u64, bool) {
    let dense = SimdBp128::group_data_bytes(bit_width(max) as usize) as u64;
    let sparse = (HYBRID_ENTRY_BITS * nnz as u64).div_ceil(8);
    if nnz > 0 && nnz <= HYBRID_MAX_SPARSE && sparse < dense {
        (sparse, true)
    } else {
        (dense, false)
    }
}

/// Floor of log2(x), i.e. the power-of-2 bucket of `x`. `x` must be at least 1.
pub fn log2_bucket(x: u64) -> usize {
    assert!(x >= 1, "log2_bucket(0) is undefined");
    (63 - x.leading_zeros()) as usize
}

/// Length in bits of the Elias-γ code of `x` (`x >= 1`): `2 ⌊log2 x⌋ + 1`.
pub fn gamma_bits(x: u64) -> u64 {
    2 * log2_bucket(x) as u64 + 1
}

/// Bits needed to store `v` (0 for 0, 8 for 128..=255): the SIMDBP group width.
pub fn bit_width(v: u8) -> u32 {
    8 - v.leading_zeros()
}

/// Lowest and highest length in power-of-2 bucket `k`.
pub fn bucket_range(k: usize) -> (u64, u64) {
    (1u64 << k, (1u64 << (k + 1)) - 1)
}

/// Histogram of nonzeros per group, for one group length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupHist {
    pub group_len: usize,
    /// `nnz_hist[c]` = number of groups with exactly `c` nonzeros (`c = 0`: all-zero).
    pub nnz_hist: Vec<u64>,
}

impl GroupHist {
    fn new(group_len: usize) -> GroupHist {
        GroupHist {
            group_len,
            nnz_hist: vec![0; group_len + 1],
        }
    }
    /// All groups, empty or not.
    pub fn groups(&self) -> u64 {
        self.nnz_hist.iter().sum()
    }
    /// Groups whose values are all 0.
    pub fn zero_groups(&self) -> u64 {
        self.nnz_hist[0]
    }
    /// Groups with at least one nonzero.
    pub fn nonempty_groups(&self) -> u64 {
        self.groups() - self.zero_groups()
    }
    /// Non-empty groups with between `lo` and `hi` nonzeros (inclusive).
    pub fn nonempty_with(&self, lo: usize, hi: usize) -> u64 {
        let lo = lo.max(1);
        let hi = hi.min(self.group_len);
        if lo > hi {
            return 0;
        }
        self.nnz_hist[lo..=hi].iter().sum()
    }
}

/// Byte counts for the size estimates (see the module docs). Header parts are kept in
/// BITS because each group's header field is 4 bits (half a byte).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Estimates {
    /// (i) SimdBp128 bytes, exact (equal to the encoder's output length).
    pub sb128_bytes: u64,
    /// 4-bit header fields of all-zero groups (their only cost: no data is stored).
    pub sb128_header_bits_zero_groups: u64,
    /// 4-bit header fields of non-empty groups.
    pub sb128_header_bits_nonempty: u64,
    /// Header fields past the last group (the last metablock header is always full).
    pub sb128_header_bits_unused: u64,
    /// Packed data of non-empty groups: 16 bytes per bit of the group's width.
    pub sb128_data_bytes: u64,
    /// The same data bytes, split by how many nonzeros the group has
    /// (`sb128_data_bytes_by_nnz[c]`, c = 1..=128; index 0 is always 0).
    pub sb128_data_bytes_by_nnz: Vec<u64>,
    /// (ii) position + value, in bits before rounding each row to whole bytes.
    pub posval_bits: u64,
    /// (ii) position + value, each row rounded up to whole bytes.
    pub posval_bytes: u64,
    /// (iii) bitmap part: ⌈groups / 8⌉ bytes per row.
    pub bitmap_bitmap_bytes: u64,
    /// (iii) SimdBp128 of the non-empty groups only.
    pub bitmap_simdbp_bytes: u64,
    /// (iv) hybrid: metablock headers (same as SimdBp128's).
    pub hybrid_header_bytes: u64,
    /// (iv) hybrid: the sparse/dense flags, ⌈groups / 8⌉ bytes per row.
    pub hybrid_flag_bytes: u64,
    /// (iv) hybrid: data of groups stored dense (bit-packed).
    pub hybrid_dense_bytes: u64,
    /// (iv) hybrid: data of groups stored sparse (position + value pairs).
    pub hybrid_sparse_bytes: u64,
    /// (iv) hybrid: number of non-empty groups stored dense / sparse.
    pub hybrid_dense_groups: u64,
    pub hybrid_sparse_groups: u64,
    /// (iv) hybrid: nonzeros that ended up in sparse groups.
    pub hybrid_sparse_nonzeros: u64,
}

impl Estimates {
    fn new() -> Estimates {
        Estimates {
            sb128_bytes: 0,
            sb128_header_bits_zero_groups: 0,
            sb128_header_bits_nonempty: 0,
            sb128_header_bits_unused: 0,
            sb128_data_bytes: 0,
            sb128_data_bytes_by_nnz: vec![0; SB128_GROUP + 1],
            posval_bits: 0,
            posval_bytes: 0,
            bitmap_bitmap_bytes: 0,
            bitmap_simdbp_bytes: 0,
            hybrid_header_bytes: 0,
            hybrid_flag_bytes: 0,
            hybrid_dense_bytes: 0,
            hybrid_sparse_bytes: 0,
            hybrid_dense_groups: 0,
            hybrid_sparse_groups: 0,
            hybrid_sparse_nonzeros: 0,
        }
    }
    /// (iii) total: bitmap + SimdBp128 on the non-empty groups.
    pub fn bitmap_bytes(&self) -> u64 {
        self.bitmap_bitmap_bytes + self.bitmap_simdbp_bytes
    }
    /// (iv) total: headers + flags + dense data + sparse data.
    pub fn hybrid_bytes(&self) -> u64 {
        self.hybrid_header_bytes
            + self.hybrid_flag_bytes
            + self.hybrid_dense_bytes
            + self.hybrid_sparse_bytes
    }
}

/// One row's numbers, for the per-term CSV and for [`check_row_estimates`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowSummary {
    pub n_values: usize,
    pub nonzero: usize,
    pub max: u8,
    pub groups_128: usize,
    pub zero_groups_128: usize,
    /// Non-empty 128-groups holding exactly 1 / exactly 2 nonzeros.
    pub groups_128_nnz1: usize,
    pub groups_128_nnz2: usize,
    pub sb128_bytes: u64,
    pub posval_bytes: u64,
    pub bitmap_bitmap_bytes: u64,
    pub bitmap_simdbp_bytes: u64,
    /// (iv) hybrid total for this row.
    pub hybrid_bytes: u64,
}

/// Statistics over a set of rows (one stratum, or the whole grid).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stats {
    pub rows: u64,
    pub values: u64,
    pub nonzero: u64,
    /// `value_hist[v]` = how many entries equal `v` (`v = 0`: zeros).
    pub value_hist: [u64; 256],
    /// Zero runs per power-of-2 bucket of their length.
    pub zero_run_hist: [u64; N_LOG2_BUCKETS],
    /// Total length of all zero runs (= number of zeros).
    pub zero_run_total: u64,
    /// Gaps between consecutive nonzeros per power-of-2 bucket.
    pub gap_hist: [u64; N_LOG2_BUCKETS],
    /// One histogram per entry of [`GROUP_LENS`].
    pub groups: Vec<GroupHist>,
    pub est: Estimates,
}

impl Default for Stats {
    fn default() -> Self {
        Stats::new()
    }
}

/// Nonzero count and max of one 128-value chunk.
#[derive(Debug, Clone, Copy)]
struct Chunk {
    nnz: usize,
    max: u8,
}

impl Stats {
    pub fn new() -> Stats {
        Stats {
            rows: 0,
            values: 0,
            nonzero: 0,
            value_hist: [0; 256],
            zero_run_hist: [0; N_LOG2_BUCKETS],
            zero_run_total: 0,
            gap_hist: [0; N_LOG2_BUCKETS],
            groups: GROUP_LENS.iter().map(|&g| GroupHist::new(g)).collect(),
            est: Estimates::new(),
        }
    }

    /// Number of zero runs.
    pub fn zero_runs(&self) -> u64 {
        self.zero_run_hist.iter().sum()
    }

    /// Number of gaps (= nonzeros minus rows that have at least one nonzero).
    pub fn gaps(&self) -> u64 {
        self.gap_hist.iter().sum()
    }

    /// Adds everything from `other` (e.g. a stratum into the overall totals).
    pub fn merge(&mut self, other: &Stats) {
        self.rows += other.rows;
        self.values += other.values;
        self.nonzero += other.nonzero;
        add_into(&mut self.value_hist, &other.value_hist);
        add_into(&mut self.zero_run_hist, &other.zero_run_hist);
        self.zero_run_total += other.zero_run_total;
        add_into(&mut self.gap_hist, &other.gap_hist);
        for (mine, theirs) in self.groups.iter_mut().zip(&other.groups) {
            add_into(&mut mine.nnz_hist, &theirs.nnz_hist);
        }
        let (e, o) = (&mut self.est, &other.est);
        e.sb128_bytes += o.sb128_bytes;
        e.sb128_header_bits_zero_groups += o.sb128_header_bits_zero_groups;
        e.sb128_header_bits_nonempty += o.sb128_header_bits_nonempty;
        e.sb128_header_bits_unused += o.sb128_header_bits_unused;
        e.sb128_data_bytes += o.sb128_data_bytes;
        add_into(&mut e.sb128_data_bytes_by_nnz, &o.sb128_data_bytes_by_nnz);
        e.posval_bits += o.posval_bits;
        e.posval_bytes += o.posval_bytes;
        e.bitmap_bitmap_bytes += o.bitmap_bitmap_bytes;
        e.bitmap_simdbp_bytes += o.bitmap_simdbp_bytes;
        e.hybrid_header_bytes += o.hybrid_header_bytes;
        e.hybrid_flag_bytes += o.hybrid_flag_bytes;
        e.hybrid_dense_bytes += o.hybrid_dense_bytes;
        e.hybrid_sparse_bytes += o.hybrid_sparse_bytes;
        e.hybrid_dense_groups += o.hybrid_dense_groups;
        e.hybrid_sparse_groups += o.hybrid_sparse_groups;
        e.hybrid_sparse_nonzeros += o.hybrid_sparse_nonzeros;
    }

    /// Scans one row and adds it to these statistics. Returns the row's own numbers.
    pub fn add_row(&mut self, row: &[u8]) -> RowSummary {
        let n = row.len();
        self.rows += 1;
        self.values += n as u64;

        // ---- Pass over the values: histogram, zero runs, gaps, position+value bits ----
        let mut nonzero = 0usize;
        let mut max = 0u8;
        let mut last_nz: Option<usize> = None; // position of the previous nonzero
        let mut gap_bits = 0u64; // Elias-γ bits of all gaps in this row
        for (i, &v) in row.iter().enumerate() {
            self.value_hist[usize::from(v)] += 1;
            if v == 0 {
                continue;
            }
            nonzero += 1;
            max = max.max(v);
            // The zero run that ends just before position i (if any).
            let run_start = last_nz.map_or(0, |p| p + 1);
            self.add_zero_run(i - run_start);
            // Gap from the previous nonzero; the first nonzero codes position + 1.
            match last_nz {
                Some(p) => {
                    self.gap_hist[log2_bucket((i - p) as u64)] += 1;
                    gap_bits += gamma_bits((i - p) as u64);
                }
                None => gap_bits += gamma_bits(i as u64 + 1),
            }
            last_nz = Some(i);
        }
        // Trailing zero run (the whole row if it has no nonzero).
        self.add_zero_run(n - last_nz.map_or(0, |p| p + 1));
        self.nonzero += nonzero as u64;

        // (ii) position + value
        let posval_bits = if n == 0 {
            0
        } else {
            gamma_bits(nonzero as u64 + 1) + gap_bits + 8 * nonzero as u64
        };
        let posval_bytes = posval_bits.div_ceil(8);
        self.est.posval_bits += posval_bits;
        self.est.posval_bytes += posval_bytes;

        // ---- 128-value chunks; 256/512 groups are unions of consecutive chunks ----
        let chunks: Vec<Chunk> = row
            .chunks(SB128_GROUP)
            .map(|c| Chunk {
                nnz: c.iter().filter(|&&v| v != 0).count(),
                max: c.iter().copied().max().unwrap_or(0),
            })
            .collect();
        for gh in &mut self.groups {
            let per_group = gh.group_len / SB128_GROUP;
            for g in chunks.chunks(per_group) {
                let nnz: usize = g.iter().map(|c| c.nnz).sum();
                gh.nnz_hist[nnz] += 1;
            }
        }

        // ---- (i) SimdBp128 and (iii) bitmap, from the 128-chunk widths ----
        let n_groups = chunks.len();
        let zero_groups = chunks.iter().filter(|c| c.nnz == 0).count();
        let nonempty = n_groups - zero_groups;
        let mut data_bytes = 0u64;
        for c in chunks.iter().filter(|c| c.nnz > 0) {
            let d = SimdBp128::group_data_bytes(bit_width(c.max) as usize) as u64;
            data_bytes += d;
            self.est.sb128_data_bytes_by_nnz[c.nnz] += d;
        }
        let headers = n_groups.div_ceil(SB128_GROUP) as u64; // one per 128 groups
        let header_bytes = headers * SimdBp128::HEADER_BYTES as u64;
        let sb128_bytes = header_bytes + data_bytes;
        let e = &mut self.est;
        e.sb128_bytes += sb128_bytes;
        e.sb128_data_bytes += data_bytes;
        e.sb128_header_bits_zero_groups += 4 * zero_groups as u64;
        e.sb128_header_bits_nonempty += 4 * nonempty as u64;
        e.sb128_header_bits_unused += 8 * header_bytes - 4 * n_groups as u64;

        let bitmap_bitmap_bytes = n_groups.div_ceil(8) as u64;
        let bitmap_simdbp_bytes =
            nonempty.div_ceil(SB128_GROUP) as u64 * SimdBp128::HEADER_BYTES as u64 + data_bytes;
        e.bitmap_bitmap_bytes += bitmap_bitmap_bytes;
        e.bitmap_simdbp_bytes += bitmap_simdbp_bytes;

        // ---- (iv) hybrid: per non-empty group, the cheaper of dense and sparse ----
        let mut hybrid_data = 0u64;
        for c in chunks.iter().filter(|c| c.nnz > 0) {
            let (bytes, sparse) = hybrid_group_bytes(c.nnz, c.max);
            hybrid_data += bytes;
            if sparse {
                e.hybrid_sparse_bytes += bytes;
                e.hybrid_sparse_groups += 1;
                e.hybrid_sparse_nonzeros += c.nnz as u64;
            } else {
                e.hybrid_dense_bytes += bytes;
                e.hybrid_dense_groups += 1;
            }
        }
        let hybrid_flag_bytes = n_groups.div_ceil(8) as u64;
        e.hybrid_header_bytes += header_bytes;
        e.hybrid_flag_bytes += hybrid_flag_bytes;
        let hybrid_bytes = header_bytes + hybrid_flag_bytes + hybrid_data;

        RowSummary {
            n_values: n,
            nonzero,
            max,
            groups_128: n_groups,
            zero_groups_128: zero_groups,
            groups_128_nnz1: chunks.iter().filter(|c| c.nnz == 1).count(),
            groups_128_nnz2: chunks.iter().filter(|c| c.nnz == 2).count(),
            sb128_bytes,
            posval_bytes,
            bitmap_bitmap_bytes,
            bitmap_simdbp_bytes,
            hybrid_bytes,
        }
    }

    fn add_zero_run(&mut self, len: usize) {
        if len > 0 {
            self.zero_run_hist[log2_bucket(len as u64)] += 1;
            self.zero_run_total += len as u64;
        }
    }
}

fn add_into(acc: &mut [u64], other: &[u64]) {
    for (a, b) in acc.iter_mut().zip(other) {
        *a += b;
    }
}

/// Checks a row's analytic SimdBp128 numbers against the real encoder:
/// - (i) `sb128_bytes` == length of `SimdBp128::encode(row)`;
/// - (iii) `bitmap_simdbp_bytes` == length of `SimdBp128::encode` of the row with its
///   all-zero 128-groups removed (0 bytes if nothing is left).
///
/// `values` and `out` are scratch buffers (reused to avoid allocating per row).
/// Returns a description of the first mismatch.
pub fn check_row_estimates(
    row: &[u8],
    s: &RowSummary,
    values: &mut Vec<u16>,
    out: &mut Vec<u8>,
) -> Result<(), String> {
    values.clear();
    values.extend(row.iter().map(|&b| u16::from(b)));
    out.clear();
    SimdBp128::encode(values, out);
    if out.len() as u64 != s.sb128_bytes {
        return Err(format!(
            "SimdBp128: analytic {} bytes, encoder wrote {}",
            s.sb128_bytes,
            out.len()
        ));
    }
    values.clear();
    for chunk in row.chunks(SB128_GROUP) {
        if chunk.iter().any(|&v| v != 0) {
            values.extend(chunk.iter().map(|&b| u16::from(b)));
        }
    }
    out.clear();
    SimdBp128::encode(values, out);
    if out.len() as u64 != s.bitmap_simdbp_bytes {
        return Err(format!(
            "bitmap estimate: analytic SimdBp128 part {} bytes, encoder wrote {} for the compacted row",
            s.bitmap_simdbp_bytes,
            out.len()
        ));
    }
    Ok(())
}

/// Result of [`stats_scan`]: one [`Stats`] per stratum and their sum.
#[derive(Debug, Clone)]
pub struct StatsScan {
    pub overall: Stats,
    pub per_stratum: Vec<Stats>,
}

/// Scans every row of `grid` (streaming, one row at a time, pages released after
/// each), adds it to its stratum's [`Stats`], and checks the SimdBp128-based numbers of
/// every row against the real encoder ([`check_row_estimates`]).
/// `on_row(term, stratum, &summary)` is called after each row.
///
/// # Panics
///
/// Panics if a check fails: the estimates would be wrong.
pub fn stats_scan(
    grid: &Grid,
    strata: &[u8],
    mut on_row: impl FnMut(usize, u8, &RowSummary),
) -> StatsScan {
    assert_eq!(strata.len(), grid.n_terms(), "need one stratum per term");
    let mut per_stratum: Vec<Stats> = (0..N_STRATA).map(|_| Stats::new()).collect();
    let mut values: Vec<u16> = Vec::with_capacity(grid.n_blocks());
    let mut out: Vec<u8> = Vec::with_capacity(2 * grid.n_blocks() + 4096);
    grid.advise_sequential();
    // `strata` has exactly one entry per term (asserted above), so this visits every row.
    for (term, &stratum) in strata.iter().enumerate() {
        let row = grid.row(term);
        let summary = per_stratum[usize::from(stratum)].add_row(row);
        if let Err(msg) = check_row_estimates(row, &summary, &mut values, &mut out) {
            panic!("term {term}: {msg}");
        }
        grid.release_row(term);
        on_row(term, stratum, &summary);
    }
    let mut overall = Stats::new();
    for s in &per_stratum {
        overall.merge(s);
    }
    StatsScan {
        overall,
        per_stratum,
    }
}
