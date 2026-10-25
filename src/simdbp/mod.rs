//! `SimdBp128` / `SimdBp256` / `SimdBp512`: Parker's SIMDBP format, as codecs.
//!
//! SIMDBP ("SIMD binary packing") is the format Parker's hierarchical_pruning uses for
//! block maxes (`BlockData::SIMDBP`, `src/index/compress_simdbp.rs`). His code picks
//! ONE group size when it is compiled (512 with AVX-512, 256 with AVX2, else 128). Here
//! all three are separate codecs, chosen by the `STRIDE` parameter, so the bytes never
//! depend on how this crate was compiled:
//!
//! | codec       | `STRIDE` (lanes) | values per group | SIMD kernel used if the CPU has it |
//! |-------------|------------------|------------------|------------------------------------|
//! | `SimdBp128` | 8                | 128              | SSE3                               |
//! | `SimdBp256` | 16               | 256              | AVX2                               |
//! | `SimdBp512` | 32               | 512              | AVX-512 F + BW                     |
//!
//! Without the instructions, the scalar [`Kernel`] produces the same bytes.
//!
//! # Format of one `encode` call (a row of `n` values)
//!
//! Let `GL = 16 × STRIDE` (values per group), `G = ceil(n / GL)` groups, and
//! `M = ceil(G / GL)` **metablocks** (one metablock = `GL` consecutive groups).
//!
//! ```text
//! [ header 0 | header 1 | ... | header M-1 ][ group 0 data ][ group 1 data ] ...
//!   8 × STRIDE bytes each                     2 × STRIDE × w0  2 × STRIDE × w1 bytes
//! ```
//!
//! - Group `g` holds values `g × GL ..`, packed at bit width `w_g` = the bit width of
//!   its largest value (0..=8), in the lane layout described in the `kernel` module.
//!   That's `w_g × STRIDE` u16 words; width 0 (all zeros) has no data at all.
//! - **Header** `m` holds the widths of groups `m × GL .. (m + 1) × GL` as 4-bit values,
//!   packed with the same lane layout at width 4: `4 × STRIDE` u16 words. Widths of
//!   groups past the last one (in the last header) are 0.
//! - **All headers come first**, then the groups' data in order. So group `g`'s data
//!   starts at `M × 8 × STRIDE + 2 × STRIDE × (w_0 + ... + w_(g-1))` bytes.
//! - The last group, if shorter than `GL`, is zero-padded to `GL` values before packing.
//! - Every u16 word is stored **little-endian**. On x86 that is exactly how Parker's
//!   `Vec<u16>` sits in memory, so the bytes equal his `compress(..)` output viewed as
//!   bytes; `tests/simdbp.rs` checks this against output of his own code.
//! - `n` is not stored (as for every codec here). `n = 0` encodes to **no bytes**;
//!   Parker's `compress` can't handle an empty input (it panics), so that case is ours.
//!
//! # Random access
//!
//! A group is one SIMDBP group. One-shot `group_range` reads the 4-bit widths of the
//! groups before it straight out of the headers, **O(g)**. `locate_groups` (used by
//! `GroupIndex`) does it once for all groups, **O(G)**, the same work as Parker's
//! `Decompressor::init`; after that each lookup is O(1). Decoding a located group
//! unpacks all `GL` values into a stack buffer and copies out the real ones; nothing
//! is allocated and nothing is cached (unlike Parker's `Decompressor`, which keeps the
//! last decoded group).
//!
//! # Size
//!
//! Every row pays at least one header: 64 / 128 / 256 bytes for 128 / 256 / 512. That's
//! small for a long row (a 512 header covers 262,144 values) but large for a short one,
//! so a row should be one long sequence (e.g. a term's whole level), not one group.
//!
//! # Provenance
//!
//! The format and the encode/decode steps follow `compress` and `Decompressor` in
//! `src/index/compress_simdbp.rs` of Parker Carlson's hierarchical_pruning (commit
//! fa0ede5241ea913ec9e293bdbef0d0b1012e7f07, Apache License 2.0, see
//! `third_party/hierarchical_pruning/`). This file is a new implementation of that
//! format, MODIFIED in behavior as listed above (range check, `n = 0`, no cache, bytes).

use std::ops::Range;

use crate::{BlockCodec, MAX_VALUE, RandomAccessCodec};

mod kernel;
#[cfg(target_arch = "x86_64")]
mod kernels;

pub use kernel::{Kernel, MAX_GROUP_LEN, MAX_STRIDE, MAX_WIDTH};

/// Parker's SIMDBP format with `STRIDE` lanes per group (groups of `16 × STRIDE`).
/// Only 8, 16, and 32 are allowed (anything else is a compile error); use the
/// [`SimdBp128`], [`SimdBp256`], and [`SimdBp512`] aliases.
pub struct SimdBp<const STRIDE: usize>;

/// SIMDBP with groups of 128 values (Parker's SSE build).
pub type SimdBp128 = SimdBp<8>;
/// SIMDBP with groups of 256 values (Parker's AVX2 build).
pub type SimdBp256 = SimdBp<16>;
/// SIMDBP with groups of 512 values (Parker's AVX-512 build, the one this machine uses).
pub type SimdBp512 = SimdBp<32>;

/// Width of the header fields that store each group's bit width.
const HEADER_FIELD_BITS: usize = 4;

impl<const STRIDE: usize> SimdBp<STRIDE> {
    /// Evaluated at compile time wherever it's used, so `SimdBp<7>` fails to build.
    const VALID_STRIDE: () = assert!(
        STRIDE == 8 || STRIDE == 16 || STRIDE == 32,
        "SimdBp: STRIDE must be 8, 16, or 32"
    );

    /// Values per group (`16 × STRIDE`), also the number of groups per metablock.
    const GL: usize = 16 * STRIDE;

    /// Bytes per metablock header: `GL` widths × 4 bits = `4 × STRIDE` u16 words.
    pub const HEADER_BYTES: usize = 2 * HEADER_FIELD_BITS * STRIDE;

    /// Bytes of data for one group at bit width `width`: `width × STRIDE` u16 words.
    pub const fn group_data_bytes(width: usize) -> usize {
        2 * width * STRIDE
    }

    /// Number of metablock headers in a row of `n` values.
    pub fn num_metablocks(n: usize) -> usize {
        n.div_ceil(Self::GL).div_ceil(Self::GL)
    }

    /// Size of all headers of a row of `n` values, in bytes. Group data starts here.
    pub fn headers_len(n: usize) -> usize {
        Self::num_metablocks(n) * Self::HEADER_BYTES
    }

    /// The bit width of group `group`, read from the headers at the start of `input`.
    /// O(1). Mostly useful for tests and reports.
    ///
    /// # Panics
    ///
    /// Panics if `group` is out of range, the headers are cut off, or the stored width
    /// is above 8 (`encode` never writes that).
    pub fn group_width(input: &[u8], n: usize, group: usize) -> usize {
        let n_groups = n.div_ceil(Self::GL);
        assert!(
            group < n_groups,
            "SimdBp: group index {group} is out of range: a row of {n} values has {n_groups} groups"
        );
        header_width::<STRIDE>(headers(input, Self::headers_len(n)), group)
    }

    /// [`BlockCodec::encode`] with a chosen [`Kernel`] (the trait method uses
    /// [`Kernel::best`]). Every kernel writes the same bytes.
    ///
    /// # Panics
    ///
    /// Panics if a value is above 255, or the kernel can't run here (see
    /// [`Kernel::pack`]).
    pub fn encode_with(kernel: Kernel, input: &[u16], out: &mut Vec<u8>) {
        let () = Self::VALID_STRIDE;
        let gl = Self::GL;
        if input.is_empty() {
            return; // our extension: an empty row is zero bytes
        }

        // Reserve room for all headers now; each is filled in once its metablock's
        // groups are written (headers come first, but need the widths found later).
        let header_start = out.len();
        out.resize(header_start + Self::headers_len(input.len()), 0);

        // Stack buffers sized for the biggest stride, so nothing is allocated here.
        let mut values = [0u16; MAX_GROUP_LEN]; // one group, zero-padded
        let mut widths = [0u16; MAX_GROUP_LEN]; // one metablock's group widths
        let mut packed = [0u16; MAX_WIDTH * MAX_STRIDE]; // one packed group or header

        // `chunks(gl * gl)` gives one metablock's values at a time (the last may be
        // short); `chunks(gl)` inside gives its groups.
        for (metablock, metablock_values) in input.chunks(gl * gl).enumerate() {
            widths.fill(0); // slots past the last group stay 0, as in Parker's code
            for (slot, group) in metablock_values.chunks(gl).enumerate() {
                let max = group.iter().copied().max().unwrap_or(0);
                // Checked in release builds too. Parker's code would silently store a
                // value above 255 at width 8 and lose its high bits.
                assert!(max <= MAX_VALUE, "SimdBp: value {max} is above {MAX_VALUE}");
                let width = bit_width(max);
                widths[slot] = width as u16;

                // Zero-pad a short last group to GL values, like Parker's `compress`.
                let values = &mut values[..gl];
                values.fill(0);
                values[..group.len()].copy_from_slice(group);

                let words = &mut packed[..width * STRIDE];
                kernel.pack(STRIDE, width, values, words);
                push_le_words(out, words);
            }

            // The header: this metablock's GL widths, packed at 4 bits each.
            let words = &mut packed[..HEADER_FIELD_BITS * STRIDE];
            kernel.pack(STRIDE, HEADER_FIELD_BITS, &widths[..gl], words);
            let at = header_start + metablock * Self::HEADER_BYTES;
            write_le_words(&mut out[at..at + Self::HEADER_BYTES], words);
        }
    }

    /// [`BlockCodec::decode`] with a chosen [`Kernel`] (the trait method uses
    /// [`Kernel::best`]). Every kernel gives the same values.
    ///
    /// # Panics
    ///
    /// Same as [`BlockCodec::decode`], or if the kernel can't run here.
    pub fn decode_with(kernel: Kernel, input: &[u8], out: &mut [u16]) -> usize {
        let () = Self::VALID_STRIDE;
        let header_bytes = Self::headers_len(out.len()); // 0 when out is empty
        let headers = headers(input, header_bytes);

        let mut pos = header_bytes; // byte offset of the next group's data
        for (group, group_out) in out.chunks_mut(Self::GL).enumerate() {
            let width = header_width::<STRIDE>(headers, group);
            let n_bytes = Self::group_data_bytes(width);
            let data = input
                .get(pos..pos + n_bytes)
                .unwrap_or_else(|| panic!("SimdBp: input ended in the middle of group {group}"));
            unpack_group::<STRIDE>(kernel, data, width, group_out);
            pos += n_bytes;
        }
        pos
    }

    /// [`RandomAccessCodec::decode_group_bytes`] with a chosen [`Kernel`].
    ///
    /// # Panics
    ///
    /// Same as the trait method, or if the kernel can't run here.
    pub fn decode_group_bytes_with(kernel: Kernel, data: &[u8], out: &mut [u16]) {
        let () = Self::VALID_STRIDE;
        assert!(
            out.len() <= Self::GL,
            "SimdBp: a group has at most {} values, but out has {}",
            Self::GL,
            out.len()
        );
        // The width isn't passed in: a group's data is exactly 2 × STRIDE × width
        // bytes, so the length gives it back.
        let bytes_per_bit = Self::group_data_bytes(1);
        let width = data.len() / bytes_per_bit;
        assert!(
            data.len().is_multiple_of(bytes_per_bit) && width <= MAX_WIDTH,
            "SimdBp: {} bytes is not a valid group size (must be {bytes_per_bit} × width, width <= 8)",
            data.len()
        );
        unpack_group::<STRIDE>(kernel, data, width, out);
    }
}

impl<const STRIDE: usize> BlockCodec for SimdBp<STRIDE> {
    /// One group. Only a hint: rows should be much longer than this, because every row
    /// pays for at least one header (see "Size" in the module docs).
    const BLOCK_SIZE: usize = 16 * STRIDE;

    fn encode(input: &[u16], out: &mut Vec<u8>) {
        Self::encode_with(Kernel::best(STRIDE), input, out);
    }

    fn decode(input: &[u8], out: &mut [u16]) -> usize {
        Self::decode_with(Kernel::best(STRIDE), input, out)
    }
}

/// Random access, one SIMDBP group at a time. See "Random access" in the module docs.
impl<const STRIDE: usize> RandomAccessCodec for SimdBp<STRIDE> {
    const GROUP_LEN: usize = 16 * STRIDE;

    fn group_range(input: &[u8], n: usize, group: usize) -> Range<usize> {
        let width = Self::group_width(input, n, group); // also checks the group index
        let header_bytes = Self::headers_len(n);
        let headers = headers(input, header_bytes);
        // Sum the widths of every earlier group: O(group).
        let skipped: usize = (0..group).map(|g| header_width::<STRIDE>(headers, g)).sum();
        let start = header_bytes + Self::group_data_bytes(skipped);
        let end = start + Self::group_data_bytes(width);
        assert!(
            end <= input.len(),
            "SimdBp: input ended in the middle of group {group}"
        );
        start..end
    }

    fn locate_groups(input: &[u8], n: usize, starts: &mut Vec<usize>) {
        let n_groups = Self::num_groups(n);
        let header_bytes = Self::headers_len(n);
        let headers = headers(input, header_bytes);
        starts.clear();
        // `reserve` does nothing if the capacity is already big enough.
        starts.reserve(n_groups + 1);
        let mut pos = header_bytes; // group 0's data starts right after the headers
        for group in 0..n_groups {
            starts.push(pos);
            pos += Self::group_data_bytes(header_width::<STRIDE>(headers, group));
        }
        starts.push(pos); // end of the row
        // Check once here, so later lookups through the index can't run off the end.
        assert!(
            pos <= input.len(),
            "SimdBp: input has {} bytes, but its headers describe {pos}",
            input.len()
        );
    }

    fn decode_group_bytes(data: &[u8], out: &mut [u16]) {
        Self::decode_group_bytes_with(Kernel::best(STRIDE), data, out);
    }
}

/// Number of bits needed to write `value` in binary (0 for 0, 3 for 5, 8 for 255).
/// Same as Parker's `bits_needed` for every value <= 255.
fn bit_width(value: u16) -> usize {
    (u16::BITS - value.leading_zeros()) as usize
}

/// The first `len` bytes of `input` (the metablock headers), or a panic saying so.
fn headers(input: &[u8], len: usize) -> &[u8] {
    input.get(..len).unwrap_or_else(|| {
        panic!(
            "SimdBp: need {len} bytes of metablock headers, but input has only {}",
            input.len()
        )
    })
}

/// The bit width of group `group`, read from its 4-bit field in the headers.
///
/// The headers use the lane layout from the `kernel` module at width 4. Inside a
/// metablock, group `group` is element `slot = group % GL`, which is lane
/// `slot % STRIDE`, entry `j = slot / STRIDE` of that lane. Entry `j` sits at bits
/// `4j .. 4j + 4` of the lane's bit stream, i.e. in the lane's word `j / 4` at bit
/// `4 × (j % 4)`. Four-bit fields never cross a word boundary, so that's one u16 read.
fn header_width<const STRIDE: usize>(headers: &[u8], group: usize) -> usize {
    let gl = 16 * STRIDE;
    let (metablock, slot) = (group / gl, group % gl);
    let (lane, j) = (slot % STRIDE, slot / STRIDE);
    let word_index = (j / 4) * STRIDE + lane; // u16 index inside this header
    let byte = metablock * 2 * HEADER_FIELD_BITS * STRIDE + 2 * word_index;
    let word = u16::from_le_bytes([headers[byte], headers[byte + 1]]);
    let width = usize::from((word >> (4 * (j % 4))) & 0xF);
    assert!(
        width <= MAX_WIDTH,
        "SimdBp: invalid bit width {width} for group {group}"
    );
    width
}

/// Unpacks one group's data bytes (`2 × STRIDE × width` of them) and writes its first
/// `out.len()` values to `out`.
fn unpack_group<const STRIDE: usize>(kernel: Kernel, data: &[u8], width: usize, out: &mut [u16]) {
    // Read the little-endian u16 words into a stack buffer. Copying (instead of
    // reinterpreting the bytes in place) works for any alignment and any CPU byte
    // order, and is at most 512 bytes.
    let mut words = [0u16; MAX_WIDTH * MAX_STRIDE];
    let words = &mut words[..width * STRIDE];
    // `as_chunks::<2>` views the bytes as `[u8; 2]` pairs (plus a leftover, which the
    // callers guarantee is empty: they pass exactly 2 bytes per word).
    let (pairs, leftover) = data.as_chunks::<2>();
    assert!(
        pairs.len() == words.len() && leftover.is_empty(),
        "SimdBp: group data has {} bytes, expected {}",
        data.len(),
        2 * words.len()
    );
    for (word, &pair) in words.iter_mut().zip(pairs) {
        *word = u16::from_le_bytes(pair);
    }
    let mut values = [0u16; MAX_GROUP_LEN];
    let values = &mut values[..16 * STRIDE];
    kernel.unpack(STRIDE, width, words, values);
    // A short last group: copy only its real values; the zero padding is dropped.
    out.copy_from_slice(&values[..out.len()]);
}

/// Appends `words` to `out` as little-endian bytes.
fn push_le_words(out: &mut Vec<u8>, words: &[u16]) {
    for &word in words {
        out.extend_from_slice(&word.to_le_bytes());
    }
}

/// Writes `words` as little-endian bytes into `out` (exactly `2 × words.len()` bytes).
fn write_le_words(out: &mut [u8], words: &[u16]) {
    let (pairs, leftover) = out.as_chunks_mut::<2>();
    assert!(pairs.len() == words.len() && leftover.is_empty());
    for (pair, &word) in pairs.iter_mut().zip(words) {
        *pair = word.to_le_bytes();
    }
}
