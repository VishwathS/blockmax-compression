//! `Bp128`: binary packing in chunks of 128 values, using the `bitpacking` crate.
//!
//! The input is cut into **chunks** of 128 values. Each chunk is stored at its own
//! **bit width**: the number of bits its largest value needs. A chunk whose max is 5
//! (`0b101`) uses 3 bits per value, so its 128 values take 128 × 3 / 8 = 48 bytes.
//! Values are at most 255 (see [`BlockCodec`]), so a width is always 0..=8.
//!
//! # Format of one `encode` call
//!
//! ```text
//! [ w0 | w1 | ... | w(G-1) ][ chunk 0 data ][ chunk 1 data ] ... [ chunk G-1 data ]
//!   one width byte per chunk   16 × w0 bytes   16 × w1 bytes        16 × w(G-1) bytes
//! ```
//!
//! - G = `ceil(n / 128)` chunks, where n is the number of values. n itself is not
//!   stored: the decoder gets it from `out.len()`, like every codec in this crate.
//! - **All width bytes come first**, before any data. This mirrors Parker's SIMDBP
//!   (headers first) and is what makes random access work (see the
//!   [`RandomAccessCodec`] impl below): chunk k's data starts at byte
//!   `G + 16 × (w0 + ... + w(k-1))`, which can be computed from the width bytes alone
//!   without touching any packed data.
//! - A chunk of width 0 (all zeros) has **no data bytes**, only its width byte.
//! - The last chunk, if shorter than 128, is padded with zeros to 128 values and packed
//!   like the others. Zeros never raise the max, so padding doesn't change the width,
//!   but it does store `width × (128 − leftover)` bits of padding. Parker's SIMDBP pads
//!   its last group the same way.
//!
//! The packing itself is done by `bitpacking::BitPacker4x`, which works on exactly 128
//! `u32`s at a time. Its byte layout interleaves the values in 4 lanes (value i goes to
//! lane i % 4) so SIMD instructions can pack 4 values at once. We only rely on the
//! crate to unpack what it packed, so the exact layout doesn't matter to this code.

use std::ops::Range;

use bitpacking::{BitPacker, BitPacker4x};

use crate::{BlockCodec, RandomAccessCodec, check_values};

/// Values per chunk. `BitPacker4x` always packs exactly this many (128).
const CHUNK_LEN: usize = BitPacker4x::BLOCK_LEN;

/// The largest valid width: values are <= 255, which fits in 8 bits.
const MAX_BITS: u8 = 8;

/// Bit-packing codec: 128-value chunks, each packed at the width of its largest value.
pub struct Bp128;

/// Number of bits needed to write `value` in binary (0 for 0, 3 for 5, 8 for 255).
fn bit_width(value: u16) -> u8 {
    // A u16 has 16 bits. Subtracting the leading zero bits leaves the bits in use.
    (u16::BITS - value.leading_zeros()) as u8
}

impl BlockCodec for Bp128 {
    const BLOCK_SIZE: usize = CHUNK_LEN;

    fn encode(input: &[u16], out: &mut Vec<u8>) {
        // `new()` picks the fastest implementation this CPU supports. It's cheap.
        // Checked first, in release builds too, so nothing is written for a bad input:
        // a value above 255 would need more than 8 bits and would break the "width
        // fits in 0..=8" rule of the format.
        check_values("Bp128", input);
        let packer = BitPacker4x::new();

        // Part 1: one width byte per chunk. `chunks` gives full 128-value slices and
        // then one shorter slice at the end if `input.len()` isn't a multiple of 128.
        let widths: Vec<u8> = input
            .chunks(CHUNK_LEN)
            .map(|chunk| {
                // `chunks` never yields an empty slice, so `max()` always finds a value.
                bit_width(chunk.iter().copied().max().unwrap_or(0))
            })
            .collect();
        out.extend_from_slice(&widths);

        // Part 2: the packed data, chunk by chunk.
        // `widened` holds one chunk converted to u32, because the crate packs u32s.
        let mut widened = [0u32; CHUNK_LEN];
        // Packed output of one chunk: at most 8 bits × 128 values = 128 bytes.
        let mut packed = [0u8; CHUNK_LEN * MAX_BITS as usize / 8];
        for (chunk, &width) in input.chunks(CHUNK_LEN).zip(&widths) {
            // Zero the whole buffer first, so a short last chunk is padded with zeros.
            widened.fill(0);
            for (wide, &value) in widened.iter_mut().zip(chunk) {
                *wide = u32::from(value);
            }
            // Returns how many bytes it wrote: 16 × width (0 when width is 0).
            let n_bytes = packer.compress(&widened, &mut packed, width);
            out.extend_from_slice(&packed[..n_bytes]);
        }
    }

    fn decode(input: &[u8], out: &mut [u16]) -> usize {
        let packer = BitPacker4x::new();

        // The caller tells us n through `out.len()`, so we know how many width bytes
        // to read: one per chunk, rounding up for a partial last chunk.
        let n_chunks = out.len().div_ceil(CHUNK_LEN);
        assert!(
            input.len() >= n_chunks,
            "Bp128: need {n_chunks} width bytes, but input has only {}",
            input.len()
        );
        let widths = &input[..n_chunks];

        let mut pos = n_chunks; // byte offset of the next chunk's data
        let mut unpacked = [0u32; CHUNK_LEN];
        for (chunk_out, &width) in out.chunks_mut(CHUNK_LEN).zip(widths) {
            let n_bytes = chunk_bytes(width); // 16 × width, after checking width <= 8
            let data = input
                .get(pos..pos + n_bytes)
                .expect("Bp128: input ended in the middle of a chunk");
            unpack_chunk(&packer, data, width, &mut unpacked, chunk_out);
            pos += n_bytes;
        }
        pos
    }
}

/// Size in bytes of one chunk's packed data at `width` bits: 16 × width (128 values ×
/// width bits / 8). Panics on a width above 8, which `encode` never writes.
fn chunk_bytes(width: u8) -> usize {
    assert!(width <= MAX_BITS, "Bp128: invalid bit width {width}");
    BitPacker4x::compressed_block_size(width)
}

/// Unpacks one chunk (`data`, packed at `width` bits) and writes its first `out.len()`
/// values to `out`. `scratch` is a reusable buffer for the full 128 unpacked values.
fn unpack_chunk(
    packer: &BitPacker4x,
    data: &[u8],
    width: u8,
    scratch: &mut [u32; CHUNK_LEN],
    out: &mut [u16],
) {
    // Always unpacks a full 128 values, including any zero padding.
    packer.decompress(data, scratch, width);
    // Copy back only as many values as this chunk has (fewer than 128 for a partial
    // last chunk). `zip` stops at the shorter of the two.
    for (value, &wide) in out.iter_mut().zip(scratch.iter()) {
        // width <= 8, so every unpacked value is <= 255 and fits in a u16.
        *value = wide as u16;
    }
}

/// Random access, one 128-value chunk at a time. A **group** here is exactly one chunk,
/// so the byte format is the same as for sequential `decode`; nothing extra is stored.
///
/// Finding chunk k needs the sum of the k width bytes before it, because chunk sizes
/// vary (16 × width bytes):
///
/// - [`group_range`](RandomAccessCodec::group_range) (used by the one-shot
///   `decode_group`) sums `w0 + ... + w(k-1)` each time: **O(k)** byte reads, no state.
///   Fine for a few lookups; decoding every chunk this way would cost O(G²) in total.
/// - [`locate_groups`](RandomAccessCodec::locate_groups) (used by `GroupIndex`) does the
///   prefix sum once for all chunks in **O(G)**, after which each lookup is **O(1)**.
///   That mirrors SIMDBP's `Decompressor::init` + `decompress_block`.
///
/// Decoding a located chunk always unpacks 128 values (O(1) with respect to the row
/// length), then copies only the real ones of a partial last chunk.
impl RandomAccessCodec for Bp128 {
    const GROUP_LEN: usize = CHUNK_LEN;

    fn group_range(input: &[u8], n: usize, group: usize) -> Range<usize> {
        let n_chunks = Self::num_groups(n);
        assert!(
            group < n_chunks,
            "Bp128: group index {group} is out of range: a row of {n} values has {n_chunks} groups"
        );
        let widths = width_bytes(input, n_chunks);
        // Data starts after all G width bytes, then skips the earlier chunks' data.
        let skipped: usize = widths[..group].iter().map(|&w| chunk_bytes(w)).sum();
        let start = n_chunks + skipped;
        let end = start + chunk_bytes(widths[group]);
        assert!(
            end <= input.len(),
            "Bp128: input ended in the middle of chunk {group}"
        );
        start..end
    }

    fn locate_groups(input: &[u8], n: usize, starts: &mut Vec<usize>) {
        let n_chunks = Self::num_groups(n);
        let widths = width_bytes(input, n_chunks);
        starts.clear();
        // `reserve` does nothing if the capacity is already big enough.
        starts.reserve(n_chunks + 1);
        let mut pos = n_chunks; // chunk 0's data starts right after the width bytes
        for &width in widths {
            starts.push(pos);
            pos += chunk_bytes(width);
        }
        starts.push(pos); // end of the row
        // Check once here, so later lookups through the index can't run off the end.
        assert!(
            pos <= input.len(),
            "Bp128: input has {} bytes, but its width bytes describe {pos}",
            input.len()
        );
    }

    fn decode_group_bytes(data: &[u8], out: &mut [u16]) {
        assert!(
            out.len() <= CHUNK_LEN,
            "Bp128: a group has at most {CHUNK_LEN} values, but out has {}",
            out.len()
        );
        // The width isn't passed in: the data is exactly 16 × width bytes, so the
        // length gives it back.
        let n_bytes_per_bit = BitPacker4x::compressed_block_size(1); // 16
        let width = data.len() / n_bytes_per_bit;
        assert!(
            data.len().is_multiple_of(n_bytes_per_bit) && width <= usize::from(MAX_BITS),
            "Bp128: {} bytes is not a valid chunk size (must be 16 × width, width <= 8)",
            data.len()
        );
        let width = width as u8; // <= 8, checked above
        let mut scratch = [0u32; CHUNK_LEN];
        unpack_chunk(&BitPacker4x::new(), data, width, &mut scratch, out);
    }
}

/// The first `n_chunks` bytes of `input`: one width byte per chunk.
fn width_bytes(input: &[u8], n_chunks: usize) -> &[u8] {
    input.get(..n_chunks).unwrap_or_else(|| {
        panic!(
            "Bp128: need {n_chunks} width bytes, but input has only {}",
            input.len()
        )
    })
}
