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
//!   (headers first) and is meant for random access later: chunk k's data starts at
//!   byte `G + 16 × (w0 + ... + w(k-1))`, which can be computed from the width bytes
//!   alone without touching any packed data.
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

use bitpacking::{BitPacker, BitPacker4x};

use crate::{BlockCodec, MAX_VALUE};

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
        let packer = BitPacker4x::new();

        // Part 1: one width byte per chunk. `chunks` gives full 128-value slices and
        // then one shorter slice at the end if `input.len()` isn't a multiple of 128.
        let widths: Vec<u8> = input
            .chunks(CHUNK_LEN)
            .map(|chunk| {
                // `chunks` never yields an empty slice, so `max()` always finds a value.
                let max = chunk.iter().copied().max().unwrap_or(0);
                // Checked in release builds too: a value above 255 would need more than
                // 8 bits and would break the "width fits in 0..=8" rule of the format.
                assert!(max <= MAX_VALUE, "Bp128: value {max} is above {MAX_VALUE}");
                bit_width(max)
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
            assert!(width <= MAX_BITS, "Bp128: invalid bit width {width}");
            let n_bytes = BitPacker4x::compressed_block_size(width); // 16 × width
            let data = input
                .get(pos..pos + n_bytes)
                .expect("Bp128: input ended in the middle of a chunk");
            // Always unpacks a full 128 values, including any zero padding.
            packer.decompress(data, &mut unpacked, width);
            // Copy back only as many values as this chunk has (fewer than 128 for a
            // partial last chunk). `zip` stops at the shorter of the two.
            for (value, &wide) in chunk_out.iter_mut().zip(&unpacked) {
                // width <= 8, so every unpacked value is <= 255 and fits in a u16.
                *value = wide as u16;
            }
            pos += n_bytes;
        }
        pos
    }
}
