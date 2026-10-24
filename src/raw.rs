//! `Raw`: no compression. Each value is stored as 4 little-endian bytes.
//!
//! This is the baseline every other codec is compared against.

use crate::BlockCodec;

/// Stores each `u32` as its 4 bytes, least significant byte first.
pub struct Raw;

impl BlockCodec for Raw {
    const BLOCK_SIZE: usize = 128;

    fn encode(input: &[u32], out: &mut Vec<u8>) {
        // We know the exact output size, so grow the buffer once up front.
        out.reserve(input.len() * 4);
        for &value in input {
            // e.g. 0x0102_0304 becomes [0x04, 0x03, 0x02, 0x01].
            out.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn decode(input: &[u8], out: &mut [u32]) -> usize {
        let n_bytes = out.len() * 4;
        assert!(
            input.len() >= n_bytes,
            "Raw: need {n_bytes} bytes to decode {} values, but input has only {}",
            out.len(),
            input.len()
        );
        // `as_chunks::<4>()` views the bytes as a slice of `[u8; 4]` arrays (plus a
        // leftover tail, which is empty here because n_bytes is a multiple of 4).
        let (chunks, _empty_tail) = input[..n_bytes].as_chunks::<4>();
        for (value, bytes) in out.iter_mut().zip(chunks) {
            // e.g. [0x04, 0x03, 0x02, 0x01] becomes 0x0102_0304.
            *value = u32::from_le_bytes(*bytes);
        }
        n_bytes
    }
}
