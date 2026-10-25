//! `Raw`: no compression. Each value is stored as one byte.
//!
//! Every value is at most 255 (see [`BlockCodec`]), so one byte is enough. This is the
//! same layout as Parker's dense `BlockData::Raw(Vec<u8>)`, and it's the 1-byte-per-value
//! baseline every other codec is compared against.

use crate::{BlockCodec, MAX_VALUE};

/// Stores each value as a single byte.
pub struct Raw;

impl BlockCodec for Raw {
    const BLOCK_SIZE: usize = 128;

    fn encode(input: &[u16], out: &mut Vec<u8>) {
        // We know the exact output size, so grow the buffer once up front.
        out.reserve(input.len());
        for &value in input {
            debug_assert!(
                value <= MAX_VALUE,
                "Raw: value {value} is above {MAX_VALUE}"
            );
            // `as u8` keeps the low 8 bits, which is the whole value when it's <= 255.
            out.push(value as u8);
        }
    }

    fn decode(input: &[u8], out: &mut [u16]) -> usize {
        let n_bytes = out.len(); // one byte per value
        assert!(
            input.len() >= n_bytes,
            "Raw: need {n_bytes} bytes to decode {n_bytes} values, but input has only {}",
            input.len()
        );
        for (value, &byte) in out.iter_mut().zip(&input[..n_bytes]) {
            // `u16::from` widens the byte losslessly (e.g. 200u8 becomes 200u16).
            *value = u16::from(byte);
        }
        n_bytes
    }
}
