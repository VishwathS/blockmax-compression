//! `Raw`: no compression. Each value is stored as one byte.
//!
//! Every value is at most 255 (see [`BlockCodec`]), so one byte is enough. This is the
//! same layout as Parker's dense `BlockData::Raw(Vec<u8>)`, and it's the 1-byte-per-value
//! baseline every other codec is compared against.

use std::ops::Range;

use crate::{BlockCodec, RandomAccessCodec, check_values};

/// Stores each value as a single byte.
pub struct Raw;

impl BlockCodec for Raw {
    const BLOCK_SIZE: usize = 128;

    fn encode(input: &[u16], out: &mut Vec<u8>) {
        // Checked first, in release builds too, so nothing is written for a bad input.
        // Without it, `as u8` below would silently turn 256 into 0.
        check_values("Raw", input);
        // We know the exact output size, so grow the buffer once up front.
        out.reserve(input.len());
        for &value in input {
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

/// Random access in 128-value groups. Every value is one byte, so group g's data is
/// simply bytes `128 × g ..` and locating it is **O(1)** arithmetic with no metadata.
/// (Any group size would work; 128 matches `BLOCK_SIZE` and `Bp128`.)
impl RandomAccessCodec for Raw {
    const GROUP_LEN: usize = 128;

    fn group_range(input: &[u8], n: usize, group: usize) -> Range<usize> {
        let len = Self::group_len(n, group); // also checks the group index
        let start = group * Self::GROUP_LEN;
        let end = start + len;
        assert!(
            end <= input.len(),
            "Raw: input ended in the middle of group {group}"
        );
        start..end
    }

    fn locate_groups(input: &[u8], n: usize, starts: &mut Vec<usize>) {
        assert!(
            input.len() >= n,
            "Raw: need {n} bytes for {n} values, but input has only {}",
            input.len()
        );
        starts.clear();
        // Group starts are 0, 128, 256, ..., and the row ends at byte n.
        starts.extend((0..n).step_by(Self::GROUP_LEN));
        starts.push(n);
    }

    fn decode_group_bytes(data: &[u8], out: &mut [u16]) {
        assert_eq!(
            data.len(),
            out.len(),
            "Raw: a group of {} values needs exactly that many bytes",
            out.len()
        );
        for (value, &byte) in out.iter_mut().zip(data) {
            *value = u16::from(byte);
        }
    }
}
