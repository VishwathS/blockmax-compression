//! `VByte` (variable-byte) encoding.
//!
//! Each number is split into 7-bit groups, starting from the **low-order** bits, and
//! each group goes into its own byte. The high bit (0x80) of every byte is a flag:
//! it's 1 on the **last** byte of a number and 0 on all earlier bytes. Small numbers
//! take fewer bytes:
//!
//! | value range        | bytes |
//! |--------------------|-------|
//! | 0 ..= 127          | 1     |
//! | 128 ..= 16_383     | 2     |
//! | 16_384 ..= 2^21-1  | 3     |
//! | 2^21 ..= 2^28-1    | 4     |
//! | 2^28 ..= u32::MAX  | 5     |
//!
//! Example: 300 is `0b10_0101100`. The low 7 bits are `0101100` = 0x2C (not last, so
//! the flag is 0). The remaining bits are `10` = 2, and with the "last" flag set that's
//! 0x82. So 300 encodes as `[0x2C, 0x82]`.

use crate::BlockCodec;

/// The high bit of a byte: set on the last byte of each number.
const LAST_BYTE_FLAG: u8 = 0x80;
/// The low 7 bits of a byte: the data bits.
const DATA_MASK: u8 = 0x7F;

/// Variable-byte codec: 1 to 5 bytes per `u32`, with smaller values using fewer bytes.
pub struct VByte;

impl BlockCodec for VByte {
    const BLOCK_SIZE: usize = 128;

    fn encode(input: &[u32], out: &mut Vec<u8>) {
        for &value in input {
            let mut remaining = value;
            // While more than 7 bits are left, write the low 7 bits with the flag off
            // ("more bytes follow"), then drop those 7 bits.
            while remaining > u32::from(DATA_MASK) {
                out.push(remaining as u8 & DATA_MASK);
                remaining >>= 7;
            }
            // At most 7 bits remain. Write them with the flag on: this is the last byte.
            out.push(remaining as u8 | LAST_BYTE_FLAG);
        }
    }

    fn decode(input: &[u8], out: &mut [u32]) -> usize {
        let mut pos = 0; // index of the next unread byte in `input`
        for slot in out.iter_mut() {
            let mut value: u32 = 0;
            let mut shift = 0; // where the next 7 data bits go in `value`
            loop {
                let byte = *input
                    .get(pos)
                    .expect("VByte: input ended in the middle of a value");
                pos += 1;
                // A u32 needs at most 5 bytes, so the 5th byte goes in at shift 28.
                // A larger shift means the input wasn't produced by `encode`.
                assert!(shift <= 28, "VByte: value is longer than 5 bytes");
                // Put this byte's 7 data bits above the bits we already have.
                value |= u32::from(byte & DATA_MASK) << shift;
                if byte & LAST_BYTE_FLAG != 0 {
                    break; // flag set: that was the last byte of this number
                }
                shift += 7;
            }
            *slot = value;
        }
        pos
    }
}
