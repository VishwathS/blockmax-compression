//! `VByte` (variable-byte) encoding.
//!
//! Each number is split into 7-bit groups, starting from the **low-order** bits, and
//! each group goes into its own byte. The high bit (0x80) of every byte is a flag:
//! it's 1 on the **last** byte of a number and 0 on all earlier bytes. Values are at
//! most 255 (see [`BlockCodec`]), so a number takes 1 or 2 bytes:
//!
//! | value range  | bytes |
//! |--------------|-------|
//! | 0 ..= 127    | 1     |
//! | 128 ..= 255  | 2     |
//!
//! Example: 130 is `0b1_0000010`. The low 7 bits are `0000010` = 0x02 (not last, so
//! the flag is 0). The remaining bit is `1`, and with the "last" flag set that's 0x81.
//! So 130 encodes as `[0x02, 0x81]`.

use crate::{BlockCodec, MAX_VALUE};

/// The high bit of a byte: set on the last byte of each number.
const LAST_BYTE_FLAG: u8 = 0x80;
/// The low 7 bits of a byte: the data bits.
const DATA_MASK: u8 = 0x7F;

/// Variable-byte codec: 1 byte for values up to 127, 2 bytes for 128..=255.
pub struct VByte;

impl BlockCodec for VByte {
    const BLOCK_SIZE: usize = 128;

    fn encode(input: &[u16], out: &mut Vec<u8>) {
        for &value in input {
            // `assert!` (not `debug_assert!`) so release builds check too. Without it,
            // a value like 20000 would encode as 3 bytes that `decode` then rejects.
            assert!(
                value <= MAX_VALUE,
                "VByte: value {value} is above {MAX_VALUE}"
            );
            let mut remaining = value;
            // While more than 7 bits are left, write the low 7 bits with the flag off
            // ("more bytes follow"), then drop those 7 bits. For values <= 255 this
            // loop runs at most once.
            while remaining > u16::from(DATA_MASK) {
                out.push(remaining as u8 & DATA_MASK);
                remaining >>= 7;
            }
            // At most 7 bits remain. Write them with the flag on: this is the last byte.
            out.push(remaining as u8 | LAST_BYTE_FLAG);
        }
    }

    fn decode(input: &[u8], out: &mut [u16]) -> usize {
        let mut pos = 0; // index of the next unread byte in `input`
        for slot in out.iter_mut() {
            let mut value: u16 = 0;
            let mut shift = 0; // where the next 7 data bits go in `value`
            loop {
                let byte = *input
                    .get(pos)
                    .expect("VByte: input ended in the middle of a value");
                pos += 1;
                // A value <= 255 needs at most 2 bytes, so the 2nd byte goes in at shift 7.
                // A larger shift means the input wasn't produced by `encode`. (This
                // also keeps `<< shift` from pushing bits off the top of the u16.)
                assert!(shift <= 7, "VByte: value is longer than 2 bytes");
                // Put this byte's 7 data bits above the bits we already have.
                value |= u16::from(byte & DATA_MASK) << shift;
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
