//! Round-trip tests: for every codec, decode(encode(x)) must give back x.
//!
//! Every generated value is in 0..=255, the range the `BlockCodec` contract allows.

use codecs::{BlockCodec, MAX_VALUE, Raw, VByte};
use proptest::prelude::*;

/// Encodes `input` with codec `C`, decodes it again, and checks that we get the same
/// values back and that `decode` consumed exactly the bytes `encode` wrote.
///
/// `C: BlockCodec` makes this generic: `roundtrip::<Raw>(..)` and
/// `roundtrip::<VByte>(..)` run the same checks on different codecs.
fn roundtrip<C: BlockCodec>(input: &[u16]) {
    let mut encoded = Vec::new();
    C::encode(input, &mut encoded);

    let mut decoded = vec![0u16; input.len()];
    let consumed = C::decode(&encoded, &mut decoded);

    assert_eq!(decoded, input, "decoded values differ from the input");
    assert_eq!(
        consumed,
        encoded.len(),
        "decode should consume exactly the bytes encode wrote"
    );
}

/// Fixed inputs, each with a name for error messages.
fn fixed_cases() -> Vec<(&'static str, Vec<u16>)> {
    let mut threes_then_200 = vec![3u16; 127];
    threes_then_200.push(200);

    vec![
        ("empty", vec![]),
        ("130 zeros", vec![0; 130]),
        ("5 x 255", vec![MAX_VALUE; 5]),
        ("127 threes then 200", threes_then_200),
        ("0..=255", (0..=MAX_VALUE).collect()),
    ]
}

#[test]
fn raw_roundtrips_fixed_cases() {
    for (name, input) in fixed_cases() {
        println!("Raw: {name}"); // shown only if the test fails
        roundtrip::<Raw>(&input);
    }
}

#[test]
fn vbyte_roundtrips_fixed_cases() {
    for (name, input) in fixed_cases() {
        println!("VByte: {name}");
        roundtrip::<VByte>(&input);
    }
}

/// Encodes a long input as several BLOCK_SIZE blocks written into one buffer, then
/// decodes them one by one, using each `decode`'s return value to find the next block.
fn roundtrip_in_blocks<C: BlockCodec>(input: &[u16]) {
    let mut encoded = Vec::new();
    for block in input.chunks(C::BLOCK_SIZE) {
        C::encode(block, &mut encoded); // appends after the previous block
    }

    let mut decoded = vec![0u16; input.len()];
    let mut pos = 0; // byte offset where the next block starts
    for block in decoded.chunks_mut(C::BLOCK_SIZE) {
        pos += C::decode(&encoded[pos..], block);
    }

    assert_eq!(decoded, input);
    assert_eq!(pos, encoded.len());
}

#[test]
fn blocks_can_be_decoded_back_to_back() {
    // 1000 values is not a multiple of 128, so the last block is a partial one.
    // `% 256` keeps every value in 0..=255, with a mix of 1- and 2-byte VByte values.
    let input: Vec<u16> = (0..1000u32).map(|i| (i * i % 256) as u16).collect();
    roundtrip_in_blocks::<Raw>(&input);
    roundtrip_in_blocks::<VByte>(&input);
}

// ---------- exact-byte tests ----------

/// Returns VByte's encoding of a single value.
fn vbyte_bytes(value: u16) -> Vec<u8> {
    let mut out = Vec::new();
    VByte::encode(&[value], &mut out);
    out
}

#[test]
fn vbyte_known_encodings() {
    assert_eq!(vbyte_bytes(0), [0x80]);
    assert_eq!(vbyte_bytes(5), [0x85]);
    assert_eq!(vbyte_bytes(130), [0x02, 0x81]);
    // 255 is 8 one-bits: the low 7 (0x7F), then the top bit (1) with the flag set (0x81).
    assert_eq!(vbyte_bytes(255), [0x7F, 0x81]);
}

#[test]
fn vbyte_length_boundaries() {
    // 127 is the largest 1-byte value, and 128 is the smallest 2-byte value.
    assert_eq!(vbyte_bytes(127), [0xFF]);
    assert_eq!(vbyte_bytes(128), [0x00, 0x81]);
}

#[test]
fn raw_is_one_byte_per_value() {
    let mut out = Vec::new();
    Raw::encode(&[0, 5, 200, 255], &mut out);
    assert_eq!(out, [0, 5, 200, 255]);
}

#[test]
fn decode_ignores_trailing_bytes() {
    // Encode two values but decode only the first: VByte should stop after 5's single byte.
    let mut encoded = Vec::new();
    VByte::encode(&[5, 200], &mut encoded);
    let mut first = [0u16; 1];
    assert_eq!(VByte::decode(&encoded, &mut first), 1);
    assert_eq!(first, [5]);
}

#[test]
#[should_panic(expected = "input ended")]
fn vbyte_panics_on_truncated_input() {
    // 0x02 has the "last byte" flag off, so a second byte should follow, but none does.
    let mut out = [0u16; 1];
    VByte::decode(&[0x02], &mut out);
}

#[test]
#[should_panic(expected = "longer than 2 bytes")]
fn vbyte_panics_on_three_byte_value() {
    // Two bytes without the flag, then a third: no value <= 255 encodes like this.
    let mut out = [0u16; 1];
    VByte::decode(&[0x00, 0x00, 0x81], &mut out);
}

// The `debug_assert!` in `encode` only exists in debug builds, so these tests are
// compiled only when debug assertions are on (the default for `cargo test`).
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "above 255")]
fn raw_rejects_values_above_255_in_debug() {
    Raw::encode(&[256], &mut Vec::new());
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "above 255")]
fn vbyte_rejects_values_above_255_in_debug() {
    VByte::encode(&[256], &mut Vec::new());
}

// ---------- property tests ----------
// proptest generates hundreds of random inputs per test. When one fails, it shrinks it
// to a minimal failing example and prints it.

/// A random block: up to 1000 values, each in 0..=255. Half of that range is 1-byte
/// VByte values and half is 2-byte, so both lengths show up often.
fn block() -> impl Strategy<Value = Vec<u16>> {
    prop::collection::vec(0u16..=MAX_VALUE, 0..1000)
}

proptest! {
    #[test]
    fn raw_roundtrips_random(input in block()) {
        roundtrip::<Raw>(&input);
    }

    #[test]
    fn vbyte_roundtrips_random(input in block()) {
        roundtrip::<VByte>(&input);
    }
}
