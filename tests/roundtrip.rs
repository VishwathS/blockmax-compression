//! Round-trip tests: for every codec, decode(encode(x)) must give back x.

use codecs::{BlockCodec, Raw, VByte};
use proptest::prelude::*;

/// Encodes `input` with codec `C`, decodes it again, and checks that we get the same
/// values back and that `decode` consumed exactly the bytes `encode` wrote.
///
/// `C: BlockCodec` makes this generic: `roundtrip::<Raw>(..)` and
/// `roundtrip::<VByte>(..)` run the same checks on different codecs.
fn roundtrip<C: BlockCodec>(input: &[u32]) {
    let mut encoded = Vec::new();
    C::encode(input, &mut encoded);

    let mut decoded = vec![0u32; input.len()];
    let consumed = C::decode(&encoded, &mut decoded);

    assert_eq!(decoded, input, "decoded values differ from the input");
    assert_eq!(
        consumed,
        encoded.len(),
        "decode should consume exactly the bytes encode wrote"
    );
}

/// The fixed inputs from the stage plan, each with a name for error messages.
fn fixed_cases() -> Vec<(&'static str, Vec<u32>)> {
    let mut threes_then_200 = vec![3u32; 127];
    threes_then_200.push(200);

    vec![
        ("empty", vec![]),
        ("130 zeros", vec![0; 130]),
        ("5 x u32::MAX", vec![u32::MAX; 5]),
        ("127 threes then 200", threes_then_200),
        ("0..1000", (0..1000).collect()),
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
fn roundtrip_in_blocks<C: BlockCodec>(input: &[u32]) {
    let mut encoded = Vec::new();
    for block in input.chunks(C::BLOCK_SIZE) {
        C::encode(block, &mut encoded); // appends after the previous block
    }

    let mut decoded = vec![0u32; input.len()];
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
    let input: Vec<u32> = (0..1000u32).map(|i| i * i).collect();
    roundtrip_in_blocks::<Raw>(&input);
    roundtrip_in_blocks::<VByte>(&input);
}

// ---------- exact-byte tests ----------

/// Returns VByte's encoding of a single value.
fn vbyte_bytes(value: u32) -> Vec<u8> {
    let mut out = Vec::new();
    VByte::encode(&[value], &mut out);
    out
}

#[test]
fn vbyte_known_encodings() {
    assert_eq!(vbyte_bytes(5), [0x85]);
    assert_eq!(vbyte_bytes(130), [0x02, 0x81]);
    assert_eq!(vbyte_bytes(300), [0x2C, 0x82]);
    // u32::MAX is 32 one-bits: four full 7-bit groups (0x7F), then the top 4 bits
    // (0b1111 = 0x0F) with the last-byte flag set (0x8F).
    assert_eq!(vbyte_bytes(u32::MAX), [0x7F, 0x7F, 0x7F, 0x7F, 0x8F]);
    assert_eq!(vbyte_bytes(u32::MAX).len(), 5);
}

#[test]
fn vbyte_length_boundaries() {
    // 127 is the largest 1-byte value, and 128 is the smallest 2-byte value.
    assert_eq!(vbyte_bytes(127), [0xFF]);
    assert_eq!(vbyte_bytes(128), [0x00, 0x81]);
    // 16383 = 2^14 - 1 is the largest 2-byte value, and 16384 = 2^14 the smallest 3-byte one.
    assert_eq!(vbyte_bytes(16383), [0x7F, 0xFF]);
    assert_eq!(vbyte_bytes(16384), [0x00, 0x00, 0x81]);
    // The remaining boundaries, checked by length only.
    assert_eq!(vbyte_bytes((1 << 21) - 1).len(), 3);
    assert_eq!(vbyte_bytes(1 << 21).len(), 4);
    assert_eq!(vbyte_bytes((1 << 28) - 1).len(), 4);
    assert_eq!(vbyte_bytes(1 << 28).len(), 5);
}

#[test]
fn raw_is_little_endian() {
    let mut out = Vec::new();
    Raw::encode(&[0x0102_0304], &mut out);
    assert_eq!(out, [0x04, 0x03, 0x02, 0x01]);
}

#[test]
fn decode_ignores_trailing_bytes() {
    // Encode two values but decode only the first: VByte should stop after 5's single byte.
    let mut encoded = Vec::new();
    VByte::encode(&[5, 300], &mut encoded);
    let mut first = [0u32; 1];
    assert_eq!(VByte::decode(&encoded, &mut first), 1);
    assert_eq!(first, [5]);
}

#[test]
#[should_panic(expected = "input ended")]
fn vbyte_panics_on_truncated_input() {
    // 0x02 has the "last byte" flag off, so a second byte should follow, but none does.
    let mut out = [0u32; 1];
    VByte::decode(&[0x02], &mut out);
}

// ---------- property tests ----------
// proptest generates hundreds of random inputs per test. When one fails, it shrinks it
// to a minimal failing example and prints it.

/// Random values, biased so that every VByte length (1-5 bytes) shows up often.
/// A plain `any::<u32>()` would almost always be a 5-byte value.
fn mixed_u32() -> impl Strategy<Value = u32> {
    prop_oneof![
        0u32..128,            // 1 byte
        128u32..16_384,       // 2 bytes
        16_384u32..(1 << 21), // 3 bytes
        any::<u32>(),         // anything, mostly 4-5 bytes
    ]
}

proptest! {
    #[test]
    fn raw_roundtrips_random(input in prop::collection::vec(any::<u32>(), 0..1000)) {
        roundtrip::<Raw>(&input);
    }

    #[test]
    fn vbyte_roundtrips_random(input in prop::collection::vec(mixed_u32(), 0..1000)) {
        roundtrip::<VByte>(&input);
    }

    #[test]
    fn vbyte_roundtrips_random_uniform(input in prop::collection::vec(any::<u32>(), 0..1000)) {
        roundtrip::<VByte>(&input);
    }
}
