//! Round-trip tests: for every codec, decode(encode(x)) must give back x.
//!
//! Every generated value is in 0..=255, the range the `BlockCodec` contract allows.

use codecs::{BlockCodec, Bp128, Codec, MAX_VALUE, Raw, SimdBp128, SimdBp256, SimdBp512, VByte};
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
        ("one value", vec![7]),
        ("128 zeros (one full Bp128 chunk)", vec![0; 128]),
        ("130 zeros", vec![0; 130]),
        ("5 x 255", vec![MAX_VALUE; 5]),
        ("127 threes then 200", threes_then_200),
        ("0..=255", (0..=MAX_VALUE).collect()),
        // 300 = 2 full chunks + 44 leftover, with small values (2-bit width).
        ("300 values in 0..=3", (0..300).map(|i| i % 4).collect()),
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

#[test]
fn bp128_roundtrips_fixed_cases() {
    for (name, input) in fixed_cases() {
        println!("Bp128: {name}");
        roundtrip::<Bp128>(&input);
    }
}

#[test]
fn simdbp_roundtrips_fixed_cases() {
    for (name, input) in fixed_cases() {
        println!("SimdBp: {name}");
        roundtrip::<SimdBp128>(&input);
        roundtrip::<SimdBp256>(&input);
        roundtrip::<SimdBp512>(&input);
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
    roundtrip_in_blocks::<Bp128>(&input);
    roundtrip_in_blocks::<SimdBp128>(&input);
    roundtrip_in_blocks::<SimdBp256>(&input);
    roundtrip_in_blocks::<SimdBp512>(&input);
}

/// Like `roundtrip_in_blocks`, but the pieces have different, arbitrary lengths
/// (not multiples of BLOCK_SIZE). Each `encode` call appends its own self-contained
/// section, and each `decode` must stop exactly at the end of its section.
fn roundtrip_concatenated<C: BlockCodec>(pieces: &[Vec<u16>]) {
    let mut encoded = Vec::new();
    for piece in pieces {
        C::encode(piece, &mut encoded);
    }

    let mut pos = 0;
    for piece in pieces {
        let mut decoded = vec![0u16; piece.len()];
        pos += C::decode(&encoded[pos..], &mut decoded);
        assert_eq!(&decoded, piece);
    }
    assert_eq!(pos, encoded.len(), "decoders should consume every byte");
}

#[test]
fn concatenated_encodes_decode_in_order() {
    let pieces = vec![
        vec![MAX_VALUE; 128],               // full chunk, width 8
        vec![1, 2, 3, 4, 5],                // partial chunk, width 3
        vec![],                             // empty: writes nothing
        vec![0; 200],                       // all zeros over 2 chunks
        (0..300).map(|i| i % 17).collect(), // 2 full chunks + 44, width 5
    ];
    roundtrip_concatenated::<Raw>(&pieces);
    roundtrip_concatenated::<VByte>(&pieces);
    roundtrip_concatenated::<Bp128>(&pieces);
    roundtrip_concatenated::<SimdBp128>(&pieces);
    roundtrip_concatenated::<SimdBp256>(&pieces);
    roundtrip_concatenated::<SimdBp512>(&pieces);
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

#[test]
#[should_panic(expected = "decoded value 16383 is above 255")]
fn vbyte_panics_on_a_two_byte_value_above_255() {
    // 0x7F (low 7 bits all set, more to come) then 0xFF (7 more bits, last byte) is
    // 14 one-bits = 16383. It's 2 bytes long, but `encode` never writes it.
    let mut out = [0u16; 1];
    VByte::decode(&[0x7F, 0xFF], &mut out);
}

#[test]
fn vbyte_accepts_the_largest_valid_two_byte_value() {
    // The boundary of the check above: 255 = [0x7F, 0x81] still decodes.
    let mut out = [0u16; 1];
    assert_eq!(VByte::decode(&[0x7F, 0x81], &mut out), 2);
    assert_eq!(out, [255]);
}

#[test]
#[should_panic(expected = "need 5 bytes")]
fn raw_panics_on_truncated_input() {
    Raw::decode(&[1, 2, 3, 4], &mut [0u16; 5]);
}

// ---------- Bp128 format tests ----------

/// Returns Bp128's encoding of `input`.
fn bp128_bytes(input: &[u16]) -> Vec<u8> {
    let mut out = Vec::new();
    Bp128::encode(input, &mut out);
    out
}

#[test]
fn bp128_empty_input_writes_nothing() {
    assert!(bp128_bytes(&[]).is_empty());
}

#[test]
fn bp128_all_zero_chunk_is_one_byte() {
    // Width 0: only the width byte is stored, no data.
    assert_eq!(bp128_bytes(&[0; 128]), [0]);
    // Two chunks (128 + 72), both zero: two width bytes.
    assert_eq!(bp128_bytes(&[0; 200]), [0, 0]);
}

#[test]
fn bp128_small_values_use_small_widths() {
    // Max 1 → width 1 → 1 width byte + 128 × 1 / 8 = 16 data bytes.
    let ones: Vec<u16> = (0..128).map(|i| i % 2).collect();
    let bytes = bp128_bytes(&ones);
    assert_eq!(bytes[0], 1, "width byte");
    assert_eq!(bytes.len(), 1 + 16);

    // Max 15 (a 4-bit quantized max) → width 4 → 1 + 64 bytes.
    let nibbles: Vec<u16> = (0..128).map(|i| i % 16).collect();
    let bytes = bp128_bytes(&nibbles);
    assert_eq!(bytes[0], 4);
    assert_eq!(bytes.len(), 1 + 64);
}

#[test]
fn bp128_full_range_uses_8_bits() {
    // One 255 forces the whole chunk to width 8: 1 + 128 bytes, slightly more than Raw.
    let mut input = vec![0u16; 128];
    input[77] = MAX_VALUE;
    let bytes = bp128_bytes(&input);
    assert_eq!(bytes[0], 8);
    assert_eq!(bytes.len(), 1 + 128);
}

#[test]
fn bp128_partial_chunk_is_padded_to_128() {
    // 5 values with max 5 → width 3. The chunk is padded to 128 values, so the data
    // is still 128 × 3 / 8 = 48 bytes.
    let bytes = bp128_bytes(&[1, 2, 3, 4, 5]);
    assert_eq!(bytes[0], 3);
    assert_eq!(bytes.len(), 1 + 48);

    // 129 values = one full chunk (width 1) + one chunk with a single value (width 2).
    // All width bytes come first: [1, 2], then 16 bytes, then 32 bytes.
    let mut input = vec![1u16; 128];
    input.push(2);
    let bytes = bp128_bytes(&input);
    assert_eq!(&bytes[..2], [1, 2]);
    assert_eq!(bytes.len(), 2 + 16 + 32);
}

#[test]
fn bp128_known_bytes() {
    // Pins the exact bytes so a change in the `bitpacking` crate's layout would be
    // caught. It also checks that debug builds (which use the crate's plain scalar
    // packer) and release builds (its SIMD packer) agree: run `cargo test --release`.
    // BitPacker4x puts value i in lane i % 4 at bit i / 4, and writes lane j's bits
    // as the little-endian u32 at bytes 4j..4j+4 (for width 1).
    // Value 0 → lane 0, bit 0 → byte 4 × 0 = 0 gets 0b01.
    // Value 5 → lane 1, bit 1 → byte 4 × 1 = 4 gets 0b10.
    let mut input = vec![0u16; 128];
    input[0] = 1;
    input[5] = 1;
    let mut expected = vec![0u8; 1 + 16];
    expected[0] = 1; // width byte
    expected[1] = 0b01; // data byte 0
    expected[1 + 4] = 0b10; // data byte 4
    assert_eq!(bp128_bytes(&input), expected);
}

#[test]
#[should_panic(expected = "input ended in the middle of a chunk")]
fn bp128_panics_on_truncated_input() {
    let mut bytes = bp128_bytes(&[MAX_VALUE; 128]);
    bytes.pop(); // drop the last data byte
    Bp128::decode(&bytes, &mut [0u16; 128]);
}

#[test]
#[should_panic(expected = "width bytes")]
fn bp128_panics_on_missing_width_bytes() {
    // 129 values need 2 width bytes, but the input has only 1.
    Bp128::decode(&[0], &mut [0u16; 129]);
}

#[test]
#[should_panic(expected = "invalid bit width 9")]
fn bp128_panics_on_invalid_width() {
    // No value <= 255 needs 9 bits, so `encode` never writes this width byte.
    Bp128::decode(&[9; 200], &mut [0u16; 128]);
}

// ---------- range check (values above 255) ----------
// `encode` uses `assert!`, not `debug_assert!`, so these must pass in both
// `cargo test` and `cargo test --release`.

#[test]
#[should_panic(expected = "above 255")]
fn raw_rejects_values_above_255() {
    Raw::encode(&[256], &mut Vec::new());
}

#[test]
#[should_panic(expected = "above 255")]
fn vbyte_rejects_values_above_255() {
    VByte::encode(&[256], &mut Vec::new());
}

#[test]
#[should_panic(expected = "above 255")]
fn simdbp_rejects_values_above_255() {
    // Parker's `compress` would silently keep only the low 8 bits (300 -> 44).
    SimdBp512::encode(&[300], &mut Vec::new());
}

#[test]
#[should_panic(expected = "above 255")]
fn bp128_rejects_values_above_255() {
    // The bad value sits in the second chunk, after a valid first chunk.
    let mut input = vec![0u16; 200];
    input[150] = 256;
    Bp128::encode(&input, &mut Vec::new());
}

// ---------- codec registry ----------
// `Codec` (src/registry.rs) forwards to the codec types. These tests check that the
// list is complete, the names are stable, and that going through the registry gives
// exactly the same bytes and byte counts as calling the codec type directly.

#[test]
fn registry_lists_exactly_the_implemented_codecs() {
    // Update this list (and only this list) when a codec is added.
    assert_eq!(
        Codec::ALL,
        [
            Codec::Raw,
            Codec::VByte,
            Codec::Bp128,
            Codec::SimdBp128,
            Codec::SimdBp256,
            Codec::SimdBp512
        ]
    );
    let names: Vec<&str> = Codec::ALL.iter().map(|codec| codec.name()).collect();
    // These exact strings appear in reports, so they must not change by accident.
    assert_eq!(
        names,
        [
            "Raw",
            "VByte",
            "Bp128",
            "SimdBp128",
            "SimdBp256",
            "SimdBp512"
        ]
    );
}

#[test]
fn registry_names_are_unique_and_found_by_from_name() {
    for codec in Codec::ALL {
        // Each name maps back to its own codec, so no two codecs share a name.
        assert_eq!(Codec::from_name(codec.name()), Some(codec));
        // Lookup ignores case, e.g. "bp128" and "BP128" both find Bp128.
        assert_eq!(Codec::from_name(&codec.name().to_lowercase()), Some(codec));
        assert_eq!(Codec::from_name(&codec.name().to_uppercase()), Some(codec));
    }
    assert_eq!(Codec::from_name("NoSuchCodec"), None);
    assert_eq!(Codec::from_name(""), None);
}

/// Encodes with the codec type directly, without the registry. Written out by hand
/// on purpose: if a registry `match` arm pointed at the wrong codec, the bytes from
/// the registry would differ from these.
fn encode_directly(codec: Codec, input: &[u16]) -> Vec<u8> {
    let mut out = Vec::new();
    match codec {
        Codec::Raw => Raw::encode(input, &mut out),
        Codec::VByte => VByte::encode(input, &mut out),
        Codec::Bp128 => Bp128::encode(input, &mut out),
        Codec::SimdBp128 => SimdBp128::encode(input, &mut out),
        Codec::SimdBp256 => SimdBp256::encode(input, &mut out),
        Codec::SimdBp512 => SimdBp512::encode(input, &mut out),
    }
    out
}

#[test]
fn registry_matches_the_codec_types() {
    assert_eq!(Codec::Raw.block_size(), Raw::BLOCK_SIZE);
    assert_eq!(Codec::VByte.block_size(), VByte::BLOCK_SIZE);
    assert_eq!(Codec::Bp128.block_size(), Bp128::BLOCK_SIZE);
    assert_eq!(Codec::SimdBp128.block_size(), 128);
    assert_eq!(Codec::SimdBp256.block_size(), 256);
    assert_eq!(Codec::SimdBp512.block_size(), 512);

    for codec in Codec::ALL {
        for (name, input) in fixed_cases() {
            println!("{}: {name}", codec.name()); // shown only if the test fails
            let mut encoded = Vec::new();
            codec.encode(&input, &mut encoded);
            assert_eq!(encoded, encode_directly(codec, &input));

            // Same round-trip checks as `roundtrip::<C>`, through the registry.
            let mut decoded = vec![0u16; input.len()];
            let consumed = codec.decode(&encoded, &mut decoded);
            assert_eq!(decoded, input);
            assert_eq!(consumed, encoded.len());
        }
    }
}

#[test]
fn registry_concatenated_encodes_decode_in_order() {
    // Every fixed case appended to one buffer, then decoded back one by one using
    // the byte counts `decode` returns, for every codec in the registry.
    let pieces: Vec<Vec<u16>> = fixed_cases().into_iter().map(|(_, v)| v).collect();
    for codec in Codec::ALL {
        let mut buffer = Vec::new();
        for piece in &pieces {
            codec.encode(piece, &mut buffer);
        }
        let mut pos = 0;
        for piece in &pieces {
            let mut decoded = vec![0u16; piece.len()];
            pos += codec.decode(&buffer[pos..], &mut decoded);
            assert_eq!(&decoded, piece, "{}", codec.name());
        }
        assert_eq!(pos, buffer.len(), "{}: bytes left over", codec.name());
    }
}

#[test]
fn registry_encode_keeps_the_range_check() {
    // Every codec in the registry must reject 256, in debug and release builds.
    for codec in Codec::ALL {
        let result = std::panic::catch_unwind(|| codec.encode(&[1, 256], &mut Vec::new()));
        assert!(result.is_err(), "{} accepted 256", codec.name());
    }
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

    #[test]
    fn bp128_roundtrips_random(input in block()) {
        roundtrip::<Bp128>(&input);
    }

    #[test]
    fn simdbp_roundtrips_random(input in block()) {
        roundtrip::<SimdBp128>(&input);
        roundtrip::<SimdBp256>(&input);
        roundtrip::<SimdBp512>(&input);
    }

    /// The same random input through every codec in the registry.
    #[test]
    fn registry_roundtrips_random(input in block()) {
        for codec in Codec::ALL {
            let mut encoded = Vec::new();
            codec.encode(&input, &mut encoded);
            let mut decoded = vec![0u16; input.len()];
            let consumed = codec.decode(&encoded, &mut decoded);
            prop_assert_eq!(&decoded, &input, "{}", codec.name());
            prop_assert_eq!(consumed, encoded.len(), "{}", codec.name());
        }
    }

    /// Small values (like 4-bit quantized maxes) with a random max width, so Bp128
    /// gets tested at every width from 0 to 8, not mostly at width 8.
    #[test]
    fn bp128_roundtrips_random_widths(
        input in (0u32..=8).prop_flat_map(|bits| {
            let max = ((1u32 << bits) - 1) as u16; // largest value with `bits` bits
            prop::collection::vec(0u16..=max, 0..1000)
        })
    ) {
        roundtrip::<Bp128>(&input);
        roundtrip::<SimdBp128>(&input);
        roundtrip::<SimdBp256>(&input);
        roundtrip::<SimdBp512>(&input);
    }
}
