//! End-to-end test: one deterministic row through the registry, for every codec.
//!
//! Unlike the round-trip tests, the expected encoded SIZES here are computed
//! independently, from the format descriptions (widths found by a separate loop in
//! this file), not from the codecs' own code. So a codec that wrote extra or missing
//! bytes, but still decoded its own output, would fail.

use codecs::{
    BlockCodec, Bp128, Codec, GroupIndex, RandomAccessCodec, Raw, SimdBp128, SimdBp256, SimdBp512,
};

/// Long enough that SimdBp512 needs 2 metablocks (512 × 512 = 262,144 values each),
/// and not a multiple of any group size, so every codec has a partial last group.
const N: usize = 270_001;

/// A deterministic row in four kinds of 10,000-value stretches: all zeros, 4-bit
/// values, a sparse "block-max-like" mix, and full 8-bit values.
fn row() -> Vec<u16> {
    let mut x = 0x2545_F491_4F6C_DD1Du64; // xorshift64 state
    (0..N)
        .map(|i| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            match (i / 10_000) % 4 {
                0 => 0,
                1 => (x % 16) as u16,
                2 => match x % 100 {
                    0..90 => 0,
                    90..99 => (1 + (x >> 32) % 20) as u16,
                    _ => (21 + (x >> 32) % 235) as u16,
                },
                _ => (x % 256) as u16,
            }
        })
        .collect()
}

/// Bit width of each group of `group_len` values: computed here on its own, not by
/// any codec.
fn widths(values: &[u16], group_len: usize) -> Vec<usize> {
    values
        .chunks(group_len)
        .map(|group| {
            let max = group.iter().copied().max().unwrap_or(0);
            (16 - max.leading_zeros()) as usize
        })
        .collect()
}

/// The encoded size each codec's format description predicts for `values`.
fn expected_size(codec: Codec, values: &[u16]) -> usize {
    let n = values.len();
    match codec {
        Codec::Raw => n,
        Codec::VByte => values.iter().map(|&v| if v < 128 { 1 } else { 2 }).sum(),
        // One width byte per 128 values, then 16 bytes per bit of width.
        Codec::Bp128 => {
            let w = widths(values, 128);
            w.len() + 16 * w.iter().sum::<usize>()
        }
        Codec::SimdBp128 => simdbp_size(values, 8),
        Codec::SimdBp256 => simdbp_size(values, 16),
        Codec::SimdBp512 => simdbp_size(values, 32),
    }
}

/// SIMDBP: one 8 × stride-byte header per metablock of 16 × stride groups, then
/// 2 × stride bytes per bit of width per group.
fn simdbp_size(values: &[u16], stride: usize) -> usize {
    let group_len = 16 * stride;
    let w = widths(values, group_len);
    let metablocks = w.len().div_ceil(group_len);
    metablocks * 8 * stride + 2 * stride * w.iter().sum::<usize>()
}

#[test]
fn every_codec_end_to_end_through_the_registry() {
    let input = row();
    for codec in Codec::ALL {
        let name = codec.name();

        // Encode into a buffer that already holds a prefix (encode must append).
        let prefix = [0xA5u8; 3];
        let mut buffer = prefix.to_vec();
        codec.encode(&input, &mut buffer);
        let row_bytes = &buffer[prefix.len()..];
        assert_eq!(
            &buffer[..prefix.len()],
            prefix,
            "{name}: prefix overwritten"
        );
        assert_eq!(
            row_bytes.len(),
            expected_size(codec, &input),
            "{name}: size"
        );

        // Junk after the row must be ignored.
        let mut with_junk = row_bytes.to_vec();
        with_junk.extend_from_slice(&[0xFF; 100]);

        // Sequential decode.
        let mut decoded = vec![0u16; N];
        let consumed = codec.decode(&with_junk, &mut decoded);
        assert_eq!(consumed, row_bytes.len(), "{name}: bytes consumed");
        assert_eq!(decoded, input, "{name}: sequential decode");

        // Random access through the registry, into a buffer bigger than any group and
        // filled with a sentinel: only the group's own values may be written.
        let Some(group_len) = codec.group_len() else {
            assert_eq!(codec.decode_group(&with_junk, N, 0, &mut [0; 1]), None);
            continue; // VByte: no random access
        };
        let sentinel = 0xBEEF;
        let mut out = vec![sentinel; 512 + 16];
        let n_groups = N.div_ceil(group_len);
        for g in [0, 1, n_groups / 2, n_groups - 2, n_groups - 1] {
            out.fill(sentinel);
            let len = codec.decode_group(&with_junk, N, g, &mut out).unwrap();
            let expected = &input[g * group_len..((g + 1) * group_len).min(N)];
            assert_eq!(len, expected.len(), "{name}: group {g} length");
            assert_eq!(&out[..len], expected, "{name}: group {g} values");
            assert!(
                out[len..].iter().all(|&v| v == sentinel),
                "{name}: group {g} wrote past its end"
            );
        }
    }
}

/// Every group through `GroupIndex`, and the index's metadata against the row.
fn check_index<C: RandomAccessCodec>(input: &[u16]) {
    let mut row_bytes = Vec::new();
    C::encode(input, &mut row_bytes);
    let mut index = GroupIndex::<C>::new();
    index.build(&row_bytes, input.len());
    assert_eq!(index.num_groups(), input.len().div_ceil(C::GROUP_LEN));
    assert_eq!(index.encoded_len(), row_bytes.len());
    let mut out = vec![0u16; C::GROUP_LEN];
    let mut previous_end = None;
    for g in 0..index.num_groups() {
        let range = index.group_range(g);
        // Groups are back to back: each starts where the previous one ended.
        if let Some(end) = previous_end {
            assert_eq!(range.start, end, "group {g} doesn't follow the one before");
        }
        previous_end = Some(range.end);
        let len = index.decode_group(&row_bytes, g, &mut out);
        let expected = &input[g * C::GROUP_LEN..((g + 1) * C::GROUP_LEN).min(input.len())];
        assert_eq!(&out[..len], expected, "group {g}");
    }
    assert_eq!(previous_end, Some(row_bytes.len()));
}

#[test]
fn every_random_access_codec_through_a_group_index() {
    let input = row();
    check_index::<Raw>(&input);
    check_index::<Bp128>(&input);
    check_index::<SimdBp128>(&input);
    check_index::<SimdBp256>(&input);
    check_index::<SimdBp512>(&input);
}

#[test]
fn splitting_into_rows_changes_size_only_for_codecs_with_row_headers() {
    // Encoding 128-value pieces as separate rows (what the old size report did) costs
    // Raw, VByte, and Bp128 nothing extra, but SIMDBP pays one header per row. This is
    // why the size report now encodes each dataset as one row.
    let input = &row()[..100_000];
    for codec in Codec::ALL {
        let mut one_row = Vec::new();
        codec.encode(input, &mut one_row);
        let mut pieces = Vec::new();
        for piece in input.chunks(128) {
            codec.encode(piece, &mut pieces);
        }
        let header_per_row = match codec {
            Codec::Raw | Codec::VByte | Codec::Bp128 => 0,
            Codec::SimdBp128 => SimdBp128::HEADER_BYTES,
            Codec::SimdBp256 => SimdBp256::HEADER_BYTES,
            Codec::SimdBp512 => SimdBp512::HEADER_BYTES,
        };
        if header_per_row == 0 {
            assert_eq!(pieces.len(), one_row.len(), "{}", codec.name());
        } else {
            assert!(pieces.len() > one_row.len(), "{}", codec.name());
        }
    }
    // The typed and registry paths agree.
    let mut typed = Vec::new();
    SimdBp512::encode(input, &mut typed);
    let mut via_registry = Vec::new();
    Codec::SimdBp512.encode(input, &mut via_registry);
    assert_eq!(typed, via_registry);
}
