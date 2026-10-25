//! Random-access tests: decoding one group must give exactly the same values as
//! decoding the whole row and slicing out that group.
//!
//! Every generated value is in 0..=255, the range the `BlockCodec` contract allows.

use codecs::{
    BlockCodec, Bp128, Codec, GroupIndex, MAX_VALUE, RandomAccessCodec, Raw, SimdBp128, SimdBp256,
    SimdBp512,
};
use proptest::prelude::*;

/// Encodes `input` with codec `C`, then checks every group two ways (one-shot
/// `decode_group` and through a `GroupIndex`) against the sequential `decode`.
fn check_all_groups<C: RandomAccessCodec>(input: &[u16]) {
    let n = input.len();
    let mut encoded = Vec::new();
    C::encode(input, &mut encoded);

    let mut sequential = vec![0u16; n];
    let consumed = C::decode(&encoded, &mut sequential);
    assert_eq!(sequential, input);

    let mut index = GroupIndex::<C>::new();
    index.build(&encoded, n);
    assert_eq!(index.n(), n);
    assert_eq!(index.num_groups(), C::num_groups(n));
    // The index's idea of where the row ends must match what `decode` consumed.
    assert_eq!(index.encoded_len(), consumed);

    let mut group_out = vec![0u16; C::GROUP_LEN];
    for g in 0..C::num_groups(n) {
        let expected = &sequential[g * C::GROUP_LEN..((g + 1) * C::GROUP_LEN).min(n)];

        let len = C::decode_group(&encoded, n, g, &mut group_out);
        assert_eq!(len, expected.len(), "group {g}: wrong length (one-shot)");
        assert_eq!(
            &group_out[..len],
            expected,
            "group {g}: wrong values (one-shot)"
        );

        let len = index.decode_group(&encoded, g, &mut group_out);
        assert_eq!(len, expected.len(), "group {g}: wrong length (index)");
        assert_eq!(
            &group_out[..len],
            expected,
            "group {g}: wrong values (index)"
        );

        // Both ways of locating a group must agree on the bytes.
        assert_eq!(C::group_range(&encoded, n, g), index.group_range(g));
    }
}

/// 4 full Bp128 groups plus a 44-value partial one, each with a different width:
/// group 0 all zeros (width 0), group 1 max 5 (width 3), group 2 max 255 (width 8),
/// group 3 max 1 (width 1), group 4 (44 values) max 15 (width 4).
fn mixed_widths() -> Vec<u16> {
    let mut v = Vec::new();
    v.extend(std::iter::repeat_n(0, 128));
    v.extend((0..128).map(|i| (i % 6) as u16));
    v.extend((0..128).map(|i| (i * 2) as u16)); // 0, 2, ..., 254
    v[2 * 128 + 7] = MAX_VALUE; // and one 255
    v.extend((0..128).map(|i| (i % 2) as u16));
    v.extend((0..44).map(|i| (i % 16) as u16));
    v
}

// ---------- fixed cases ----------

#[test]
fn bp128_first_middle_and_last_group() {
    let input = mixed_widths();
    let mut encoded = Vec::new();
    Bp128::encode(&input, &mut encoded);
    let n = input.len();
    let mut out = [0u16; Bp128::GROUP_LEN];

    let len = Bp128::decode_group(&encoded, n, 0, &mut out);
    assert_eq!(&out[..len], &input[0..128]);
    let len = Bp128::decode_group(&encoded, n, 2, &mut out);
    assert_eq!(&out[..len], &input[256..384]);
    let len = Bp128::decode_group(&encoded, n, 4, &mut out);
    assert_eq!(&out[..len], &input[512..556]);
}

#[test]
fn bp128_mixed_widths_have_the_expected_offsets() {
    let input = mixed_widths();
    let mut encoded = Vec::new();
    Bp128::encode(&input, &mut encoded);
    assert_eq!(&encoded[..5], &[0, 3, 8, 1, 4], "width bytes");

    // Data starts after the 5 width bytes; group g takes 16 × width bytes.
    let mut index = GroupIndex::<Bp128>::new();
    index.build(&encoded, input.len());
    let starts = [5, 5, 5 + 48, 5 + 48 + 128, 5 + 48 + 128 + 16];
    let widths = [0, 3, 8, 1, 4];
    for g in 0..5 {
        let range = starts[g]..starts[g] + 16 * widths[g];
        assert_eq!(index.group_range(g), range, "group {g} (index)");
        assert_eq!(
            Bp128::group_range(&encoded, input.len(), g),
            range,
            "group {g}"
        );
    }
    assert_eq!(index.encoded_len(), encoded.len());
    check_all_groups::<Bp128>(&input);
}

/// Reads groups backwards, jumping around, and repeating, with codec `C`. Group g has
/// values up to (2g mod 9) bits, so neighboring groups have different sizes.
fn check_out_of_order<C: RandomAccessCodec>() {
    let gl = C::GROUP_LEN;
    let n = 4 * gl + gl / 3; // 4 full groups + a partial one
    let input: Vec<u16> = (0..n)
        .map(|i| {
            let max = (1u32 << ((i / gl) * 2 % 9)) - 1; // widths 0, 2, 4, 6, 8
            (i as u32 * 37 % (max + 1)) as u16
        })
        .collect();
    let mut encoded = Vec::new();
    C::encode(&input, &mut encoded);
    let mut index = GroupIndex::<C>::new();
    index.build(&encoded, n);
    let mut out = vec![0u16; gl];
    for g in [4, 0, 3, 1, 2, 2, 0, 4] {
        let expected = &input[g * gl..((g + 1) * gl).min(n)];
        let len = index.decode_group(&encoded, g, &mut out);
        assert_eq!(&out[..len], expected, "group {g} (index)");
        let len = C::decode_group(&encoded, n, g, &mut out);
        assert_eq!(&out[..len], expected, "group {g} (one-shot)");
    }
}

#[test]
fn every_codec_can_read_groups_out_of_order() {
    check_out_of_order::<Raw>();
    check_out_of_order::<Bp128>();
    check_out_of_order::<SimdBp128>();
    check_out_of_order::<SimdBp256>();
    check_out_of_order::<SimdBp512>();
}

#[test]
fn groups_can_be_read_out_of_order() {
    let input = mixed_widths();
    let n = input.len();
    let mut encoded = Vec::new();
    Bp128::encode(&input, &mut encoded);
    let mut index = GroupIndex::<Bp128>::new();
    index.build(&encoded, n);

    let mut out = [0u16; Bp128::GROUP_LEN];
    // Backwards, jumping around, and repeating a group.
    for g in [4, 0, 3, 1, 2, 2, 0, 4] {
        let expected = &input[g * 128..((g + 1) * 128).min(n)];
        let len = index.decode_group(&encoded, g, &mut out);
        assert_eq!(&out[..len], expected, "group {g} (index)");
        let len = Bp128::decode_group(&encoded, n, g, &mut out);
        assert_eq!(&out[..len], expected, "group {g} (one-shot)");
    }
}

#[test]
fn all_zero_groups_have_no_data_and_decode_to_zeros() {
    // Zeros, then a non-zero group, then zeros again (partial).
    let mut input = vec![0u16; 128];
    input.extend(std::iter::repeat_n(9, 128));
    input.extend(std::iter::repeat_n(0, 50));
    let mut encoded = Vec::new();
    Bp128::encode(&input, &mut encoded);

    // An all-zero group is an empty byte range (width 0 → no data bytes).
    assert!(Bp128::group_range(&encoded, input.len(), 0).is_empty());
    assert!(Bp128::group_range(&encoded, input.len(), 2).is_empty());

    // The output buffer starts full of garbage; the group's values must be overwritten.
    let mut out = [999u16; Bp128::GROUP_LEN];
    let len = Bp128::decode_group(&encoded, input.len(), 0, &mut out);
    assert_eq!(len, 128);
    assert!(out.iter().all(|&v| v == 0));

    check_all_groups::<Bp128>(&input);
    check_all_groups::<Bp128>(&[0; 1000]);
}

#[test]
fn partial_last_group_writes_only_its_values() {
    let input: Vec<u16> = (0..300).map(|i| (i % 200) as u16).collect();
    check_all_groups::<Bp128>(&input);
    check_all_groups::<Raw>(&input);
    check_all_groups::<SimdBp128>(&input);
    check_all_groups::<SimdBp256>(&input);
    check_all_groups::<SimdBp512>(&input);

    let mut encoded = Vec::new();
    Bp128::encode(&input, &mut encoded);
    assert_eq!(Bp128::group_len(300, 2), 44);

    let mut out = [777u16; Bp128::GROUP_LEN];
    let len = Bp128::decode_group(&encoded, 300, 2, &mut out);
    assert_eq!(len, 44);
    assert_eq!(&out[..44], &input[256..300]);
    // Past the group's length, the buffer is left alone (no padding zeros written).
    assert!(out[44..].iter().all(|&v| v == 777));

    // A buffer of exactly the group's length is enough.
    let mut exact = [0u16; 44];
    assert_eq!(Bp128::decode_group(&encoded, 300, 2, &mut exact), 44);
    assert_eq!(&exact[..], &input[256..300]);
}

#[test]
fn single_value_and_exact_multiple_rows() {
    for input in [
        vec![200],
        vec![5; 128],
        vec![17; 256],
        vec![9; 512],
        vec![1; 1024],
    ] {
        check_all_groups::<Bp128>(&input);
        check_all_groups::<Raw>(&input);
        check_all_groups::<SimdBp128>(&input);
        check_all_groups::<SimdBp256>(&input);
        check_all_groups::<SimdBp512>(&input);
    }
}

#[test]
fn empty_row_has_no_groups() {
    let mut encoded = Vec::new();
    Bp128::encode(&[], &mut encoded);
    let mut index = GroupIndex::<Bp128>::new();
    index.build(&encoded, 0);
    assert_eq!(index.num_groups(), 0);
    assert_eq!(index.encoded_len(), 0);
    assert_eq!(Bp128::num_groups(0), 0);
}

#[test]
fn raw_groups_are_plain_byte_ranges() {
    let input: Vec<u16> = (0..300).map(|i| (i % 256) as u16).collect();
    let mut encoded = Vec::new();
    Raw::encode(&input, &mut encoded);
    assert_eq!(Raw::group_range(&encoded, 300, 0), 0..128);
    assert_eq!(Raw::group_range(&encoded, 300, 1), 128..256);
    assert_eq!(Raw::group_range(&encoded, 300, 2), 256..300);
    check_all_groups::<Raw>(&input);
}

// ---------- several rows in one buffer ----------

/// Appends every row to one buffer with codec `C`, then reads each row back group by
/// group, reusing one `GroupIndex` and one output buffer for all rows.
fn check_multiple_rows<C: RandomAccessCodec>(rows: &[Vec<u16>]) {
    let mut buffer = Vec::new();
    for row in rows {
        C::encode(row, &mut buffer);
    }

    let mut index = GroupIndex::<C>::new();
    let mut out = vec![0u16; C::GROUP_LEN];
    let mut row_start = 0;
    for (r, row) in rows.iter().enumerate() {
        // `input` starts at this row; the rows after it are just trailing bytes.
        let input = &buffer[row_start..];
        index.build(input, row.len());

        // Read the groups backwards to make sure nothing depends on order.
        for g in (0..index.num_groups()).rev() {
            let expected = &row[g * C::GROUP_LEN..((g + 1) * C::GROUP_LEN).min(row.len())];
            let len = index.decode_group(input, g, &mut out);
            assert_eq!(&out[..len], expected, "row {r} group {g} (index)");
            let len = C::decode_group(input, row.len(), g, &mut out);
            assert_eq!(&out[..len], expected, "row {r} group {g} (one-shot)");
        }

        // The next row starts where this one ends; the index and decode must agree.
        let mut sequential = vec![0u16; row.len()];
        let consumed = C::decode(input, &mut sequential);
        assert_eq!(index.encoded_len(), consumed, "row {r}");
        row_start += consumed;
    }
    assert_eq!(row_start, buffer.len(), "all rows together fill the buffer");
}

#[test]
fn multiple_rows_in_one_buffer() {
    // Three rows of different lengths and widths.
    let rows: Vec<Vec<u16>> = vec![
        mixed_widths(),
        (0..130).map(|i| (i % 3) as u16).collect(),
        vec![MAX_VALUE; 70],
    ];
    check_multiple_rows::<Bp128>(&rows);
    check_multiple_rows::<Raw>(&rows);
    check_multiple_rows::<SimdBp128>(&rows);
    check_multiple_rows::<SimdBp256>(&rows);
    check_multiple_rows::<SimdBp512>(&rows);
}

/// Builds a `GroupIndex` for `input` with codec `C` and returns its RAM in bytes.
fn index_heap_bytes<C: RandomAccessCodec>(input: &[u16]) -> usize {
    let mut encoded = Vec::new();
    C::encode(input, &mut encoded);
    let mut index = GroupIndex::<C>::new();
    index.build(&encoded, input.len());
    assert_eq!(index.num_groups(), input.len().div_ceil(C::GROUP_LEN));
    index.heap_bytes()
}

#[test]
fn index_memory_is_one_usize_per_group_plus_one() {
    // 600 values: 5 groups of 128, 3 of 256, 2 of 512. This RAM is separate from the
    // encoded bytes, which don't grow.
    let input = vec![1u16; 600];
    let word = std::mem::size_of::<usize>();
    assert!(index_heap_bytes::<Bp128>(&input) >= 6 * word);
    assert!(index_heap_bytes::<SimdBp256>(&input) >= 4 * word);
    assert!(index_heap_bytes::<SimdBp512>(&input) >= 3 * word);
}

#[test]
fn rebuilding_an_index_reuses_its_memory() {
    let big: Vec<u16> = vec![1; 128 * 50];
    let small: Vec<u16> = vec![1; 128 * 3];
    let (mut big_enc, mut small_enc) = (Vec::new(), Vec::new());
    Bp128::encode(&big, &mut big_enc);
    Bp128::encode(&small, &mut small_enc);

    let mut index = GroupIndex::<Bp128>::new();
    index.build(&big_enc, big.len());
    let bytes_after_big = index.heap_bytes();
    // 51 entries (50 groups + end) of 8 bytes each, at least.
    assert!(bytes_after_big >= 51 * std::mem::size_of::<usize>());

    // A smaller row fits in the same allocation, so nothing new is allocated.
    index.build(&small_enc, small.len());
    assert_eq!(index.heap_bytes(), bytes_after_big);
    assert_eq!(index.num_groups(), 3);
}

// ---------- invalid group indices ----------

#[test]
#[should_panic(expected = "out of range")]
fn bp128_group_index_past_the_end_panics() {
    let mut encoded = Vec::new();
    Bp128::encode(&[1; 300], &mut encoded);
    let mut out = [0u16; Bp128::GROUP_LEN];
    Bp128::decode_group(&encoded, 300, 3, &mut out); // groups are 0, 1, 2
}

#[test]
#[should_panic(expected = "out of range")]
fn bp128_group_range_past_the_end_panics() {
    let mut encoded = Vec::new();
    Bp128::encode(&[1; 300], &mut encoded);
    Bp128::group_range(&encoded, 300, 3);
}

#[test]
#[should_panic(expected = "out of range")]
fn raw_group_index_past_the_end_panics() {
    let mut encoded = Vec::new();
    Raw::encode(&[1; 128], &mut encoded);
    let mut out = [0u16; Raw::GROUP_LEN];
    Raw::decode_group(&encoded, 128, 1, &mut out);
}

#[test]
#[should_panic(expected = "out of range")]
fn group_index_lookup_past_the_end_panics() {
    let mut encoded = Vec::new();
    Bp128::encode(&[1; 300], &mut encoded);
    let mut index = GroupIndex::<Bp128>::new();
    index.build(&encoded, 300);
    let mut out = [0u16; Bp128::GROUP_LEN];
    index.decode_group(&encoded, 3, &mut out);
}

#[test]
#[should_panic(expected = "out of range")]
fn empty_row_has_no_group_zero() {
    let mut out = [0u16; Bp128::GROUP_LEN];
    Bp128::decode_group(&[], 0, 0, &mut out);
}

#[test]
#[should_panic(expected = "out of range")]
fn unbuilt_index_has_no_groups() {
    let mut out = [0u16; Bp128::GROUP_LEN];
    GroupIndex::<Bp128>::new().decode_group(&[], 0, &mut out);
}

// ---------- buffer and metadata boundaries ----------

#[test]
#[should_panic(expected = "room for")]
fn output_buffer_too_small_panics() {
    let mut encoded = Vec::new();
    Bp128::encode(&[1; 300], &mut encoded);
    let mut out = [0u16; 100]; // group 0 has 128 values
    Bp128::decode_group(&encoded, 300, 0, &mut out);
}

#[test]
#[should_panic(expected = "width bytes")]
fn missing_width_bytes_panics() {
    // 300 values need 3 width bytes, but there are only 2.
    Bp128::group_range(&[0, 0], 300, 0);
}

#[test]
#[should_panic(expected = "input ended")]
fn truncated_group_data_panics() {
    let mut encoded = Vec::new();
    Bp128::encode(&[3; 256], &mut encoded); // widths [2, 2], 32 data bytes each
    encoded.pop(); // the last group is now one byte short
    Bp128::group_range(&encoded, 256, 1);
}

#[test]
#[should_panic(expected = "describe")]
fn truncated_row_fails_when_building_the_index() {
    let mut encoded = Vec::new();
    Bp128::encode(&[3; 256], &mut encoded);
    encoded.pop();
    GroupIndex::<Bp128>::new().build(&encoded, 256);
}

#[test]
#[should_panic(expected = "invalid bit width")]
fn invalid_width_byte_panics() {
    // One group whose width byte is 9, which encode never writes.
    let mut bad = vec![9u8];
    bad.extend(std::iter::repeat_n(0, 16 * 9));
    Bp128::group_range(&bad, 128, 0);
}

#[test]
#[should_panic(expected = "not a valid chunk size")]
fn decode_group_bytes_rejects_a_bad_length() {
    let mut out = [0u16; 128];
    Bp128::decode_group_bytes(&[0; 17], &mut out);
}

#[test]
#[should_panic(expected = "input ended")]
fn raw_truncated_group_panics() {
    Raw::group_range(&[0; 200], 300, 2); // group 2 needs bytes 256..300
}

#[test]
fn trailing_bytes_after_the_row_are_ignored() {
    let input: Vec<u16> = (0..200).map(|i| (i % 50) as u16).collect();
    let mut encoded = Vec::new();
    Bp128::encode(&input, &mut encoded);
    let row_len = encoded.len();
    encoded.extend_from_slice(&[0xFF; 64]); // junk that is not part of the row

    let mut index = GroupIndex::<Bp128>::new();
    index.build(&encoded, input.len());
    assert_eq!(index.encoded_len(), row_len);
    let mut out = [0u16; Bp128::GROUP_LEN];
    let len = index.decode_group(&encoded, 1, &mut out);
    assert_eq!(&out[..len], &input[128..200]);
}

// ---------- registry ----------

#[test]
fn registry_reports_which_codecs_support_random_access() {
    assert_eq!(Codec::Raw.group_len(), Some(128));
    assert_eq!(Codec::VByte.group_len(), None);
    assert_eq!(Codec::Bp128.group_len(), Some(128));
    assert_eq!(Codec::SimdBp128.group_len(), Some(128));
    assert_eq!(Codec::SimdBp256.group_len(), Some(256));
    assert_eq!(Codec::SimdBp512.group_len(), Some(512));
}

/// The one-shot `decode_group` of the codec type itself, written out by hand (like
/// `encode_directly` in roundtrip.rs) so a registry arm pointing at the wrong codec
/// would give different output.
fn decode_group_directly(
    codec: Codec,
    input: &[u8],
    n: usize,
    g: usize,
    out: &mut [u16],
) -> Option<usize> {
    match codec {
        Codec::Raw => Some(Raw::decode_group(input, n, g, out)),
        Codec::VByte => None,
        Codec::Bp128 => Some(Bp128::decode_group(input, n, g, out)),
        Codec::SimdBp128 => Some(SimdBp128::decode_group(input, n, g, out)),
        Codec::SimdBp256 => Some(SimdBp256::decode_group(input, n, g, out)),
        Codec::SimdBp512 => Some(SimdBp512::decode_group(input, n, g, out)),
    }
}

#[test]
fn registry_decode_group_matches_direct_calls() {
    let input = mixed_widths();
    let n = input.len();
    for codec in Codec::ALL {
        let mut encoded = Vec::new();
        codec.encode(&input, &mut encoded);
        // Sized from the codec's own group length (not a fixed 128), so 256- and
        // 512-value groups fit. VByte has no groups; 1 is just a placeholder size.
        let mut out = vec![0u16; codec.group_len().unwrap_or(1)];
        let mut direct = out.clone();
        match codec.group_len() {
            None => assert_eq!(codec.decode_group(&encoded, n, 0, &mut out), None),
            Some(group_len) => {
                for g in 0..n.div_ceil(group_len) {
                    let len = codec.decode_group(&encoded, n, g, &mut out).unwrap();
                    let direct_len = decode_group_directly(codec, &encoded, n, g, &mut direct);
                    assert_eq!(Some(len), direct_len, "{} group {g}", codec.name());
                    assert_eq!(
                        out,
                        direct,
                        "{} group {g}: registry vs direct",
                        codec.name()
                    );
                    let expected = &input[g * group_len..((g + 1) * group_len).min(n)];
                    assert_eq!(&out[..len], expected, "{} group {g}", codec.name());
                }
            }
        }
    }
}

// ---------- property tests ----------

/// A random row: up to 2000 values (up to 16 groups) with a random max width 0..=8,
/// so every Bp128 width shows up, not mostly width 8.
fn row() -> impl Strategy<Value = Vec<u16>> {
    (0u32..=8).prop_flat_map(|bits| {
        let max = ((1u32 << bits) - 1) as u16; // largest value with `bits` bits
        prop::collection::vec(0u16..=max, 0..2000)
    })
}

/// A random row built from up to 12 pieces of 1..=128 values, each piece with its own
/// random width, so neighboring groups have different sizes (what makes offsets
/// interesting).
fn row_with_mixed_widths() -> impl Strategy<Value = Vec<u16>> {
    let piece = (0u32..=8, prop::collection::vec(any::<u16>(), 1..=128));
    prop::collection::vec(piece, 0..12).prop_map(|pieces| {
        let mut row = Vec::new();
        for (bits, values) in pieces {
            let max = (1u32 << bits) - 1; // largest value with `bits` bits
            // `% (max + 1)` maps any u16 into 0..=max.
            row.extend(values.iter().map(|&v| (u32::from(v) % (max + 1)) as u16));
        }
        row
    })
}

proptest! {
    #[test]
    fn bp128_random_access_matches_sequential(input in row()) {
        check_all_groups::<Bp128>(&input);
    }

    #[test]
    fn bp128_random_access_matches_sequential_mixed_widths(input in row_with_mixed_widths()) {
        check_all_groups::<Bp128>(&input);
    }

    #[test]
    fn raw_random_access_matches_sequential(input in row()) {
        check_all_groups::<Raw>(&input);
    }

    #[test]
    fn simdbp_random_access_matches_sequential_mixed_widths(input in row_with_mixed_widths()) {
        check_all_groups::<SimdBp128>(&input);
        check_all_groups::<SimdBp256>(&input);
        check_all_groups::<SimdBp512>(&input);
    }

    /// Random rows appended to one buffer, read back group by group.
    #[test]
    fn multiple_random_rows(rows in prop::collection::vec(row_with_mixed_widths(), 1..5)) {
        check_multiple_rows::<Bp128>(&rows);
        check_multiple_rows::<Raw>(&rows);
        check_multiple_rows::<SimdBp128>(&rows);
        check_multiple_rows::<SimdBp512>(&rows);
    }
}
