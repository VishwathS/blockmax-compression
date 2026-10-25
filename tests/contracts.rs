//! The agreed behavior at the edges of the contract (see `notes/design_decisions.md`,
//! B1–B3 and P1):
//!
//! - **B1** An out-of-range value (> 255) panics BEFORE anything is written: the
//!   caller's output buffer is left exactly as it was, for every codec and kernel.
//! - **P1-style reuse** A `GroupIndex` and output buffer reused for the next row of the
//!   same length never return the previous row's data.
//! - **Truncated input** to a `GroupIndex` panics instead of reading past the end.
//! - **B2 (lenient decoding, kept on purpose)** Decoders accept some byte patterns our
//!   encoders never write (VByte over-long values, nonzero padding bits, unused SIMDBP
//!   header slots), like Parker's decoder does. These tests pin that behavior.
//! - **B3 (the caller supplies `n`)** The formats don't store the row length. Decoding
//!   with the wrong `n` is NOT detected in general; these tests show what happens, so
//!   nobody relies on it being caught. Pass the exact logical row length.

use std::panic::{AssertUnwindSafe, catch_unwind};

use codecs::simdbp::{Kernel, SimdBp};
use codecs::{
    BlockCodec, Bp128, Codec, GroupIndex, RandomAccessCodec, Raw, SimdBp128, SimdBp256, SimdBp512,
    VByte,
};

/// Runs `f`, which must panic, and returns its panic message.
fn panic_message(f: impl FnOnce()) -> String {
    // `AssertUnwindSafe`: we promise not to use half-updated state after the panic
    // without checking it (checking it is exactly what these tests do).
    let payload = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        String::from("<non-string panic>")
    }
}

fn encode<C: BlockCodec>(input: &[u16]) -> Vec<u8> {
    let mut out = Vec::new();
    C::encode(input, &mut out);
    out
}

// ---------- B1 + invalid values: rejected before anything is written ----------

/// Out-of-range values to try: just above the limit, a few in between, the largest u16.
const BAD_VALUES: [u16; 5] = [256, 300, 1000, 0x8000, u16::MAX];

/// Rows (length, position of the bad value) that put the bad value at the start, in
/// the middle, at the very end of a partial last group (for 128/256/512 groups), and
/// in the second SIMDBP-128 metablock (past 128 × 128 values).
const PLACES: [(usize, usize); 6] = [
    (1, 0),
    (600, 0),
    (600, 299),
    (600, 599),
    (128 * 128 + 50, 128 * 128 + 10),
    (128 * 128 + 50, 128 * 128 + 49),
];

/// A valid row of `n` values with `bad` at `at`.
fn row_with(n: usize, at: usize, bad: u16) -> Vec<u16> {
    let mut row: Vec<u16> = (0..n).map(|i| (i % 256) as u16).collect();
    row[at] = bad;
    row
}

/// Calls `encode` on every bad row and checks the panic message and that the output
/// buffer (which already holds some bytes) is unchanged.
fn check_rejects(name: &str, encode: impl Fn(&[u16], &mut Vec<u8>)) {
    for (n, at) in PLACES {
        for bad in BAD_VALUES {
            let row = row_with(n, at, bad);
            let before = vec![0xA5u8, 0x5A, 0x3C]; // earlier rows' bytes, must survive
            let mut out = before.clone();
            let message = panic_message(|| encode(&row, &mut out));
            assert!(
                message.contains(&format!("value {bad} is above 255")),
                "{name}, n={n}, at={at}: unexpected message {message:?}"
            );
            assert_eq!(
                out, before,
                "{name}, n={n}, at={at}, value {bad}: buffer changed"
            );
        }
    }
}

#[test]
fn every_codec_rejects_bad_values_before_writing_anything() {
    check_rejects("Raw", Raw::encode);
    check_rejects("VByte", VByte::encode);
    check_rejects("Bp128", Bp128::encode);
    check_rejects("SimdBp128", SimdBp128::encode);
    check_rejects("SimdBp256", SimdBp256::encode);
    check_rejects("SimdBp512", SimdBp512::encode);
}

#[test]
fn registry_rejects_bad_values_before_writing_anything() {
    for codec in Codec::ALL {
        check_rejects(codec.name(), |input, out| codec.encode(input, out));
    }
}

/// Every kernel this CPU can run, forced, must reject the same way.
fn check_rejects_with_every_kernel<const S: usize>() {
    for kernel in Kernel::ALL {
        let runs_here = kernel == Kernel::Scalar
            || (Kernel::simd_for_stride(S) == Some(kernel) && kernel.is_available());
        if runs_here {
            check_rejects(&format!("SimdBp<{S}> {kernel:?}"), |input, out| {
                SimdBp::<S>::encode_with(kernel, input, out)
            });
        }
    }
}

#[test]
fn every_simdbp_kernel_rejects_bad_values_before_writing_anything() {
    check_rejects_with_every_kernel::<8>();
    check_rejects_with_every_kernel::<16>();
    check_rejects_with_every_kernel::<32>();
}

#[test]
fn valid_rows_still_append_after_a_rejected_row() {
    // After a rejected encode, the same buffer keeps working: the next valid row is
    // appended right after the earlier bytes and decodes normally.
    for codec in Codec::ALL {
        let mut out = Vec::new();
        codec.encode(&[1, 2, 3], &mut out);
        let first_len = out.len();
        let _ = panic_message(|| codec.encode(&[4, 256], &mut out));
        assert_eq!(out.len(), first_len, "{}", codec.name());
        codec.encode(&[7, 8], &mut out);
        let mut a = [0u16; 3];
        let used = codec.decode(&out, &mut a);
        let mut b = [0u16; 2];
        codec.decode(&out[used..], &mut b);
        assert_eq!((a, b), ([1, 2, 3], [7, 8]), "{}", codec.name());
    }
}

// ---------- P1-style reuse: no stale data from the previous row ----------

/// Parker's P1: a reused `Decompressor`, re-initialized for row B of the same length,
/// returned row A's cached group. Our `GroupIndex` keeps no decoded data, so the same
/// sequence must return row B.
fn check_no_stale_group<C: RandomAccessCodec>() {
    let n = 3 * C::GROUP_LEN; // same length for both rows, as in P1
    let row_a: Vec<u16> = (0..n).map(|i| (i % 7) as u16).collect();
    let row_b: Vec<u16> = (0..n).map(|i| (100 + i % 5) as u16).collect();
    let (bytes_a, bytes_b) = (encode::<C>(&row_a), encode::<C>(&row_b));

    let mut index = GroupIndex::<C>::new(); // one index ...
    let mut out = vec![0u16; C::GROUP_LEN]; // ... and one output buffer for both rows
    for g in [0, C::num_groups(n) - 1] {
        let group_a = &row_a[g * C::GROUP_LEN..(g + 1) * C::GROUP_LEN];
        let group_b = &row_b[g * C::GROUP_LEN..(g + 1) * C::GROUP_LEN];
        index.build(&bytes_a, n);
        let len = index.decode_group(&bytes_a, g, &mut out);
        assert_eq!(&out[..len], group_a);
        // Same group of the next row: must be row B's values, not A's.
        index.build(&bytes_b, n);
        let len = index.decode_group(&bytes_b, g, &mut out);
        assert_eq!(&out[..len], group_b, "group {g}");
        // And the one-shot path, with the buffer still holding B's group.
        let len = C::decode_group(&bytes_a, n, g, &mut out);
        assert_eq!(&out[..len], group_a, "group {g}");
    }
}

#[test]
fn reused_index_never_returns_the_previous_rows_group() {
    check_no_stale_group::<Raw>();
    check_no_stale_group::<Bp128>();
    check_no_stale_group::<SimdBp128>();
    check_no_stale_group::<SimdBp256>();
    check_no_stale_group::<SimdBp512>();
}

// ---------- truncated input to a GroupIndex ----------

fn check_truncated_index_input<C: RandomAccessCodec>() {
    let n = 2 * C::GROUP_LEN + 5;
    let input: Vec<u16> = (0..n).map(|i| (i % 200 + 1) as u16).collect();
    let row = encode::<C>(&input);
    let mut index = GroupIndex::<C>::new();
    index.build(&row, n);
    let last = index.num_groups() - 1;
    let short = &row[..row.len() - 1];
    let mut out = vec![0u16; C::GROUP_LEN];

    // Built on the full row, then given a shorter slice: panics, never reads past it.
    let message = panic_message(|| {
        index.decode_group(short, last, &mut out);
    });
    assert!(message.contains("shorter than the row"), "{message}");
    // Groups that still fit in the shorter slice decode fine.
    let len = index.decode_group(short, 0, &mut out);
    assert_eq!(&out[..len], &input[..len]);

    // Building on a truncated row panics too (the metadata describes more bytes), and
    // so does building on no bytes at all.
    let _ = panic_message(|| GroupIndex::<C>::new().build(short, n));
    let _ = panic_message(|| GroupIndex::<C>::new().build(&[], n));
}

#[test]
fn group_index_rejects_truncated_input() {
    check_truncated_index_input::<Raw>();
    check_truncated_index_input::<Bp128>();
    check_truncated_index_input::<SimdBp128>();
    check_truncated_index_input::<SimdBp256>();
    check_truncated_index_input::<SimdBp512>();
}

#[test]
fn registry_rejects_bad_group_indices() {
    for codec in Codec::ALL {
        let mut row = Vec::new();
        codec.encode(&[9u16; 300], &mut row);
        let mut out = [0u16; 512];
        match codec.group_len() {
            None => assert_eq!(codec.decode_group(&row, 300, 0, &mut out), None),
            Some(group_len) => {
                let n_groups = 300usize.div_ceil(group_len);
                let message = panic_message(|| {
                    codec.decode_group(&row, 300, n_groups, &mut out);
                });
                assert!(
                    message.contains("out of range"),
                    "{}: {message}",
                    codec.name()
                );
            }
        }
    }
}

// ---------- B2: lenient decoding (kept, pinned here) ----------

#[test]
fn vbyte_accepts_an_over_long_encoding() {
    // `encode` writes 5 as the single byte 0x85. The two bytes [0x05, 0x80] also mean
    // 5 (low 7 bits 5, then a last byte holding 0). Decoding accepts them and
    // consumes 2 bytes. Not rejected on purpose (B2): no canonical-form check.
    let mut out = [0u16; 1];
    assert_eq!(VByte::decode(&[0x05, 0x80], &mut out), 2);
    assert_eq!(out, [5]);
    assert_eq!(encode::<VByte>(&[5]), [0x85]);
}

/// SIMDBP: nonzero bits in the zero padding of the last group are ignored.
///
/// The row is GL + 10 values of 255: two groups of width 8, the second with 10 real
/// values. Its last data byte is the high byte of lane STRIDE−1's word 7 = bits
/// 120..128 of that lane's stream = entry 15 = value STRIDE−1 + 15 × STRIDE =
/// 16 × STRIDE − 1 of the group, which is padding.
fn check_simdbp_ignores_padding_bits<const S: usize>() {
    let gl = 16 * S;
    let input = vec![255u16; gl + 10];
    let mut row = encode::<SimdBp<S>>(&input);
    let last = row.len() - 1;
    assert_eq!(row[last], 0, "padding is written as zeros");
    row[last] = 0xFF; // set the padding bits

    let mut decoded = vec![0u16; input.len()];
    assert_eq!(SimdBp::<S>::decode(&row, &mut decoded), row.len());
    assert_eq!(decoded, input);
    let mut out = vec![0u16; gl];
    let len = SimdBp::<S>::decode_group(&row, input.len(), 1, &mut out);
    assert_eq!(&out[..len], &input[gl..]);
    // The padding value is visible only to someone who unpacks the whole group.
    let range = SimdBp::<S>::group_range(&row, input.len(), 1);
    SimdBp::<S>::decode_group_bytes(&row[range], &mut out);
    assert_eq!(out[gl - 1], 255, "the padding value we set");
}

/// SIMDBP: unused slots of the last header (groups past the end) are ignored, even if
/// they hold a width (like 15) that would be invalid for a real group.
fn check_simdbp_ignores_unused_header_slots<const S: usize>() {
    let input: Vec<u16> = (0..10).map(|i| i as u16).collect(); // one group, width 4
    let mut row = encode::<SimdBp<S>>(&input);
    // Header byte 1 = bits 8..16 of header word 0 = lane 0, fields 2 and 3 = slots
    // 2 × STRIDE and 3 × STRIDE: both unused. The header's last byte is unused too.
    row[1] = 0xFF;
    row[SimdBp::<S>::HEADER_BYTES - 1] = 0xFF;
    let mut decoded = vec![0u16; input.len()];
    assert_eq!(SimdBp::<S>::decode(&row, &mut decoded), row.len());
    assert_eq!(decoded, input);
    let mut index = GroupIndex::<SimdBp<S>>::new();
    index.build(&row, input.len());
    assert_eq!(index.encoded_len(), row.len());
}

#[test]
fn simdbp_ignores_padding_bits_and_unused_header_slots() {
    check_simdbp_ignores_padding_bits::<8>();
    check_simdbp_ignores_padding_bits::<16>();
    check_simdbp_ignores_padding_bits::<32>();
    check_simdbp_ignores_unused_header_slots::<8>();
    check_simdbp_ignores_unused_header_slots::<16>();
    check_simdbp_ignores_unused_header_slots::<32>();
}

#[test]
fn bp128_ignores_padding_bits() {
    // 100 values of 255: one chunk at width 8. Its last data byte (row byte 1 + 127)
    // is lane 3's u32 word 7, bits 24..32 = lane 3 entry 31 = value 127: padding.
    let input = vec![255u16; 100];
    let mut row = encode::<Bp128>(&input);
    assert_eq!(row.len(), 1 + 128);
    assert_eq!(row[128], 0);
    row[128] = 0xFF;
    let mut decoded = vec![0u16; 100];
    assert_eq!(Bp128::decode(&row, &mut decoded), row.len());
    assert_eq!(decoded, input);
}

// ---------- B3: the caller must supply the correct row length ----------

/// A 100-value SIMDBP row decoded as 200 values: the extra values read header slots
/// that are 0 for unused groups (or the zero padding), so they come back as 0, with no
/// panic and the same byte count. Nothing in the bytes says the row had 100 values.
fn check_simdbp_too_large_n<const S: usize>() {
    let input = vec![5u16; 100];
    let row = encode::<SimdBp<S>>(&input);
    let mut wrong = vec![7u16; 200];
    assert_eq!(SimdBp::<S>::decode(&row, &mut wrong), row.len());
    assert_eq!(&wrong[..100], &input[..]);
    assert!(wrong[100..].iter().all(|&v| v == 0), "extra values are 0");
}

#[test]
fn simdbp_with_a_too_large_n_silently_returns_zeros() {
    check_simdbp_too_large_n::<8>();
    check_simdbp_too_large_n::<16>();
    check_simdbp_too_large_n::<32>();
}

#[test]
fn raw_and_bp128_with_a_too_large_n_read_the_next_rows_bytes() {
    // Raw: two 10-value rows in one buffer; decoding the first as 20 values returns
    // both rows, no panic.
    let mut buffer = Vec::new();
    Raw::encode(&[1; 10], &mut buffer);
    Raw::encode(&[2; 10], &mut buffer);
    let mut wrong = [0u16; 20];
    assert_eq!(Raw::decode(&buffer, &mut wrong), 20);
    assert_eq!(&wrong[10..], &[2; 10]);

    // Bp128: row A = 100 zeros (bytes [0]), then row B = ten 5s (bytes [3, 48 data
    // bytes]). Decoding A as 200 values reads B's width byte as A's second width and
    // B's data as A's second chunk: no panic, and A's values 128.. are B's values.
    let mut buffer = Vec::new();
    Bp128::encode(&[0; 100], &mut buffer);
    Bp128::encode(&[5; 10], &mut buffer);
    let mut wrong = vec![0u16; 200];
    assert_eq!(Bp128::decode(&buffer, &mut wrong), buffer.len());
    assert_eq!(&wrong[128..138], &[5; 10]);
}
