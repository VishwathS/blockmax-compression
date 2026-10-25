//! Rows at the SIMDBP metablock boundary (GL² − 1, GL², GL² + 1 values, where GL is the
//! group length), for every codec, plus an opt-in stress test with a row of MS MARCO
//! size.
//!
//! What these prove: every codec round-trips these rows, the encoded size matches the
//! format description (computed independently below), every group decodes through a
//! `GroupIndex`, and the scalar and SIMD kernels give the same SIMDBP bytes. They are
//! our-encoder-vs-our-decoder checks plus an independent size formula; byte-for-byte
//! agreement with Parker's code is checked by the golden tests in `tests/simdbp.rs`.
//!
//! The stress test is `#[ignore]`d, so plain `cargo test` skips it. Run it with:
//!     cargo test --release --test boundaries -- --ignored

use codecs::simdbp::{Kernel, SimdBp};
use codecs::{Codec, GroupIndex, RandomAccessCodec};

/// xorshift64, for deterministic test values.
struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// `n` values where every 128-value stretch has its own random width 0..=8, so groups
/// of 128, 256, and 512 all see varied widths (including all-zero groups).
fn row(n: usize, seed: u64) -> Vec<u16> {
    let mut rng = XorShift64(seed);
    let mut mask = 0u16;
    (0..n)
        .map(|i| {
            let r = rng.next();
            if i % 128 == 0 {
                mask = ((1u32 << (r % 9)) - 1) as u16; // new width for the next 128
            }
            (r >> 16) as u16 & mask
        })
        .collect()
}

/// Bit widths of each group of `group_len` values, computed here (not by a codec).
fn widths(values: &[u16], group_len: usize) -> Vec<usize> {
    values
        .chunks(group_len)
        .map(|group| (16 - group.iter().copied().max().unwrap_or(0).leading_zeros()) as usize)
        .collect()
}

/// The encoded size each format description predicts.
fn expected_size(codec: Codec, values: &[u16]) -> usize {
    let simdbp = |stride: usize| {
        let w = widths(values, 16 * stride);
        w.len().div_ceil(16 * stride) * 8 * stride + 2 * stride * w.iter().sum::<usize>()
    };
    match codec {
        Codec::Raw => values.len(),
        Codec::VByte => values.iter().map(|&v| if v < 128 { 1 } else { 2 }).sum(),
        Codec::Bp128 => {
            let w = widths(values, 128);
            w.len() + 16 * w.iter().sum::<usize>()
        }
        Codec::SimdBp128 => simdbp(8),
        Codec::SimdBp256 => simdbp(16),
        Codec::SimdBp512 => simdbp(32),
    }
}

/// Every codec: size, round trip, and (where supported) a few groups through the
/// registry's one-shot `decode_group`.
fn check_every_codec(input: &[u16]) {
    let n = input.len();
    for codec in Codec::ALL {
        let name = codec.name();
        let mut bytes = Vec::new();
        codec.encode(input, &mut bytes);
        assert_eq!(
            bytes.len(),
            expected_size(codec, input),
            "{name}, n={n}: size"
        );
        let mut decoded = vec![0u16; n];
        assert_eq!(
            codec.decode(&bytes, &mut decoded),
            bytes.len(),
            "{name}, n={n}"
        );
        assert!(decoded == input, "{name}, n={n}: round trip");
        if let Some(gl) = codec.group_len() {
            let mut out = vec![0u16; gl];
            let last = n.div_ceil(gl) - 1;
            for g in [0, last / 2, last.saturating_sub(1), last] {
                let len = codec.decode_group(&bytes, n, g, &mut out).unwrap();
                assert_eq!(
                    &out[..len],
                    &input[g * gl..(g * gl + len)],
                    "{name} group {g}"
                );
            }
        }
    }
}

/// Every group of a typed random-access codec through a `GroupIndex`.
fn check_every_group<C: RandomAccessCodec>(input: &[u16]) {
    let mut bytes = Vec::new();
    C::encode(input, &mut bytes);
    let mut index = GroupIndex::<C>::new();
    index.build(&bytes, input.len());
    assert_eq!(index.encoded_len(), bytes.len());
    let mut out = vec![0u16; C::GROUP_LEN];
    for g in 0..index.num_groups() {
        let len = index.decode_group(&bytes, g, &mut out);
        let start = g * C::GROUP_LEN;
        assert_eq!(&out[..len], &input[start..start + len], "group {g}");
    }
}

/// The scalar kernel and the SIMD kernel (if this CPU has it) write the same SIMDBP
/// bytes and decode them back.
fn check_kernels_agree<const S: usize>(input: &[u16]) {
    let mut scalar = Vec::new();
    SimdBp::<S>::encode_with(Kernel::Scalar, input, &mut scalar);
    let simd = Kernel::simd_for_stride(S).unwrap();
    if !simd.is_available() {
        println!(
            "SKIPPED: {simd:?} not supported here; scalar only (n={})",
            input.len()
        );
        return;
    }
    let mut fast = Vec::new();
    SimdBp::<S>::encode_with(simd, input, &mut fast);
    assert!(
        fast == scalar,
        "{simd:?} vs scalar bytes, n={}",
        input.len()
    );
    let mut decoded = vec![0u16; input.len()];
    SimdBp::<S>::decode_with(simd, &scalar, &mut decoded);
    assert!(decoded == input);
}

/// GL² − 1 (last metablock one value short), GL² (exactly one full metablock), and
/// GL² + 1 (a second metablock holding a 1-value group), for stride `S`.
fn check_metablock_boundary<const S: usize>(seed: u64) {
    let gl = 16 * S;
    for n in [gl * gl - 1, gl * gl, gl * gl + 1] {
        let input = row(n, seed + n as u64);
        assert_eq!(
            SimdBp::<S>::num_metablocks(n),
            if n > gl * gl { 2 } else { 1 }
        );
        check_every_codec(&input);
        check_every_group::<SimdBp<S>>(&input);
        check_kernels_agree::<S>(&input);
    }
}

#[test]
fn simdbp128_metablock_boundary() {
    check_metablock_boundary::<8>(1);
}

#[test]
fn simdbp256_metablock_boundary() {
    check_metablock_boundary::<16>(2);
}

#[test]
fn simdbp512_metablock_boundary() {
    check_metablock_boundary::<32>(3);
}

#[test]
fn group_boundaries_for_every_codec() {
    // One value less, exactly, and one more than 1 and 2 groups of each size.
    for gl in [128usize, 256, 512] {
        for n in [gl - 1, gl, gl + 1, 2 * gl - 1, 2 * gl, 2 * gl + 1] {
            check_every_codec(&row(n, n as u64));
        }
    }
}

/// About the size of one MS MARCO row: 8,841,823 passages, one value each (a level
/// with one document per block). Opt-in, see the module docs.
#[test]
#[ignore = "stress test, ~8.8M values; run with --release -- --ignored"]
fn stress_ms_marco_sized_row() {
    const N: usize = 8_841_823;
    let input = row(N, 42);
    check_every_codec(&input);
    check_every_group::<codecs::Raw>(&input);
    check_every_group::<codecs::Bp128>(&input);
    check_every_group::<codecs::SimdBp128>(&input);
    check_every_group::<codecs::SimdBp256>(&input);
    check_every_group::<codecs::SimdBp512>(&input);
    check_kernels_agree::<8>(&input);
    check_kernels_agree::<16>(&input);
    check_kernels_agree::<32>(&input);
}
