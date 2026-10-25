//! SIMDBP tests. The most important ones compare our bytes with **golden vectors
//! produced by Parker's own code** (`tests/data/simdbp/*.txt`, written by
//! `tools/simdbp_oracle`), so "compatible" means "byte-identical to his `compress`",
//! not just "our encoder and decoder agree with each other".
//!
//! These tests need nothing but this crate: the golden files are committed, and the
//! scalar kernel runs on any CPU. On a CPU with SSE3 / AVX2 / AVX-512 the SIMD kernels
//! are tested too (set `CODECS_REQUIRE_SIMD=1` to fail if any of them is missing).

use codecs::simdbp::{Kernel, SimdBp};
use codecs::{
    BlockCodec, GroupIndex, MAX_VALUE, RandomAccessCodec, SimdBp128, SimdBp256, SimdBp512,
};
use proptest::prelude::*;

// ---------- input generators (MUST match tools/simdbp_oracle/src/main.rs) ----------

/// xorshift64, same as in the oracle.
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

/// Copy of the oracle's `generate`. If this drifts from the oracle, the `input_fnv`
/// check in `check_golden` fails.
fn generate(spec: &str, n: usize, gl: usize) -> Vec<u16> {
    let parts: Vec<&str> = spec.split(':').collect();
    let num = |i: usize| -> u64 { parts[i].parse().expect("bad number in spec") };
    match parts[0] {
        "const" => vec![num(1) as u16; n],
        "width" => {
            let w = num(1) as u32;
            let mask = ((1u32 << w) - 1) as u16;
            let mut rng = XorShift64(1000 + u64::from(w));
            (0..n)
                .map(|i| {
                    let r = rng.next() as u16 & mask;
                    if i == 0 { mask } else { r }
                })
                .collect()
        }
        "random" => {
            let (max, seed) = (num(1), num(2));
            let mut rng = XorShift64(seed);
            (0..n).map(|_| (rng.next() % (max + 1)) as u16).collect()
        }
        "groupwidths" => {
            let mut rng = XorShift64(num(1));
            (0..n)
                .map(|i| {
                    let r = rng.next();
                    let w = (i / gl) % 9;
                    let mask = ((1u32 << w) - 1) as u16;
                    if i % gl == 0 { mask } else { r as u16 & mask }
                })
                .collect()
        }
        "sparse" => {
            let mut rng = XorShift64(num(1));
            let n_groups = n.div_ceil(gl);
            let nonzero = [0, 1, gl - 1, gl, gl + 1, n_groups / 2, n_groups - 1];
            (0..n)
                .map(|i| {
                    let r = rng.next();
                    let g = i / gl;
                    let mask = ((1u32 << (g % 8 + 1)) - 1) as u16;
                    if !nonzero.contains(&g) {
                        0
                    } else if i % gl == 0 {
                        mask
                    } else {
                        r as u16 & mask
                    }
                })
                .collect()
        }
        "headers" => {
            let mut rng = XorShift64(num(1));
            (0..n)
                .map(|i| {
                    let r = rng.next();
                    let g = i / gl;
                    let w = (g + 3 * (g / gl)) % 9;
                    let mask = ((1u32 << w) - 1) as u16;
                    if i % gl == 0 { mask } else { r as u16 & mask }
                })
                .collect()
        }
        "groupmax" => {
            let w = num(1) as u32;
            let mask = ((1u32 << w) - 1) as u16;
            let mut rng = XorShift64(num(2));
            (0..n)
                .map(|i| {
                    let r = rng.next() as u16 & mask;
                    if i % gl == 0 { mask } else { r }
                })
                .collect()
        }
        "randgroups" => {
            let seed = num(1);
            let mut rng = XorShift64(seed);
            (0..n)
                .map(|i| {
                    let r = rng.next();
                    let g = (i / gl) as u64;
                    let w = ((g ^ seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) % 9;
                    let mask = ((1u32 << w) - 1) as u16;
                    if i % gl == 0 { mask } else { r as u16 & mask }
                })
                .collect()
        }
        "blockmax" => {
            let mut rng = XorShift64(num(1));
            (0..n)
                .map(|_| {
                    let r = rng.next();
                    let high = r >> 32;
                    match r % 100 {
                        0..=89 => 0,
                        90..=98 => (1 + high % 20) as u16,
                        _ => (21 + high % 235) as u16,
                    }
                })
                .collect()
        }
        other => panic!("unknown generator {other}"),
    }
}

/// FNV-1a 64-bit, same as in the oracle.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn le_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn from_hex(hex: &str) -> Vec<u8> {
    assert!(hex.len().is_multiple_of(2), "odd-length hex");
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("bad hex"))
        .collect()
}

fn parse_list(s: &str) -> Vec<usize> {
    s.split(',').map(|x| x.parse().unwrap()).collect()
}

// ---------- golden vectors from Parker's code ----------

const GOLDEN_128: &str = include_str!("data/simdbp/simdbp128.txt");
const GOLDEN_256: &str = include_str!("data/simdbp/simdbp256.txt");
const GOLDEN_512: &str = include_str!("data/simdbp/simdbp512.txt");

/// The kernels to test for `stride` lanes: always the scalar one, plus the SIMD one if
/// this CPU can run it. If it can't, a SKIPPED line is printed (see it with
/// `-- --nocapture`), or the test fails when `CODECS_REQUIRE_SIMD=1`. Which kernel
/// actually runs is proven by the unit tests in `src/simdbp/kernel_selection_tests.rs`.
fn kernels_for(stride: usize) -> Vec<Kernel> {
    let mut kernels = vec![Kernel::Scalar];
    let simd = Kernel::simd_for_stride(stride).unwrap();
    if simd.is_available() {
        kernels.push(simd);
    } else {
        assert!(
            std::env::var("CODECS_REQUIRE_SIMD").as_deref() != Ok("1"),
            "{simd:?} is required (CODECS_REQUIRE_SIMD=1) but this CPU can't run it"
        );
        println!("SKIPPED: {simd:?} (stride {stride}) not supported here; scalar only");
    }
    kernels
}

/// Our encoding of `input` at the given group size (128, 256, or 512).
fn encode_size(group_size: usize, input: &[u16]) -> Vec<u8> {
    let mut out = Vec::new();
    match group_size {
        128 => SimdBp128::encode(input, &mut out),
        256 => SimdBp256::encode(input, &mut out),
        512 => SimdBp512::encode(input, &mut out),
        _ => unreachable!(),
    }
    out
}

/// One golden case, parsed from a `case key=value ...` line.
struct GoldenCase {
    name: String,
    n: usize,
    spec: String,
    input_fnv: u64,
    words: usize,
    widths: Vec<usize>,
    offsets: Vec<usize>,
    /// (other group size, its transcode() length in u16, FNV of its bytes)
    transcoded: Vec<(usize, usize, u64)>,
    /// FNV-1a of his `compress` output bytes (every case).
    bytes_fnv: u64,
    /// His `compress` output bytes, for cases small enough to store in full (<= 32 KiB).
    /// Large cases have only `words` and `bytes_fnv`.
    bytes: Option<Vec<u8>>,
}

/// Parses `0x...` as a u64.
fn hex_u64(s: &str) -> u64 {
    u64::from_str_radix(s.trim_start_matches("0x"), 16).expect("bad hex number")
}

fn parse_golden(text: &str, expected_block_size: usize) -> Vec<GoldenCase> {
    // The header must say which of Parker's builds made this file.
    assert!(
        text.contains(&format!("block_size={expected_block_size} ")),
        "golden file is not from a BLOCK_SIZE={expected_block_size} build"
    );
    assert!(text.contains("parker_commit=fa0ede5241ea913ec9e293bdbef0d0b1012e7f07"));

    let mut cases = Vec::new();
    for line in text.lines().filter(|l| l.starts_with("case ")) {
        let optional_field = |key: &str| -> Option<&str> {
            line.split(' ')
                .find_map(|kv| kv.strip_prefix(key).and_then(|rest| rest.strip_prefix('=')))
        };
        let field = |key: &str| -> &str {
            optional_field(key).unwrap_or_else(|| panic!("missing {key} in {line:.80}"))
        };
        // Parker's own decoder and transcoder must have agreed with his encoder.
        assert_eq!(field("parker_roundtrip"), "ok", "{}", field("name"));
        assert_eq!(field("tc_back"), "ok", "{}", field("name"));
        let mut transcoded = Vec::new();
        for other in [128, 256, 512] {
            if other != expected_block_size {
                let (len, fnv) = field(&format!("tc{other}")).split_once(':').unwrap();
                transcoded.push((other, len.parse().unwrap(), hex_u64(fnv)));
            }
        }
        cases.push(GoldenCase {
            name: field("name").to_string(),
            n: field("n").parse().unwrap(),
            spec: field("gen").to_string(),
            input_fnv: hex_u64(field("input_fnv")),
            words: field("words").parse().unwrap(),
            widths: parse_list(field("widths")),
            offsets: parse_list(field("offsets")),
            transcoded,
            bytes_fnv: hex_u64(field("bytes_fnv")),
            bytes: optional_field("bytes").map(from_hex),
        });
    }
    assert_eq!(cases.len(), 25, "expected 25 golden cases");
    // The small cases (every case of the original 21, and more) must have exact bytes.
    let exact = cases.iter().filter(|c| c.bytes.is_some()).count();
    assert!(exact >= 21, "only {exact} cases have exact bytes");
    cases
}

/// Checks every golden case of one group size against our implementation.
fn check_golden<const S: usize>(text: &str) {
    let gl = 16 * S;
    for case in parse_golden(text, gl) {
        let name = &case.name;
        let input = generate(&case.spec, case.n, gl);
        assert_eq!(input.len(), case.n);
        assert_eq!(
            fnv1a(&le_bytes(&input)),
            case.input_fnv,
            "{name}: our generator differs from the oracle's"
        );
        if let Some(his) = &case.bytes {
            assert_eq!(his.len(), 2 * case.words, "{name}: golden byte count");
            assert_eq!(fnv1a(his), case.bytes_fnv, "{name}: golden bytes vs hash");
        }

        // `bytes` = Parker's exact bytes when stored. For a large case only his length
        // and hash are stored; once our bytes match both, they are used in his place.
        // (A hash match shows agreement on this input, with a negligible chance of a
        // coincidence; it is not a byte-by-byte comparison.)
        let mut bytes = case.bytes.clone();
        for kernel in kernels_for(S) {
            // 1. Encoding gives exactly Parker's bytes (or his length and hash).
            let mut ours = Vec::new();
            SimdBp::<S>::encode_with(kernel, &input, &mut ours);
            match &case.bytes {
                Some(his) => assert!(
                    &ours == his,
                    "{name} ({kernel:?}): our bytes differ from Parker's (ours {} bytes, his {}; first difference at byte {:?})",
                    ours.len(),
                    his.len(),
                    ours.iter().zip(his).position(|(a, b)| a != b)
                ),
                None => {
                    assert_eq!(ours.len(), 2 * case.words, "{name} ({kernel:?}): length");
                    assert_eq!(
                        fnv1a(&ours),
                        case.bytes_fnv,
                        "{name} ({kernel:?}): hash of our bytes differs from Parker's"
                    );
                    bytes.get_or_insert(ours);
                }
            }
        }
        let bytes = bytes.expect("set above");

        for kernel in kernels_for(S) {
            // 2. Decoding Parker's bytes gives the input and consumes all of them.
            let mut decoded = vec![0u16; case.n];
            let used = SimdBp::<S>::decode_with(kernel, &bytes, &mut decoded);
            assert_eq!(decoded, input, "{name} ({kernel:?}): decoding his bytes");
            assert_eq!(used, bytes.len(), "{name} ({kernel:?}): bytes consumed");
        }

        // 3. Our view of the headers matches his Decompressor after `init`.
        let n_groups = SimdBp::<S>::num_groups(case.n);
        assert_eq!(case.widths.len(), n_groups, "{name}");
        let mut index = GroupIndex::<SimdBp<S>>::new();
        index.build(&bytes, case.n);
        for g in 0..n_groups {
            assert_eq!(
                SimdBp::<S>::group_width(&bytes, case.n, g),
                case.widths[g],
                "{name}: width of group {g}"
            );
            // His offsets count units of STRIDE u16 words = 2 × STRIDE bytes.
            let start = case.offsets[g] * 2 * S;
            assert_eq!(
                index.group_range(g).start,
                start,
                "{name}: start of group {g}"
            );
        }
        assert_eq!(index.encoded_len(), bytes.len(), "{name}");

        // 4. Random access into his bytes: every group, indexed, and one-shot for up to
        //    600 groups (one-shot is O(g) per group, so every group of the largest
        //    case would be slow in debug builds; the index covers all of them).
        let mut out = vec![0u16; gl];
        for g in 0..n_groups {
            let expected = &input[g * gl..((g + 1) * gl).min(case.n)];
            if g < 300 || g + 300 >= n_groups {
                let len = SimdBp::<S>::decode_group(&bytes, case.n, g, &mut out);
                assert_eq!(&out[..len], expected, "{name}: group {g} (one-shot)");
            }
            let len = index.decode_group(&bytes, g, &mut out);
            assert_eq!(&out[..len], expected, "{name}: group {g} (index)");
        }

        // 5. Parker's scalar `transcode` to the other group sizes gives the same bytes
        //    as our encoder for those sizes (a second, independent piece of his code).
        for &(other, len, fnv) in &case.transcoded {
            let ours = encode_size(other, &input);
            assert_eq!(
                ours.len(),
                2 * len,
                "{name}: size {other} via transcode, length"
            );
            assert_eq!(
                fnv1a(&ours),
                fnv,
                "{name}: size {other} via transcode, bytes"
            );
        }
    }
}

#[test]
fn simdbp128_matches_parkers_sse_build() {
    check_golden::<8>(GOLDEN_128);
}

#[test]
fn simdbp256_matches_parkers_avx2_build() {
    check_golden::<16>(GOLDEN_256);
}

#[test]
fn simdbp512_matches_parkers_avx512_build() {
    check_golden::<32>(GOLDEN_512);
}

/// The golden cases must exercise every header field of a metablock, down to the bit:
/// - each of the GL slots (so every header word, every 4-bit position in a word, and
///   every lane) holds a nonzero width in some case;
/// - each of the 4 bits of every slot's field is 1 in some case and 0 in another
///   (width 8 = 0b1000 and width 7 = 0b0111 in every slot do this);
/// - one case has width 8 in every slot of at least two whole metablocks, so bit 3 is
///   set in every field of more than one metablock header, across the boundary;
/// - nonzero fields appear in at least three metablocks.
///
/// This is checked from Parker's own widths (`Decompressor.offsets`), so it measures
/// what his code was compared on, not what our code reads.
fn check_header_coverage<const S: usize>(text: &str) {
    let gl = 16 * S;
    let mut slot_seen = vec![false; gl];
    // bit_set[slot] / bit_clear[slot]: the bits that were 1 / 0 in that slot's field in
    // some group (over all cases and metablocks).
    let mut bit_set = vec![0usize; gl];
    let mut bit_clear = vec![0usize; gl];
    let mut metablocks_seen = std::collections::BTreeSet::new();
    let mut most_full_width8_metablocks = 0;
    for case in parse_golden(text, gl) {
        for (g, &width) in case.widths.iter().enumerate() {
            bit_set[g % gl] |= width;
            bit_clear[g % gl] |= !width & 0xF;
            if width > 0 {
                slot_seen[g % gl] = true;
                metablocks_seen.insert(g / gl);
            }
        }
        // Metablocks of this case in which all GL slots hold width 8.
        let full_width8 = case
            .widths
            .chunks(gl)
            .filter(|mb| mb.len() == gl && mb.iter().all(|&w| w == 8))
            .count();
        most_full_width8_metablocks = most_full_width8_metablocks.max(full_width8);
    }
    let missing: Vec<usize> = (0..gl).filter(|&slot| !slot_seen[slot]).collect();
    assert!(
        missing.is_empty(),
        "GL={gl}: header slots never nonzero: {missing:?}"
    );
    // Slot s = lane s % S, field j = s / S, header word j / 4, bit position 4 × (j % 4).
    // All slots seen implies all of these; spell it out for the report.
    for word in 0..4 {
        for position in 0..4 {
            for lane in 0..S {
                assert!(slot_seen[(4 * word + position) * S + lane]);
            }
        }
    }
    let not_all_bits: Vec<usize> = (0..gl)
        .filter(|&slot| bit_set[slot] != 0xF || bit_clear[slot] != 0xF)
        .collect();
    assert!(
        not_all_bits.is_empty(),
        "GL={gl}: slots where some header bit is never 1 or never 0: {not_all_bits:?}"
    );
    assert!(
        most_full_width8_metablocks >= 2,
        "GL={gl}: no case has width 8 in every slot of two metablocks"
    );
    assert!(
        metablocks_seen.len() >= 3,
        "GL={gl}: nonzero widths only in metablocks {metablocks_seen:?}"
    );
}

#[test]
fn golden_cases_cover_every_header_bit() {
    check_header_coverage::<8>(GOLDEN_128);
    check_header_coverage::<16>(GOLDEN_256);
    check_header_coverage::<32>(GOLDEN_512);
}

// ---------- hand-computed bytes (independent of Parker's code and of ours) ----------

#[test]
fn simdbp128_single_one_by_hand() {
    // n = 1, value 1: one group (width 1), one metablock header of 8 × 8 = 64 bytes.
    // Header: group 0 is lane 0, entry 0 → the low 4 bits of header word 0 = 1.
    // Data: width 1 → 8 words; value 0 is lane 0, bit 0 → word 0 = 1.
    let mut expected = vec![0u8; 64 + 16];
    expected[0] = 1; // header word 0, little-endian low byte
    expected[64] = 1; // data word 0
    let mut out = Vec::new();
    SimdBp128::encode(&[1], &mut out);
    assert_eq!(out, expected);
}

#[test]
fn simdbp128_lane_layout_by_hand() {
    // 128 values, all 0 except value 9 = 3 → width 2 → 2 × 8 = 16 data words.
    // Value 9 = lane 9 % 8 = 1, entry 9 / 8 = 1 → bits 2..4 of lane 1's stream →
    // lane 1's word 0, which is output word 0 × 8 + 1 = 1, holds 3 << 2 = 12.
    let mut input = vec![0u16; 128];
    input[9] = 3;
    let mut expected = vec![0u8; 64 + 32];
    expected[0] = 2; // header: group 0 has width 2
    expected[64 + 2] = 12; // data word 1 (bytes 2..4 of the data), low byte
    let mut out = Vec::new();
    SimdBp128::encode(&input, &mut out);
    assert_eq!(out, expected);
}

#[test]
fn simdbp128_header_field_position_by_hand() {
    // 18 groups; only group 17 is nonzero (max 5 → width 3).
    // Group 17 is header entry 17 = lane 17 % 8 = 1, entry j = 17 / 8 = 2 →
    // lane 1's word j / 4 = 0 (header word 1) at bit 4 × (j % 4) = 8 → word = 3 << 8.
    let mut input = vec![0u16; 18 * 128];
    input[17 * 128] = 5;
    let mut out = Vec::new();
    SimdBp128::encode(&input, &mut out);
    assert_eq!(out.len(), 64 + 3 * 16, "one header + one width-3 group");
    assert_eq!(&out[2..4], &[0x00, 0x03], "header word 1 = 0x0300");
    assert!(out[..64].iter().enumerate().all(|(i, &b)| b == 0 || i == 3));
    assert_eq!(SimdBp128::group_width(&out, input.len(), 17), 3);
    // Value 0 of group 17 (= 5) is lane 0, bits 0..3 → data word 0 = 5.
    assert_eq!(&out[64..66], &[5, 0]);
}

#[test]
fn simdbp256_header_word_1_by_hand() {
    // GL = 256, STRIDE = 16, header = 4 × 16 words = 128 bytes. Only group 93 is
    // nonzero: its value 0 is 5 (max 5 → width 3).
    // Header: slot 93 = lane 93 % 16 = 13, field j = 93 / 16 = 5 → header word
    // j / 4 = 1 of lane 13 = word 1 × 16 + 13 = 29 (bytes 58..60), at bit 4 × (5 % 4) = 4
    // → word 29 = 3 << 4 = 0x0030.
    // Data: groups 0..93 have width 0 (no bytes), so group 93 starts right after the
    // header, at byte 128. Value 0 = lane 0, bits 0..3 → data word 0 = 5.
    // Size: 128 + 2 × 16 × 3 = 224 bytes.
    let mut input = vec![0u16; 94 * 256];
    input[93 * 256] = 5;
    let mut expected = vec![0u8; 224];
    expected[58] = 0x30;
    expected[128] = 5;
    let mut out = Vec::new();
    SimdBp256::encode(&input, &mut out);
    assert_eq!(out, expected);
    assert_eq!(SimdBp256::group_width(&out, input.len(), 93), 3);
}

#[test]
fn simdbp512_high_lane_and_header_word_1_by_hand() {
    // GL = 512, STRIDE = 32, header = 4 × 32 words = 256 bytes. Only group 219 is
    // nonzero: its value 59 is 100 (0b110_0100, width 7).
    // Header: slot 219 = lane 219 % 32 = 27, field j = 219 / 32 = 6 → header word
    // j / 4 = 1 of lane 27 = word 1 × 32 + 27 = 59 (bytes 118..120), at bit
    // 4 × (6 % 4) = 8 → word 59 = 7 << 8 = 0x0700 → byte 118 = 0x00, byte 119 = 0x07.
    // Data starts at byte 256 (all earlier groups are width 0). Value 59 = lane
    // 59 % 32 = 27, entry 59 / 32 = 1 → bits 7..14 of lane 27's stream → lane 27's
    // word 0 = data word 0 × 32 + 27 = 27 (bytes 256 + 54 = 310..312), holding
    // 100 << 7 = 12800 = 0x3200 → byte 310 = 0x00, byte 311 = 0x32.
    // Size: 256 + 2 × 32 × 7 = 704 bytes.
    let mut input = vec![0u16; 220 * 512];
    input[219 * 512 + 59] = 100;
    let mut expected = vec![0u8; 704];
    expected[119] = 0x07;
    expected[311] = 0x32;
    let mut out = Vec::new();
    SimdBp512::encode(&input, &mut out);
    assert_eq!(out, expected);
    assert_eq!(SimdBp512::group_width(&out, input.len(), 219), 7);
}

#[test]
fn simdbp128_second_metablock_header_by_hand() {
    // GL = 128, STRIDE = 8, header = 64 bytes; 174 groups → 2 metablocks → 2 headers
    // (128 bytes). Only group 173 is nonzero: its value 0 is 3 (width 2).
    // Group 173 = metablock 1, slot 45 = lane 45 % 8 = 5, field j = 45 / 8 = 5 → word
    // j / 4 = 1 of lane 5 = word 1 × 8 + 5 = 13 of header 1, at bit 4 × (5 % 4) = 4.
    // Header 1 starts at byte 64, so word 13 is bytes 64 + 26 = 90..92 → 2 << 4 = 0x20.
    // Data starts after both headers, at byte 128: value 0 → data word 0 = 3.
    // Size: 128 + 2 × 8 × 2 = 160 bytes.
    let mut input = vec![0u16; 174 * 128];
    input[173 * 128] = 3;
    let mut expected = vec![0u8; 160];
    expected[90] = 0x20;
    expected[128] = 3;
    let mut out = Vec::new();
    SimdBp128::encode(&input, &mut out);
    assert_eq!(out, expected);
    assert_eq!(SimdBp128::group_width(&out, input.len(), 173), 2);
}

#[test]
fn simdbp512_header_and_metablock_sizes() {
    // 512-value groups: a header is 4 × 32 words = 256 bytes and covers 512 groups.
    assert_eq!(SimdBp512::HEADER_BYTES, 256);
    assert_eq!(SimdBp256::HEADER_BYTES, 128);
    assert_eq!(SimdBp128::HEADER_BYTES, 64);
    assert_eq!(SimdBp512::num_metablocks(512 * 512), 1);
    assert_eq!(SimdBp512::num_metablocks(512 * 512 + 1), 2);
    assert_eq!(SimdBp512::group_data_bytes(8), 512);
    // All zeros: only headers, no data.
    let mut out = Vec::new();
    SimdBp512::encode(&vec![0; 512 * 512 + 1], &mut out);
    assert_eq!(out, vec![0u8; 2 * 256]);
}

// ---------- scalar and SIMD kernels agree ----------

/// Kernels this CPU can run, with the stride each handles. If CODECS_REQUIRE_SIMD=1,
/// all three SIMD kernels must be available (used to prove they really ran).
#[test]
fn report_which_kernels_run_here() {
    for stride in [8, 16, 32] {
        let simd = Kernel::simd_for_stride(stride).unwrap();
        println!(
            "stride {stride}: {simd:?} available = {}, best = {:?}",
            simd.is_available(),
            Kernel::best(stride)
        );
        if std::env::var("CODECS_REQUIRE_SIMD").as_deref() == Ok("1") {
            assert!(
                simd.is_available(),
                "{simd:?} is required but not available"
            );
            assert_eq!(Kernel::best(stride), simd);
        }
    }
    assert!(Kernel::Scalar.is_available());
}

/// Packs `values` with every runnable kernel for `stride` and checks they all agree,
/// then unpacks with every kernel and checks we get the values back.
fn check_kernels_agree(stride: usize, width: usize, values: &[u16]) {
    let mut reference = vec![0u16; width * stride];
    Kernel::Scalar.pack(stride, width, values, &mut reference);
    for kernel in kernels_for(stride) {
        let mut packed = vec![0xAAAAu16; width * stride]; // garbage that must be overwritten
        kernel.pack(stride, width, values, &mut packed);
        assert_eq!(
            packed, reference,
            "{kernel:?} pack, stride {stride}, width {width}"
        );
        let mut unpacked = vec![0xAAAAu16; 16 * stride];
        kernel.unpack(stride, width, &reference, &mut unpacked);
        assert_eq!(
            unpacked, values,
            "{kernel:?} unpack, stride {stride}, width {width}"
        );
    }
}

#[test]
fn kernels_agree_for_every_width_and_stride() {
    for stride in [8, 16, 32] {
        for width in 0..=8 {
            let mask = ((1u32 << width) - 1) as u16;
            for seed in 1..=20 {
                let mut rng = XorShift64(seed * 7919 + width as u64);
                let values: Vec<u16> = (0..16 * stride).map(|_| rng.next() as u16 & mask).collect();
                check_kernels_agree(stride, width, &values);
            }
            // The extremes: all zero and all at the max.
            check_kernels_agree(stride, width, &vec![0; 16 * stride]);
            check_kernels_agree(stride, width, &vec![mask; 16 * stride]);
        }
    }
}

proptest! {
    /// Random values at a random width and stride: all kernels pack identically.
    #[test]
    fn kernels_agree_on_random_groups(
        (stride, width, values) in prop::sample::select(vec![8usize, 16, 32])
            .prop_flat_map(|stride| (Just(stride), 0usize..=8))
            .prop_flat_map(|(stride, width)| {
                let mask = ((1u32 << width) - 1) as u16;
                (Just(stride), Just(width),
                 prop::collection::vec(any::<u16>().prop_map(move |v| v & mask), 16 * stride))
            })
    ) {
        check_kernels_agree(stride, width, &values);
    }

    /// Any packed bits at all (not just ones `pack` wrote): all kernels unpack them the
    /// same way. This covers the SIMD unpack kernels' masking on arbitrary input.
    #[test]
    fn kernels_agree_on_arbitrary_packed_words(
        (stride, width, words) in prop::sample::select(vec![8usize, 16, 32])
            .prop_flat_map(|stride| (Just(stride), 0usize..=8))
            .prop_flat_map(|(stride, width)| {
                (Just(stride), Just(width), prop::collection::vec(any::<u16>(), width * stride))
            })
    ) {
        let mut reference = vec![0u16; 16 * stride];
        Kernel::Scalar.unpack(stride, width, &words, &mut reference);
        for kernel in kernels_for(stride) {
            let mut out = vec![0xAAAAu16; 16 * stride];
            kernel.unpack(stride, width, &words, &mut out);
            prop_assert_eq!(&out, &reference, "{:?}", kernel);
        }
    }

    /// Whole rows: every kernel encodes to the same bytes, and decodes them back.
    #[test]
    fn every_kernel_writes_the_same_row(input in prop::collection::vec(0u16..=MAX_VALUE, 0..3000)) {
        check_row_kernels::<8>(&input);
        check_row_kernels::<16>(&input);
        check_row_kernels::<32>(&input);
    }
}

fn check_row_kernels<const S: usize>(input: &[u16]) {
    let mut reference = Vec::new();
    SimdBp::<S>::encode_with(Kernel::Scalar, input, &mut reference);
    for kernel in kernels_for(S) {
        let mut bytes = Vec::new();
        SimdBp::<S>::encode_with(kernel, input, &mut bytes);
        assert_eq!(bytes, reference, "{kernel:?}");
        let mut decoded = vec![0u16; input.len()];
        assert_eq!(
            SimdBp::<S>::decode_with(kernel, &bytes, &mut decoded),
            bytes.len()
        );
        assert_eq!(decoded, input, "{kernel:?}");
    }
}

// ---------- format details ----------

#[test]
fn empty_row_is_zero_bytes() {
    // Our extension: Parker's `compress` panics on an empty input.
    let mut out = Vec::new();
    SimdBp512::encode(&[], &mut out);
    assert!(out.is_empty());
    assert_eq!(SimdBp512::decode(&[], &mut []), 0);
    let mut index = GroupIndex::<SimdBp512>::new();
    index.build(&[], 0);
    assert_eq!(index.num_groups(), 0);
    assert_eq!(index.encoded_len(), 0);
}

#[test]
fn decoding_works_at_any_byte_alignment() {
    // Our bytes are read as little-endian pairs, never reinterpreted in place, so a
    // row starting at an odd address must decode the same.
    let input: Vec<u16> = (0..1500).map(|i| (i * 7 % 200) as u16).collect();
    let mut row = Vec::new();
    SimdBp256::encode(&input, &mut row);
    for shift in 0..4 {
        let mut buffer = vec![0xEEu8; shift]; // `shift` junk bytes before the row
        buffer.extend_from_slice(&row);
        let shifted = &buffer[shift..];
        let mut decoded = vec![0u16; input.len()];
        assert_eq!(SimdBp256::decode(shifted, &mut decoded), row.len());
        assert_eq!(decoded, input, "shift {shift}");
        let mut out = [0u16; 256];
        let len = SimdBp256::decode_group(shifted, input.len(), 5, &mut out);
        assert_eq!(&out[..len], &input[5 * 256..1500]);
    }
}

#[test]
fn partial_group_padding_is_zero_and_not_written_out() {
    // 300 values in 512-groups: one partial group. Its 212 padding values are packed
    // as zeros (same width as without them) and never copied to `out`.
    let input: Vec<u16> = (0..300).map(|i| (i % 16) as u16).collect(); // width 4
    let mut row = Vec::new();
    SimdBp512::encode(&input, &mut row);
    assert_eq!(row.len(), 256 + 4 * 64);
    let mut out = [999u16; 512];
    assert_eq!(SimdBp512::decode_group(&row, 300, 0, &mut out), 300);
    assert_eq!(&out[..300], &input[..]);
    assert!(
        out[300..].iter().all(|&v| v == 999),
        "past the group: untouched"
    );
    // Unpacking the full group shows the padding really is zero.
    let range = SimdBp512::group_range(&row, 300, 0);
    let mut full = [999u16; 512];
    SimdBp512::decode_group_bytes(&row[range], &mut full);
    assert!(full[300..].iter().all(|&v| v == 0));
}

#[test]
fn concatenated_rows_decode_in_order() {
    let rows: Vec<Vec<u16>> = vec![
        vec![5; 10],
        vec![],
        (0..2000).map(|i| (i % 256) as u16).collect(),
        vec![0; 600],
    ];
    let mut buffer = Vec::new();
    for row in &rows {
        SimdBp128::encode(row, &mut buffer);
    }
    let mut pos = 0;
    for row in &rows {
        let mut decoded = vec![0u16; row.len()];
        pos += SimdBp128::decode(&buffer[pos..], &mut decoded);
        assert_eq!(&decoded, row);
    }
    assert_eq!(pos, buffer.len());
}

// ---------- invalid input ----------

#[test]
#[should_panic(expected = "above 255")]
fn rejects_values_above_255() {
    // In the second group, so the check isn't just on the first one.
    let mut input = vec![0u16; 300];
    input[200] = 256;
    SimdBp128::encode(&input, &mut Vec::new());
}

#[test]
#[should_panic(expected = "metablock headers")]
fn truncated_header_panics() {
    SimdBp128::decode(&[0; 63], &mut [0u16; 5]); // a header is 64 bytes
}

#[test]
#[should_panic(expected = "input ended in the middle of group")]
fn truncated_group_data_panics() {
    let mut row = Vec::new();
    SimdBp128::encode(&[MAX_VALUE; 200], &mut row);
    row.pop();
    SimdBp128::decode(&row, &mut [0u16; 200]);
}

#[test]
#[should_panic(expected = "describe")]
fn truncated_row_fails_when_building_the_index() {
    let mut row = Vec::new();
    SimdBp256::encode(&[3; 600], &mut row);
    row.pop();
    GroupIndex::<SimdBp256>::new().build(&row, 600);
}

#[test]
#[should_panic(expected = "invalid bit width 9")]
fn invalid_width_in_header_panics() {
    // Header field 0 = 9: no value <= 255 needs 9 bits, so `encode` never writes it.
    let mut bad = vec![0u8; 64 + 9 * 16];
    bad[0] = 9;
    SimdBp128::decode(&bad, &mut [0u16; 128]);
}

#[test]
#[should_panic(expected = "not a valid group size")]
fn decode_group_bytes_rejects_a_bad_length() {
    SimdBp128::decode_group_bytes(&[0; 17], &mut [0u16; 128]);
}

#[test]
#[should_panic(expected = "out of range")]
fn group_index_past_the_end_panics() {
    let mut row = Vec::new();
    SimdBp512::encode(&[1; 600], &mut row);
    SimdBp512::decode_group(&row, 600, 2, &mut [0u16; 512]);
}

#[test]
#[should_panic(expected = "doesn't handle stride")]
fn simd_kernel_for_the_wrong_stride_panics() {
    // Avx2 is for stride 16 only. (Panics before touching any SIMD instruction, so it
    // is safe on every CPU.)
    Kernel::Avx2.pack(8, 1, &[0; 128], &mut [0; 8]);
}

#[test]
#[should_panic(expected = "doesn't fit in 2 bits")]
fn packing_a_value_wider_than_the_width_panics() {
    let mut values = vec![0u16; 128];
    values[3] = 4; // needs 3 bits
    Kernel::Scalar.pack(8, 2, &values, &mut [0; 16]);
}

#[test]
#[should_panic(expected = "a group has 128 values")]
fn kernel_rejects_a_wrong_group_length() {
    Kernel::Scalar.pack(8, 1, &[0; 127], &mut [0; 8]);
}

// ---------- random access ----------

proptest! {
    /// Random rows at every group size: each group read on its own (one-shot and via
    /// an index) equals the same slice of a sequential decode.
    #[test]
    fn random_access_matches_sequential(
        input in (0u32..=8).prop_flat_map(|bits| {
            let max = ((1u32 << bits) - 1) as u16;
            prop::collection::vec(0u16..=max, 0..3000)
        })
    ) {
        check_random_access::<SimdBp128>(&input);
        check_random_access::<SimdBp256>(&input);
        check_random_access::<SimdBp512>(&input);
    }
}

fn check_random_access<C: RandomAccessCodec>(input: &[u16]) {
    let n = input.len();
    let mut row = Vec::new();
    C::encode(input, &mut row);
    let mut sequential = vec![0u16; n];
    let used = C::decode(&row, &mut sequential);
    assert_eq!(sequential, input);
    let mut index = GroupIndex::<C>::new();
    index.build(&row, n);
    assert_eq!(index.encoded_len(), used);
    let mut out = vec![0u16; C::GROUP_LEN];
    // Backwards, to make sure nothing depends on reading in order.
    for g in (0..C::num_groups(n)).rev() {
        let expected = &input[g * C::GROUP_LEN..((g + 1) * C::GROUP_LEN).min(n)];
        let len = C::decode_group(&row, n, g, &mut out);
        assert_eq!(&out[..len], expected);
        let len = index.decode_group(&row, g, &mut out);
        assert_eq!(&out[..len], expected);
        assert_eq!(C::group_range(&row, n, g), index.group_range(g));
    }
}
