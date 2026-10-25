//! Independent check of Bp128's bytes: a small model of the format written here from
//! its description, NOT from the `bitpacking` crate that `Bp128` uses. If the crate's
//! layout ever changed (or `Bp128` used it wrongly), the bytes would differ.
//!
//! Run in debug AND release (`cargo test --release`): the crate uses a scalar packer
//! in debug builds and an SSE3 one in release builds, and both must match the model.
//!
//! The model (`BitPacker4x`, 128 values per chunk, width w):
//! - value i goes to lane i % 4, as that lane's entry j = i / 4 (32 entries per lane);
//! - each lane is a stream of 32 × w bits, entry j at bits j × w .. (j + 1) × w,
//!   stored as w little-endian u32 words, lowest bits first;
//! - u32 word k of lane l is stored at bytes 16 × k + 4 × l (so one "row" of 16 bytes
//!   holds word k of all 4 lanes).
//!
//! The whole row is `[one width byte per chunk][chunk 0 data][chunk 1 data]...`, with a
//! short last chunk padded with zeros to 128 values.

use codecs::{BlockCodec, Bp128};

/// The model's packed bytes for one chunk of 128 values at width `w` (16 × w bytes).
fn model_chunk(values: &[u16; 128], w: usize) -> Vec<u8> {
    let mut words = vec![0u32; 4 * w];
    if w > 0 {
        for (i, &value) in values.iter().enumerate() {
            let (lane, j) = (i % 4, i / 4);
            let bit = j * w; // where entry j starts in its lane's bit stream
            let (word, shift) = (bit / 32, bit % 32);
            let v = u32::from(value);
            // Low part; bits pushed past bit 31 are dropped here and written below.
            words[word * 4 + lane] |= v << shift;
            // High part, if the entry crosses into the lane's next u32 word.
            if shift + w > 32 {
                words[(word + 1) * 4 + lane] |= v >> (32 - shift);
            }
        }
    }
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// The model's bytes for a whole row.
fn model_row(input: &[u16]) -> Vec<u8> {
    let mut widths = Vec::new();
    let mut data = Vec::new();
    for chunk in input.chunks(128) {
        let max = chunk.iter().copied().max().unwrap_or(0);
        let w = (16 - max.leading_zeros()) as usize;
        let mut padded = [0u16; 128]; // zero padding for a short last chunk
        padded[..chunk.len()].copy_from_slice(chunk);
        widths.push(w as u8);
        data.extend(model_chunk(&padded, w));
    }
    widths.extend(data);
    widths
}

fn bp128_bytes(input: &[u16]) -> Vec<u8> {
    let mut out = Vec::new();
    Bp128::encode(input, &mut out);
    out
}

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

/// 128 values of width exactly `w` (value 0 is the max).
fn chunk_of_width(w: u32, seed: u64) -> Vec<u16> {
    let mask = ((1u32 << w) - 1) as u16;
    let mut rng = XorShift64(seed);
    (0..128)
        .map(|i| {
            if i == 0 {
                mask
            } else {
                rng.next() as u16 & mask
            }
        })
        .collect()
}

#[test]
fn model_reproduces_the_pinned_width_1_example() {
    // Same input as `bp128_known_bytes` in roundtrip.rs, so the model agrees with the
    // hand-computed bytes there.
    let mut input = vec![0u16; 128];
    input[0] = 1;
    input[5] = 1;
    let mut expected = vec![0u8; 17];
    expected[0] = 1;
    expected[1] = 0b01;
    expected[5] = 0b10;
    assert_eq!(model_row(&input), expected);
}

#[test]
fn every_width_matches_the_model() {
    for w in 0..=8 {
        for seed in 1..=10 {
            let input = chunk_of_width(w, seed * 31 + u64::from(w));
            assert_eq!(
                bp128_bytes(&input),
                model_row(&input),
                "width {w}, seed {seed}"
            );
        }
        // All values at the max for that width: every bit of the data is set.
        let all_max = vec![((1u32 << w) - 1) as u16; 128];
        assert_eq!(
            bp128_bytes(&all_max),
            model_row(&all_max),
            "width {w}, all max"
        );
    }
}

#[test]
fn partial_chunks_and_mixed_widths_match_the_model() {
    for n in [
        1, 2, 3, 4, 5, 31, 32, 33, 100, 127, 129, 255, 256, 257, 300, 1000,
    ] {
        // Each 128-value chunk gets a different width (cycling 0..=8).
        let mut rng = XorShift64(n as u64 + 7);
        let input: Vec<u16> = (0..n)
            .map(|i| {
                let w = (i / 128) % 9;
                let mask = ((1u32 << w) - 1) as u16;
                if i % 128 == 0 {
                    mask
                } else {
                    rng.next() as u16 & mask
                }
            })
            .collect();
        assert_eq!(bp128_bytes(&input), model_row(&input), "n = {n}");
    }
}

#[test]
fn every_value_0_to_255_at_every_lane_position_matches_the_model() {
    // Value v alone at position p (so it lands in lane p % 4, entry p / 4), for every
    // v and a spread of positions, including both ends of each lane.
    for v in 0..=255u16 {
        for p in [0, 1, 2, 3, 4, 63, 64, 124, 125, 126, 127] {
            let mut input = vec![0u16; 128];
            input[p] = v;
            assert_eq!(bp128_bytes(&input), model_row(&input), "value {v} at {p}");
        }
    }
}
