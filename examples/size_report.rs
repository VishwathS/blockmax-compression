//! Prints bytes per value for each codec on a few synthetic datasets.
//!
//! Run with: `cargo run --release --example size_report`

use codecs::{BlockCodec, Raw, VByte};

/// Number of passages in MS MARCO, used as the doc-ID range for the gap datasets.
const MSMARCO_DOCS: u32 = 8_841_823;

/// Encodes `values` in BLOCK_SIZE blocks (the way an index would store them),
/// checks that it decodes back correctly, and returns the total encoded size in bytes.
fn encoded_size<C: BlockCodec>(values: &[u32]) -> usize {
    let mut encoded = Vec::new();
    for block in values.chunks(C::BLOCK_SIZE) {
        C::encode(block, &mut encoded);
    }

    // Sanity check: a size number is meaningless if the codec is wrong.
    let mut decoded = vec![0u32; values.len()];
    let mut pos = 0;
    for block in decoded.chunks_mut(C::BLOCK_SIZE) {
        pos += C::decode(&encoded[pos..], block);
    }
    assert_eq!(decoded, values, "round trip failed");

    encoded.len()
}

/// A tiny pseudo-random number generator (xorshift64). It's not cryptographic, but
/// it's deterministic, so the report gives the same numbers every run.
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

/// Builds a synthetic posting list: each doc in 0..MSMARCO_DOCS is included with
/// probability `per_million / 1_000_000`. Returns the d-gaps: the first doc ID itself,
/// then the difference between each doc ID and the one before it.
fn doc_id_gaps(per_million: u64, seed: u64) -> Vec<u32> {
    let mut rng = XorShift64(seed);
    let mut gaps = Vec::new();
    let mut previous = 0;
    for doc_id in 0..MSMARCO_DOCS {
        if rng.next() % 1_000_000 < per_million {
            gaps.push(doc_id - previous);
            previous = doc_id;
        }
    }
    gaps
}

fn main() {
    // (name, values, can it be stored as raw u8?)
    let datasets: Vec<(String, Vec<u32>, bool)> = vec![
        ("values 0..=255".into(), (0..=255).collect(), true),
        ("values 0..=15".into(), (0..=15).collect(), true),
        (
            "doc gaps, 10% of docs".into(),
            doc_id_gaps(100_000, 1),
            false,
        ),
        ("doc gaps, 1% of docs".into(), doc_id_gaps(10_000, 2), false),
        (
            "doc gaps, 0.1% of docs".into(),
            doc_id_gaps(1_000, 3),
            false,
        ),
    ];

    println!(
        "{:<24} {:>9} {:>9} {:>11} {:>9} {:>9}",
        "dataset", "n", "mean", "raw u8", "Raw", "VByte"
    );
    for (name, values, fits_in_u8) in &datasets {
        let n = values.len();
        // Mean value, to show how big the numbers are.
        let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let raw = encoded_size::<Raw>(values) as f64 / n as f64;
        let vbyte = encoded_size::<VByte>(values) as f64 / n as f64;
        // Parker's `BlockData::Raw` stores one byte per block max, so 1.0 B/value.
        let raw_u8 = if *fits_in_u8 {
            "1.000".to_string()
        } else {
            "n/a".to_string()
        };
        println!("{name:<24} {n:>9} {mean:>9.1} {raw_u8:>11} {raw:>9.3} {vbyte:>9.3}");
    }
    println!("(columns raw u8 / Raw / VByte are bytes per value)");
}
