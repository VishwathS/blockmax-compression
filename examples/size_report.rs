//! Prints bytes per value for each codec on a few synthetic datasets.
//!
//! Run with: `cargo run --release --example size_report`

use codecs::{BlockCodec, Bp128, MAX_VALUE, Raw, VByte};

/// Values per dataset. Large enough that per-block effects (like a partial last
/// block) don't affect the averages.
const N: usize = 1_000_000;

/// Encodes `values` in BLOCK_SIZE blocks (the way an index would store them),
/// checks that it decodes back correctly, and returns the total encoded size in bytes.
fn encoded_size<C: BlockCodec>(values: &[u16]) -> usize {
    let mut encoded = Vec::new();
    for block in values.chunks(C::BLOCK_SIZE) {
        C::encode(block, &mut encoded);
    }

    // Sanity check: a size number is meaningless if the codec is wrong.
    let mut decoded = vec![0u16; values.len()];
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

    /// A random value in `lo..=hi`. (`%` is slightly biased unless the range size is
    /// a power of two, but the bias is far too small to matter for a size report.)
    fn in_range(&mut self, lo: u16, hi: u16) -> u16 {
        let span = u64::from(hi - lo) + 1;
        lo + (self.next() % span) as u16
    }
}

/// `N` values drawn uniformly from `0..=max`.
fn uniform(max: u16, seed: u64) -> Vec<u16> {
    let mut rng = XorShift64(seed);
    (0..N).map(|_| rng.in_range(0, max)).collect()
}

/// Imitates one term's block maxes: most blocks have no posting for the term (max 0),
/// most of the rest have a small max, and a few have a large one.
/// - 90%: 0
/// - 9%: 1..=20
/// - 1%: outliers, 21..=255
fn block_max_like(seed: u64) -> Vec<u16> {
    let mut rng = XorShift64(seed);
    (0..N)
        .map(|_| match rng.next() % 100 {
            0..90 => 0,
            90..99 => rng.in_range(1, 20),
            _ => rng.in_range(21, MAX_VALUE),
        })
        .collect()
}

fn main() {
    let datasets: Vec<(&str, Vec<u16>)> = vec![
        ("uniform 0..=255", uniform(MAX_VALUE, 1)),
        ("uniform 0..=15 (4-bit)", uniform(15, 2)),
        ("all zeros", vec![0; N]),
        ("block-max-like", block_max_like(3)),
    ];

    println!(
        "{:<24} {:>9} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "dataset", "n", "mean", "zeros", "Raw", "VByte", "Bp128"
    );
    for (name, values) in &datasets {
        let n = values.len();
        // Mean value and share of zeros, to show what the data looks like.
        let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
        let zeros = values.iter().filter(|&&v| v == 0).count() as f64 / n as f64;
        let raw = encoded_size::<Raw>(values) as f64 / n as f64;
        let vbyte = encoded_size::<VByte>(values) as f64 / n as f64;
        let bp128 = encoded_size::<Bp128>(values) as f64 / n as f64;
        println!(
            "{name:<24} {n:>9} {mean:>7.2} {:>6.1}% {raw:>7.3} {vbyte:>7.3} {bp128:>7.3}",
            zeros * 100.0
        );
    }
    println!("(Raw / VByte / Bp128 columns are bytes per value)");
}
