//! Prints bytes per value for each codec on a few SYNTHETIC datasets.
//!
//! Run with: `cargo run --release --example size_report`
//!
//! The datasets are made up (uniform random values, all zeros, and an invented
//! "block-max-like" mix). They check that the codecs behave as expected; they are NOT
//! measurements of real block maxes, and the numbers are not research results.

use codecs::{Codec, MAX_VALUE};

/// Values per dataset. Large enough that per-row effects (headers, a partial last
/// group) don't affect the averages much.
const N: usize = 1_000_000;

/// Encodes all of `values` as ONE row with `codec` (one `encode` call, the way one
/// term's level would be stored), checks that it decodes back correctly, and returns
/// the encoded size in bytes.
///
/// One row, not one row per `block_size()` values: codecs with a per-row header
/// (SIMDBP's metablock headers are 64 to 256 bytes) would otherwise pay that header
/// every few hundred values, which is not how an index stores them.
fn encoded_size(codec: Codec, values: &[u16]) -> usize {
    let mut encoded = Vec::new();
    codec.encode(values, &mut encoded);

    // Sanity check: a size number is meaningless if the codec is wrong.
    let mut decoded = vec![0u16; values.len()];
    let consumed = codec.decode(&encoded, &mut decoded);
    // Every byte must be accounted for: decode must stop exactly at the end.
    assert_eq!(
        consumed,
        encoded.len(),
        "{}: decode left bytes unread",
        codec.name()
    );
    assert_eq!(decoded, values, "round trip failed");

    encoded.len()
}

/// RAM a `GroupIndex` for one row of `n` values needs, in bytes, or `None` if the codec
/// has no random access. One `usize` per group plus one (see `GroupIndex`); this lives
/// in memory while a row is read and is NOT part of the encoded size.
fn index_bytes(codec: Codec, n: usize) -> Option<usize> {
    let group_len = codec.group_len()?;
    Some((n.div_ceil(group_len) + 1) * std::mem::size_of::<usize>())
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

/// An invented imitation of one term's block maxes (the proportions are made up, not
/// measured): most blocks have no posting for the term (max 0), most of the rest have
/// a small max, and a few have a large one.
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
        ("block-max-like (invented)", block_max_like(3)),
    ];

    println!("SYNTHETIC data only: these numbers check the codecs, they are not research results.");
    println!("Each dataset is encoded as one row of {N} values.");
    println!();

    // Table 1: encoded size. Header: the fixed columns, then one column per codec.
    println!("Encoded size (bytes per value):");
    print!("{:<26} {:>9} {:>7} {:>7}", "dataset", "n", "mean", "zeros");
    for codec in Codec::ALL {
        print!(" {:>9}", codec.name());
    }
    println!();
    for (name, values) in &datasets {
        let n = values.len();
        // Mean value and share of zeros, to show what the data looks like.
        let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
        let zeros = values.iter().filter(|&&v| v == 0).count() as f64 / n as f64;
        print!("{name:<26} {n:>9} {mean:>7.2} {:>6.1}%", zeros * 100.0);
        for codec in Codec::ALL {
            let bytes_per_value = encoded_size(codec, values) as f64 / n as f64;
            print!(" {bytes_per_value:>9.3}");
        }
        println!();
    }

    // Table 2: index RAM, kept separate because it isn't stored, and it's the same for
    // every dataset of the same length (it depends only on n and the group size).
    println!();
    println!(
        "GroupIndex RAM while one row is open (bytes per value; not part of the encoded size):"
    );
    print!("{:<26} {:>9} {:>7} {:>7}", "", "n", "", "");
    for codec in Codec::ALL {
        print!(" {:>9}", codec.name());
    }
    println!();
    print!("{:<26} {N:>9} {:>7} {:>7}", "any dataset", "", "");
    for codec in Codec::ALL {
        match index_bytes(codec, N) {
            Some(bytes) => print!(" {:>9.4}", bytes as f64 / N as f64),
            None => print!(" {:>9}", "n/a"), // no random access (VByte)
        }
    }
    println!();
}

// These tests run with `cargo test` because Cargo.toml sets `test = true` for this
// example. They check the report's own `encoded_size` against every registered codec.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoded_size_works_for_every_codec() {
        // 300 values in one row.
        let zeros = vec![0u16; 300];
        let small = vec![3u16; 300]; // 2 bits each
        for codec in Codec::ALL {
            // Every codec must at least round trip; `encoded_size` asserts that.
            assert!(encoded_size(codec, &small) > 0, "{}", codec.name());
            // Raw, VByte, and Bp128 give the same numbers as before the switch to one
            // row per dataset (they have no per-row header). SIMDBP stores only its
            // metablock header for all-zero data: 8 × STRIDE bytes.
            let expected_zeros = match codec {
                Codec::Raw | Codec::VByte => 300, // 1 byte per value
                Codec::Bp128 => 3,                // 1 width byte per 128-value chunk
                Codec::SimdBp128 => 64,
                Codec::SimdBp256 => 128,
                Codec::SimdBp512 => 256,
            };
            assert_eq!(
                encoded_size(codec, &zeros),
                expected_zeros,
                "{}",
                codec.name()
            );
        }
        // Width-2 data: Bp128 = 3 chunks × (1 width byte + 16 × 2 data bytes).
        assert_eq!(encoded_size(Codec::Bp128, &small), 3 * (1 + 32));
        // SIMDBP = header + groups × (2 × STRIDE × width) bytes.
        assert_eq!(encoded_size(Codec::SimdBp128, &small), 64 + 3 * (2 * 8 * 2));
        assert_eq!(
            encoded_size(Codec::SimdBp256, &small),
            128 + 2 * (2 * 16 * 2)
        );
        assert_eq!(encoded_size(Codec::SimdBp512, &small), 256 + 2 * 32 * 2);
    }

    #[test]
    fn index_bytes_counts_groups_plus_one() {
        assert_eq!(index_bytes(Codec::VByte, 1000), None);
        let word = std::mem::size_of::<usize>();
        assert_eq!(index_bytes(Codec::Bp128, 300), Some(4 * word)); // 3 groups + 1
        assert_eq!(index_bytes(Codec::SimdBp512, 300), Some(2 * word)); // 1 group + 1
        assert_eq!(index_bytes(Codec::Raw, 0), Some(word)); // just the end entry
    }
}
