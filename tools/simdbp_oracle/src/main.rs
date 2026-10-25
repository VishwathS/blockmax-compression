//! Oracle for SIMDBP: runs Parker's real `compress` / `Decompressor` / `transcode`
//! (hierarchical_pruning, `src/index/compress_simdbp.rs`) on deterministic inputs and
//! writes the results as golden vectors for `codecs/tests/simdbp.rs`.
//!
//! Parker's group size is fixed when his crate is compiled (`BLOCK_SIZE` = 128, 256,
//! or 512, from the target features), so this program is built three times with
//! different RUSTFLAGS; see `regenerate.sh`.
//!
//! Usage:
//!   simdbp_oracle golden <out file> <build label> <expected block size> <parker commit> <rustflags> <rustc version>
//!   simdbp_oracle probe
//!
//! The input generators below MUST stay identical to `generate` in
//! `codecs/tests/simdbp.rs`. Every golden case stores a hash of its input
//! (`input_fnv`), so if the two ever drift apart the test fails instead of
//! silently comparing different data.

use std::fmt::Write as _;
use std::panic::catch_unwind;

use bmp::index::compress_simdbp::{compress, transcode, Decompressor, BLOCK_SIZE};

/// xorshift64: tiny, deterministic pseudo-random numbers (same as in the tests).
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

/// `n` values described by `spec`, for groups of `gl` values. Every non-`const`
/// generator draws exactly one random number per value.
fn generate(spec: &str, n: usize, gl: usize) -> Vec<u16> {
    let parts: Vec<&str> = spec.split(':').collect();
    let num = |i: usize| -> u64 { parts[i].parse().expect("bad number in spec") };
    match parts[0] {
        // Every value equal to V.
        "const" => vec![num(1) as u16; n],
        // Random values of exactly W bits (value 0 is forced to the max, 2^W - 1).
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
        // Uniform in 0..=MAX.
        "random" => {
            let (max, seed) = (num(1), num(2));
            let mut rng = XorShift64(seed);
            (0..n).map(|_| (rng.next() % (max + 1)) as u16).collect()
        }
        // Group g has width g % 9 (its first value is the max for that width).
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
        // All zeros except a few groups near the start, the metablock boundary, the
        // middle, and the end; nonzero group g has width g % 8 + 1.
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
        // Group g has width (g + 3 × (g / GL)) % 9: inside metablock m, slot s gets
        // width (s + 3m) % 9, so every header slot is nonzero in metablock 0 or 1 (all
        // header words, field positions, and lanes are used), and a slot has different
        // widths in different metablocks. Each group's first value is its max.
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
        // Group g has a pseudo-random width 0..=8, taken from a multiplicative hash of
        // g and SEED (not from the value stream). Each group's first value is its max.
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
        // Synthetic "block-max-like": 90% zeros, 9% in 1..=20, 1% in 21..=255.
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

/// The cases, for groups of `gl` values: (name, n, generator spec). Same list in
/// every build; sizes scale with the group size.
fn cases(gl: usize) -> Vec<(String, usize, String)> {
    let mut cases = vec![
        ("n1_zero".into(), 1, "const:0".into()),
        ("n1_max".into(), 1, "const:255".into()),
    ];
    for w in 0..=8 {
        cases.push((format!("width_{w}"), gl, format!("width:{w}")));
    }
    cases.extend([
        ("partial_small".into(), gl / 2 + 3, "random:255:11".into()),
        ("one_more_than_group".into(), gl + 1, "random:255:12".into()),
        ("one_less_than_group".into(), gl - 1, "random:15:13".into()),
        ("three_groups_exact".into(), 3 * gl, "random:255:14".into()),
        ("all_255".into(), 2 * gl + 5, "const:255".into()),
        ("group_widths".into(), 20 * gl + 17, "groupwidths:15".into()),
        ("blockmax_like".into(), 10 * gl + 100, "blockmax:16".into()),
        // Exactly one full metablock (gl groups of gl values).
        ("one_full_metablock".into(), gl * gl, "sparse:17".into()),
        // A second metablock with 3 groups, the last one partial.
        ("metablock_cross".into(), gl * gl + 2 * gl + 9, "sparse:18".into()),
        // A third metablock holding a single 1-value group.
        ("two_metablocks_plus_one".into(), 2 * gl * gl + 1, "sparse:19".into()),
        // Every header word, field position, and lane nonzero; three metablocks, the
        // last one with 6 groups (the last partial).
        ("all_header_fields".into(), 2 * gl * gl + 5 * gl + 7, "headers:20".into()),
        // Random per-group widths across a metablock boundary.
        ("random_group_widths".into(), gl * gl + 3 * gl + 1, "randgroups:21".into()),
    ]);
    cases
}

/// Outputs up to this many bytes are written out in full (`bytes=`); longer ones only
/// as a length and an FNV-1a hash, to keep the golden files small.
const MAX_INLINE_BYTES: usize = 32 * 1024;

/// FNV-1a, 64 bit: a simple, well-known hash, written out so the tests can repeat it.
fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// The u16 words as little-endian bytes (their in-memory layout on x86).
fn le_bytes(words: &[u16]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(2 * bytes.len());
    for b in bytes {
        write!(s, "{b:02x}").unwrap();
    }
    s
}

fn join<T: ToString>(items: &[T]) -> String {
    items.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
}

fn golden(args: &[String]) {
    let [out_path, label, expected_bs, commit, rustflags, rustc] = args else {
        panic!("golden needs: <out> <label> <block size> <commit> <rustflags> <rustc>");
    };
    let expected_bs: usize = expected_bs.parse().unwrap();
    // Refuse to write a file under the wrong name, e.g. if `target-cpu=native` on
    // another machine picks AVX2 instead of AVX-512.
    assert_eq!(
        BLOCK_SIZE, expected_bs,
        "this build of Parker's code uses BLOCK_SIZE {BLOCK_SIZE}, not {expected_bs}"
    );
    let bs = BLOCK_SIZE;

    let mut out = String::new();
    writeln!(out, "# SIMDBP golden vectors produced by Parker Carlson's own code").unwrap();
    writeln!(out, "# (hierarchical_pruning, src/index/compress_simdbp.rs, Apache-2.0).").unwrap();
    writeln!(out, "# DO NOT EDIT: regenerate with tools/simdbp_oracle/regenerate.sh").unwrap();
    writeln!(out, "# parker_commit={commit}").unwrap();
    writeln!(out, "# build={label} block_size={bs} rustflags=\"{rustflags}\" rustc=\"{rustc}\"").unwrap();
    writeln!(out, "# fields: name n gen input_fnv parker_roundtrip words widths offsets tc<size>=len:fnv tc_back bytes_fnv [bytes]").unwrap();
    writeln!(out, "#   words: length of compress() output in u16; bytes: that output as little-endian hex").unwrap();
    writeln!(out, "#   bytes_fnv: FNV-1a 64 of those bytes; bytes is present only if they are <= {MAX_INLINE_BYTES} bytes").unwrap();
    writeln!(out, "#   input_fnv: FNV-1a 64 of the generated input as little-endian u16 bytes").unwrap();
    writeln!(out, "#   widths/offsets: Decompressor.offsets / accumulated_offsets after init (offsets in units of {} u16)", bs / 16).unwrap();
    writeln!(out, "#   tc<size>: transcode() of the output to another block size: u16 length and FNV-1a of its bytes").unwrap();

    for (name, n, spec) in cases(bs) {
        let input = generate(&spec, n, bs);
        let input_fnv = fnv1a(le_bytes(&input));
        let compressed = compress(input.clone());

        // Parker's decoder: init once, then decode every group and compare.
        let n_groups = n.div_ceil(bs);
        let mut dcmp = Decompressor::new(n);
        dcmp.init(&compressed);
        let widths = dcmp.offsets[..n_groups].to_vec();
        let offsets = dcmp.accumulated_offsets[..n_groups].to_vec();
        let mut roundtrip_ok = true;
        for g in 0..n_groups {
            let end = ((g + 1) * bs).min(n);
            let decoded = dcmp.decompress_block(&compressed, g * bs);
            // The real values must match, and the padding of a short group must be 0.
            if decoded[..end - g * bs] != input[g * bs..end]
                || decoded[end - g * bs..].iter().any(|&v| v != 0)
            {
                roundtrip_ok = false;
            }
        }

        // His scalar transcoder to the other two block sizes, and back.
        let mut tc_fields = String::new();
        let mut tc_back_ok = true;
        for dst in [128, 256, 512] {
            if dst == bs {
                continue;
            }
            let other = transcode(&compressed, n, bs, dst);
            write!(tc_fields, " tc{dst}={}:{:#018x}", other.len(), fnv1a(le_bytes(&other))).unwrap();
            if transcode(&other, n, dst, bs) != compressed {
                tc_back_ok = false;
            }
        }

        let bytes = le_bytes(&compressed);
        let bytes_field = if bytes.len() <= MAX_INLINE_BYTES {
            format!(" bytes={}", hex(&bytes))
        } else {
            String::new() // too long: only the hash (bytes_fnv) is stored
        };
        writeln!(
            out,
            "case name={name} n={n} gen={spec} input_fnv={input_fnv:#018x} parker_roundtrip={} words={} widths={} offsets={}{tc_fields} tc_back={} bytes_fnv={:#018x}{bytes_field}",
            if roundtrip_ok { "ok" } else { "FAIL" },
            compressed.len(),
            join(&widths),
            join(&offsets),
            if tc_back_ok { "ok" } else { "FAIL" },
            fnv1a(bytes.iter().copied()),
        )
        .unwrap();
        eprintln!("{label}: {name} n={n} words={} roundtrip={roundtrip_ok} tc_back={tc_back_ok}", compressed.len());
    }
    std::fs::write(out_path, out).unwrap();
}

/// Experiments on the concerns found while reading Parker's code (P1, P2, P3, P5 in
/// `notes/design_decisions.md`). Uses only his public API; prints what happened.
fn probe() {
    let bs = BLOCK_SIZE;
    println!("== probe, BLOCK_SIZE={bs}");

    // P1: `Decompressor::init` doesn't reset the cached group (`current_block`).
    // Decode group 0 of row A, re-init the SAME decompressor for row B (same n, as
    // when search.rs reuses its preallocated decompressors for the next query), and
    // ask for group 0 again.
    let n = 3 * bs;
    let row_a: Vec<u16> = (0..n).map(|i| (i % 7) as u16).collect();
    let row_b: Vec<u16> = (0..n).map(|i| (100 + i % 5) as u16).collect();
    let (ca, cb) = (compress(row_a.clone()), compress(row_b.clone()));
    let mut dcmp = Decompressor::new(n);
    dcmp.init(&ca);
    let first_a = dcmp.decompress_block(&ca, 0)[..4].to_vec();
    dcmp.init(&cb);
    let got = dcmp.decompress_block(&cb, 0)[..4].to_vec();
    let verdict = if got == row_b[..4] {
        "NOT reproduced (returned row B)"
    } else if got == row_a[..4] {
        "REPRODUCED (returned row A's cached group)"
    } else {
        "unexpected values"
    };
    println!("P1 stale cache: row A group 0 starts {first_a:?}; after init(B), group 0 starts {got:?}; expected {:?} -> {verdict}", &row_b[..4]);
    // Controls: a fresh decompressor for B, and a different group cached first.
    let mut fresh = Decompressor::new(n);
    fresh.init(&cb);
    println!("P1 control (fresh decompressor): group 0 of B starts {:?}", &fresh.decompress_block(&cb, 0)[..4]);
    let mut other = Decompressor::new(n);
    other.init(&ca);
    other.decompress_block(&ca, bs); // group 1 of A is now cached
    other.init(&cb);
    println!("P1 control (group 1 of A cached): group 0 of B starts {:?}", &other.decompress_block(&cb, 0)[..4]);

    // P2: empty rows.
    let empty = catch_unwind(|| compress(Vec::new()));
    println!("P2 compress(vec![]): {}", if empty.is_err() { "PANICS" } else { "ok" });
    let new0 = catch_unwind(|| {
        let mut d = Decompressor::new(0);
        d.init(&[]);
    });
    println!("P2 Decompressor::new(0) + init(&[]): {}", if new0.is_err() { "PANICS" } else { "ok" });

    // P3: a value above 255 is stored at width 8 and loses its high bits.
    let mut big = vec![0u16; bs];
    big[3] = 300;
    let c = compress(big);
    let mut d = Decompressor::new(bs);
    d.init(&c);
    println!("P3 compress([.., 300, ..]) decodes to {}", d.decompress_block(&c, 0)[3]);

    probe_p5();
}

/// P5: `single_unpack256_6` (used by `decompress_register`, AVX2 builds only).
#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
fn probe_p5() {
    for width in 1..=8u32 {
        let max = ((1u32 << width) - 1) as u16;
        // 256 values with max exactly `max` (value 0 is the max).
        let mut rng = XorShift64(77 + u64::from(width));
        let input: Vec<u16> = (0..256)
            .map(|i| if i == 0 { max } else { rng.next() as u16 & max })
            .collect();
        let c = compress(input.clone());
        let mut d = Decompressor::new(256);
        d.init(&c);
        let mut bad = Vec::new();
        for s in 0..16 {
            let reg = d.decompress_register(&c, s * 16);
            // SAFETY: __m256i and [u16; 16] are both 32 bytes of plain data.
            let lanes: [u16; 16] = unsafe { std::mem::transmute(reg) };
            if lanes[..] != input[s * 16..s * 16 + 16] {
                bad.push(s);
            }
        }
        println!("P5 decompress_register width {width}: wrong register offsets {bad:?}");
    }
}

#[cfg(not(all(target_feature = "avx2", not(target_feature = "avx512f"))))]
fn probe_p5() {
    println!("P5: decompress_register only exists in AVX2 builds; skipped in this build");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("golden") => golden(&args[1..]),
        Some("probe") => {
            // Don't print the panic messages of the catch_unwind experiments.
            std::panic::set_hook(Box::new(|_| {}));
            probe();
        }
        _ => {
            eprintln!("usage: simdbp_oracle golden <out> <label> <block size> <commit> <rustflags> <rustc> | probe");
            std::process::exit(2);
        }
    }
}
