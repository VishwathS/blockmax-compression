# codecs: block-max compression research

Undergrad research project with Parker Carlson (PhD student, UCSB Search Systems Lab).
The user is new to Rust: explain each step in plain language, prefer clear code over
clever code, and comment non-obvious lines. Work in stages and stop after each one
for the user to say "continue".

## Ground rules
- `~/hierarchical_pruning` is Parker's repo and stays **read-only**. Never edit it.
- Only push changes to `VishwathS/blockmax-compression` when explicitly requested.
  Never modify or push to Parker's repository. Verify the Git remote
  (`git remote -v`) before every push. Committing in `~/codecs` is fine when a stage
  asks for it.
- Both repos live inside WSL Ubuntu (`/home/vishw/...`). Run cargo inside WSL, not on Windows.

## Research goal
The setting is learned sparse retrieval (SPLADE) on MS MARCO (8.8M passages). Parker's
LSP/SP code (Rust, built on BMP) stores per-term **block maxes**: for each term and each
block of consecutive doc IDs, the max impact score. A query sums these per block
("BoundSums") to prune blocks.
1. Characterize how block maxes are distributed, compared with BM25.
2. Benchmark compression codecs on that data.
3. Design storage that is smaller AND faster.

## Facts about Parker's repo (hierarchical_pruning)
- `BlockData` enum (`src/index/posting_list.rs`):
  - `Raw(Vec<u8>)`: dense
  - `Compressed(Vec<CompressedBlock>)`: sparse (offset, max) pairs
  - `SIMDBP(Vec<u16>)`
- `range_maxes: Vec<BlockData>` per term. Level 0 = superblocks, level 1 = blocks.
- SIMDBP (`src/index/compress_simdbp.rs`): u16 values, <= 8 bits. Groups of 128/256/512
  are packed at the group's max bit width. Metablock headers store each group's bit
  width as a 4-bit value, which allows random access.
- Hot path: `decompress_register` / `decompress_block` in `src/query/live_block.rs`.
- Quantization: block maxes are stored as `score >> quant` and compared against
  `threshold >> quant` in `search.rs` (around line 358).
- `bin/export_data.rs` dumps raw block-max matrices: `u32 n_terms`, `u32 n_blocks`, then
  a u8 T×C grid. It only works on a Raw index.
- Build: needs `nightly-2025-11-18` (`#![feature]` in `src/lib.rs`), x86_64 only. SIMD
  width is fixed at compile time via `target-cpu`:
  `cargo +nightly-2025-11-18 build --release --config 'build.rustflags=["-C","target-cpu=native"]'`
  We add `--target-dir ~/hp-target` so build output stays outside Parker's tree. Binaries
  are in `~/hp-target/release/`. This machine (i7-11800H) has AVX-512, so the build uses
  the SIMDBP **AVX-512 path (groups of 512, 32 u16s per register)**.

## This crate (codecs)
- `BlockCodec` trait in `src/lib.rs`. Contract:
  - Inputs are `u16`, every value `<= 255` (`codecs::MAX_VALUE`).
  - `encode(input: &[u16], out: &mut Vec<u8>)` appends bytes. It `assert!`s that
    every value is `<= 255` and panics otherwise, in debug **and** release builds.
  - `decode(input: &[u8], out: &mut [u16]) -> usize` fills `out` and returns the
    number of bytes consumed. `n` is not stored; it comes from `out.len()`.
  - Each `encode` call writes a self-contained section, so several sections can be
    appended to one buffer and decoded in order using the returned byte counts.
  - No random access yet; a later stage adds it (discuss trait changes first).
- Codecs (all implemented and tested):
  - `src/raw.rs`: `Raw`, 1 byte/value, the baseline, same as Parker's `BlockData::Raw`.
  - `src/vbyte.rs`: `VByte`, 7-bit groups, low-order first, stop bit = high bit set on
    the LAST byte; 0..=127 → 1 byte, 128..=255 → 2 bytes.
  - `src/bp128.rs`: `Bp128`, built on `bitpacking::BitPacker4x`. Values are cut into
    128-value chunks, and each chunk is packed at the bit width of its max (0..=8).
    Format: `[one width byte per chunk][chunk 0 data][chunk 1 data]...`, with
    chunk data = 16 × width bytes. All width bytes come first (like SIMDBP headers),
    so chunk k's offset = `G + 16 × sum(widths[..k])`, which a later random-access
    stage can use. A width-0 (all-zero) chunk costs 1 byte. A short last chunk is
    zero-padded to 128 (like SIMDBP's last group). u16 is widened to u32 for the crate.
  - Caveat: the crate uses a scalar packer in debug builds and SIMD in release builds.
    `bp128_known_bytes` pins the bytes, so run `cargo test --release` too.
- Codec registry: `src/registry.rs`, re-exported as `codecs::Codec`. An enum with one
  variant per codec; `Codec::ALL` lists them all (Raw, VByte, Bp128), `name()` gives a
  stable display name, `from_name()` looks one up (case-insensitive), and
  `block_size()` / `encode()` / `decode()` forward to the codec type via `match`. It
  doesn't change the trait or any byte format. Anything that loops over codecs
  (reports, CLI, benchmarks) should use `Codec::ALL`, not list codecs by hand.
  To add a codec:
  1. implement `BlockCodec` in its own file
  2. add a variant to `Codec` and append it to `Codec::ALL`
  3. add an arm to each `match` in `registry.rs` (the compiler flags any missing one)
  4. update the expected list in the `registry_lists_exactly_the_implemented_codecs` test
- Size report: `cargo run --release --example size_report` (synthetic datasets, one
  bytes-per-value column per registered codec). Its unit test runs with `cargo test`
  (`test = true` in `Cargo.toml`).
- Tests are in `tests/roundtrip.rs` (generic roundtrip helper + proptest, exact-format
  tests, panic tests, registry tests that compare registry output with direct calls).
- Open design questions (details in `notes/code_map.md` section (c)): parameterized
  codecs as enum variants with fields, measuring `match` dispatch cost in stage 8, and
  whether SIMDBP fits the trait without random access.
- Before committing, `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test`, and `cargo test --release` must all pass. This crate builds on stable;
  the nightly toolchain is only for Parker's repo.
- Notes go in `notes/` (e.g. `notes/code_map.md`, which also has the codec status).

## Stages
Done: (1) BlockCodec trait + Raw + VByte, (2) switch to u16 values <= 255, (3) BP128,
(4) codec registry (`codecs::Codec`, size report loops over it).
Next, in order (do not start one without the user saying "continue"):
5. Real index loader: read `export_data` matrices (`u32 n_terms`, `u32 n_blocks`, u8
   T×C grid) so codecs run on real MS MARCO block maxes instead of synthetic data.
6. SIMDBP integration: port or call Parker's SIMDBP format as a codec, to compare
   against it directly (read-only use of his repo).
7. CLI: run size/speed reports on a chosen dataset and codec set.
8. Benchmarking: criterion encode/decode speed, and bytes per value on real data.
