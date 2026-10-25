# codecs: block-max compression research

Undergrad research project with Parker Carlson (PhD student, UCSB Search Systems Lab).
The user is new to Rust: explain each step in plain language, prefer clear code over
clever code, and comment non-obvious lines. Work in stages and stop after each one
for the user to say "continue".

## Ground rules
- `~/hierarchical_pruning` is Parker's repo and stays **read-only**. Never edit it.
- Never `git push`. Committing in `~/codecs` is fine when a stage asks for it.
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
  - `encode(input: &[u16], out: &mut Vec<u8>)` appends bytes. It `debug_assert!`s that
    every value is `<= 255`.
  - `decode(input: &[u8], out: &mut [u16]) -> usize` fills `out` and returns the
    number of bytes consumed.
  - No random access yet; a later stage adds it.
- Codecs: `src/raw.rs` (`Raw`, 1 byte/value, the baseline, same as Parker's
  `BlockData::Raw`) and `src/vbyte.rs` (`VByte`: 7-bit groups, low-order first, stop
  bit = high bit set on the LAST byte; 0..=127 → 1 byte, 128..=255 → 2 bytes).
- Size report: `cargo run --release --example size_report` (synthetic datasets).
- Tests are in `tests/roundtrip.rs` (generic roundtrip helper + proptest).
- Before committing, `cargo fmt`, `cargo clippy`, and `cargo test` must all pass.
- Notes go in `notes/` (e.g. `notes/code_map.md`).
