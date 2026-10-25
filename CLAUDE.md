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
- Issues found in his SIMDBP code in stage 5, reproduced with his public API (details
  and exact conditions in `notes/design_decisions.md` section 5; his repo untouched):
  P1 `Decompressor::init` doesn't reset the cached group, so a reused decompressor can
  return the previous row's group (search reuses them across queries; effect on search
  results NOT tested); P2 empty rows panic; P3 values > 255 are silently truncated
  (300 → 44); P5 AVX2 `decompress_register` is wrong for width 6, offset 10 (dead code).

## This crate (codecs)
- `BlockCodec` trait in `src/lib.rs`. Contract:
  - Inputs are `u16`, every value `<= 255` (`codecs::MAX_VALUE`).
  - `encode(input: &[u16], out: &mut Vec<u8>)` appends bytes. It checks that every
    value is `<= 255` and panics otherwise, in debug **and** release builds, BEFORE
    writing anything (`check_values` in `src/lib.rs`), so after that panic `out` is
    unchanged (B1).
  - Decoding is lenient on purpose (B2): byte patterns our encoders never write
    (VByte over-long values, nonzero padding bits, unused SIMDBP header slots) are
    accepted, like Parker's decoder. `n` is not stored, so the caller must pass the
    exact logical row length; a wrong `n` is not detected in general (B3).
  - `decode(input: &[u8], out: &mut [u16]) -> usize` fills `out` and returns the
    number of bytes consumed. `n` is not stored; it comes from `out.len()`.
  - Each `encode` call writes a self-contained section, so several sections can be
    appended to one buffer and decoded in order using the returned byte counts.
  - The trait itself has no random access; that lives in a separate trait (below), so
    `BlockCodec` and every existing byte format are unchanged.
- Random access: `src/random_access.rs`, re-exported as `codecs::RandomAccessCodec` and
  `codecs::GroupIndex`. A **row** = the output of one `encode` call; a **group** =
  `GROUP_LEN` consecutive values (the last group may be partial).
  - `RandomAccessCodec: BlockCodec` (stateless, like `BlockCodec`): `GROUP_LEN`,
    `num_groups(n)`, `group_len(n, g)`, `group_range(input, n, g) -> Range<usize>` (byte
    range of group g's data, found from metadata only), `locate_groups(input, n, &mut
    Vec<usize>)` (all G+1 starts in one pass), `decode_group_bytes(data, out)`, and the
    one-shot `decode_group(input, n, g, out) -> usize` (writes the group's values to the
    front of `out`, returns how many; `out[len..]` is untouched).
  - `GroupIndex<C>`: `build(input, n)` runs `locate_groups` once (O(G)), then
    `decode_group(input, g, out)` locates in O(1). Reusing one index for many rows
    doesn't allocate once its `Vec` is big enough. RAM cost: 8 bytes × (G+1), about
    0.0625 B/value for 128-value groups, held only while a row is open; nothing is
    added to the encoded bytes. `heap_bytes()` reports it.
  - Supported: `Raw` (group = 128 bytes at `128 × g`, O(1) without an index),
    `Bp128` (group = one 128-value chunk; one-shot locate sums the g width bytes before
    it, **O(g)**, so use `GroupIndex` for many lookups; decode = one 128-value unpack),
    and `SimdBp128/256/512` (group = one SIMDBP group of 128/256/512; one-shot locate
    sums the g 4-bit header fields before it, **O(g)**).
    `VByte` does **not** implement it: positions depend on every earlier value.
  - Registry: `Codec::group_len()` (`None` for VByte) and `Codec::decode_group()`
    (returns `Option<usize>`).
  - Invalid group index, too-small `out`, or truncated/invalid input → panic with a
    message ("out of range", "room for", "input ended", "width bytes", ...).
  - Relation to Parker's SIMDBP: same idea as `Decompressor::init` (prefix sum of the
    widths, O(G)) + `decompress_block` (O(1) locate, decode the whole group), but with
    no cached "current group" inside the index (the caller owns the output buffer),
    which avoids his stale-cache issue (P1).
- Codecs (all implemented and tested):
  - `src/raw.rs`: `Raw`, 1 byte/value, the baseline, same as Parker's `BlockData::Raw`.
  - `src/vbyte.rs`: `VByte`, 7-bit groups, low-order first, stop bit = high bit set on
    the LAST byte; 0..=127 → 1 byte, 128..=255 → 2 bytes.
  - `src/bp128.rs`: `Bp128`, built on `bitpacking::BitPacker4x`. Values are cut into
    128-value chunks, and each chunk is packed at the bit width of its max (0..=8).
    Format: `[one width byte per chunk][chunk 0 data][chunk 1 data]...`, with
    chunk data = 16 × width bytes. All width bytes come first (like SIMDBP headers),
    so chunk k's offset = `G + 16 × sum(widths[..k])`, which is how random access
    finds a chunk. A width-0 (all-zero) chunk costs 1 byte. A short last chunk is
    zero-padded to 128 (like SIMDBP's last group). u16 is widened to u32 for the crate.
  - Caveat: the crate uses a scalar packer in debug builds and SIMD in release builds.
    `bp128_known_bytes` pins the bytes, so run `cargo test --release` too.
  - `src/simdbp/`: `SimdBp128` / `SimdBp256` / `SimdBp512` (= `SimdBp<STRIDE>` with
    STRIDE 8 / 16 / 32), **Parker's SIMDBP format, byte-identical to his `compress`**
    (u16 words stored little-endian). Format: `[M metablock headers, 8 × STRIDE bytes
    each][group data, 2 × STRIDE × width bytes each]`; group = 16 × STRIDE values packed
    at the bit width of their max in his lane layout; header = the groups' widths as 4-bit
    fields (same lane layout); one metablock = 16 × STRIDE groups; last group zero-padded;
    `n = 0` → no bytes (his code panics). All three exist in every build (his code picks
    one at compile time). Packing uses `simdbp::Kernel`: `Scalar` (port of his
    `pack_scalar`) or his generated SSE3 / AVX2 / AVX-512 (F+BW) kernels, chosen at run
    time by `Kernel::best`; all give the same bytes. A test-only counter
    (`kernel::trace`, `cfg(test)` only) records which kernel ran;
    `src/simdbp/kernel_selection_tests.rs` uses it to prove the SIMD kernels execute
    (and prints `SKIPPED` where the CPU lacks them). The kernels in
    `src/simdbp/kernels/{sse3,avx2,avx512}.rs` are vendored by
    `scripts/vendor_simdbp_kernels.py` (Apache-2.0; `third_party/hierarchical_pruning/`
    has the license and provenance): **never edit them by hand**, re-run the script.
    Every row pays at least one header (64 / 128 / 256 bytes), so encode a long sequence
    (e.g. a term's whole level) as one row.
- Codec registry: `src/registry.rs`, re-exported as `codecs::Codec`. An enum with one
  variant per codec; `Codec::ALL` lists them all (Raw, VByte, Bp128, SimdBp128,
  SimdBp256, SimdBp512), `name()` gives a
  stable display name, `from_name()` looks one up (case-insensitive), and
  `block_size()` / `encode()` / `decode()` forward to the codec type via `match`. It
  doesn't change the trait or any byte format. Anything that loops over codecs
  (reports, CLI, benchmarks) should use `Codec::ALL`, not list codecs by hand.
  To add a codec:
  1. implement `BlockCodec` in its own file (and `RandomAccessCodec` if it can seek
     cheaply)
  2. add a variant to `Codec` and append it to `Codec::ALL`
  3. add an arm to each `match` in `registry.rs` (the compiler flags any missing one;
     `group_len` / `decode_group` return `None` for codecs without random access)
  4. update the expected list in the `registry_lists_exactly_the_implemented_codecs` test
- Size report: `cargo run --release --example size_report`. SYNTHETIC datasets only
  (labelled as such; not research results). Each dataset is encoded as **one row**
  (encoding many short rows would charge SIMDBP a header per row). Table 1: encoded
  bytes per value per registered codec; table 2: `GroupIndex` RAM per value, kept
  separate (it isn't stored). Its unit tests run with `cargo test` (`test = true` in
  `Cargo.toml`).
- Tests are in `tests/roundtrip.rs` (generic roundtrip helper + proptest, exact-format
  tests, panic tests, registry tests that compare registry output with direct calls),
  `tests/random_access.rs` (every group, one-shot and via `GroupIndex`, checked
  against sequential `decode`, for every random-access codec; exact Bp128 offsets;
  out-of-order access; zero and partial groups; several rows in one buffer; invalid
  indices and truncated input; proptest with per-group random widths),
  `tests/simdbp.rs` (**golden vectors from Parker's own code**, header-slot coverage,
  hand-computed bytes, scalar vs SIMD kernels, invalid input), `tests/bp128_layout.rs`
  (Bp128 bytes vs an independent model of the BitPacker4x layout, widths 0..=8),
  `tests/contracts.rs` (B1 buffer unchanged after a rejected value, P1-style stale-index
  regression, truncated `GroupIndex` input, B2 lenient decoding, B3 wrong `n`),
  `tests/boundaries.rs` (GL² − 1 / GL² / GL² + 1 rows for every codec; opt-in
  ~8.8M-value stress test: `cargo test --release --test boundaries -- --ignored`),
  `tests/end_to_end.rs` (one 270,001-value row through the registry for every codec,
  with sizes computed independently), and the unit tests in
  `src/simdbp/kernel_selection_tests.rs` (which kernel ran). `notes/code_map.md` has a
  table of what each file proves and whether it is independent of our own decoder.
- Golden vectors: `tests/data/simdbp/simdbp{128,256,512}.txt`, 23 cases each, produced
  by `tools/simdbp_oracle` (a separate nightly crate that depends on Parker's repo
  read-only; NOT part of `cargo test`). Outputs up to 32 KiB are stored in full; larger
  ones as length + FNV-1a 64 hash (`bytes_fnv`); every case stores `input_fnv` so the
  test proves its regenerated input equals the oracle's. Regenerate with
  `tools/simdbp_oracle/regenerate.sh`; its README documents the format, the hash, and
  how to check reproducibility in a throwaway copy (`~/codecs_audit`). `cargo test`
  needs only the committed files. Set `CODECS_REQUIRE_SIMD=1` to make the tests fail if
  the SSE3 / AVX2 / AVX-512 kernels aren't all available (instead of printing SKIPPED).
- Research-validity rules (see `notes/design_decisions.md` section 7): byte
  compatibility with Parker is not speed equivalence; synthetic results are not
  evidence about real block maxes; encoder-vs-own-decoder tests don't verify a format;
  a hash match shows agreement for that input only.
- Design decisions (all of them, with class A/B/C/D, evidence, and whether they were
  changed or deferred), what is confirmed compatible with Parker's code vs only
  self-consistent, and measured results vs hypotheses: `notes/design_decisions.md`.
- Open design questions: parameterized codecs as enum variants with fields, measuring
  `match` dispatch cost in stage 9, whether `GroupIndex` should store `u32` offsets to
  halve its RAM, zero-copy SIMDBP decode (bytes are copied into a stack buffer now),
  and panics vs `Result` once files are read from disk (stage 8).
- Before committing, `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
  `CODECS_REQUIRE_SIMD=1 cargo test`, and `CODECS_REQUIRE_SIMD=1 cargo test --release`
  must all pass. If the oracle, the golden files, or a SIMDBP byte format changed, also
  run the reproducibility check in `tools/simdbp_oracle/README.md`. This crate builds on
  stable; the nightly toolchain is only for Parker's repo.
- Notes go in `notes/` (e.g. `notes/code_map.md`, which also has the codec status).

## Stages (roadmap)
1. Foundation: `BlockCodec` trait, Raw and VByte (incl. the switch to u16 values <= 255). Done.
2. BP128. Done.
3. Codec registry (`codecs::Codec`, size report loops over it). Done.
4. Random-access infrastructure (`RandomAccessCodec` + `GroupIndex`; branch
   `feature/random-access`; no byte format changed). Done.
5. Parker SIMDBP integration and comprehensive correctness audit (**current**, branch
   `feature/simdbp-integration`): `SimdBp128/256/512`, byte-identical to Parker's
   `compress` (golden vectors from his code, every header slot covered); VByte decode
   rejects values > 255; size report encodes one row per dataset; encoders reject bad
   values before writing (B1); B2/B3 documented and pinned; kernel execution traced;
   independent Bp128 layout model; boundary and opt-in stress tests.
Next, in order (do not start one without the user saying "continue"):
6. Seeded synthetic data infrastructure.
7. CLI and experiment configuration.
8. Real-data loading and integration (read `export_data` matrices: `u32 n_terms`,
   `u32 n_blocks`, u8 T×C grid), pending Parker's guidance/data. See
   `notes/design_decisions.md` section 8 for loader concerns.
9. Benchmarking methodology and experiments, pending discussion with Parker.
