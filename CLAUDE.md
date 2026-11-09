# codecs: block-max compression research

Undergrad research project with Parker Carlson (PhD student, UCSB Search Systems Lab).
The user is new to Rust: explain each step in plain language, prefer clear code over
clever code, and comment non-obvious lines. Work in stages and stop after each one for
the user to say "continue".

Machine-wide rules (no sudo, no deletes, no killing processes, Parker's repo and
`/home/sammy/BMP/data/` read-only, where files may go, how binaries are run) live in
`~/lsp-work/CLAUDE.md`. They apply here and are not repeated.

## Where this runs
- **nairobi** (shared lab machine, i7-1260P): the main workspace and the source of all
  benchmark numbers. Repo: `~/lsp-work/blockmax-compression`. Parker's repo (read-only):
  `~/lsp-work/hierarchical_pruning`.
- **Laptop** (WSL, i7-11800H, AVX-512): only for AVX-512 testing and offline reading.
- Sync the two copies through GitHub (`VishwathS/blockmax-compression`). Never edit the
  same branch in both places.
- nairobi has SSE3 + AVX2, **no AVX-512**. `Kernel::best` picks SimdBp128 → SSE3,
  SimdBp256 → AVX2, **SimdBp512 → Scalar**, so SimdBp512 timings on nairobi are scalar
  code. The AVX-512 kernels are only exercised on the laptop.
- Toolchain: stable Rust (no `rust-toolchain` file; nairobi's default is 1.99.1),
  edition 2024. The nightly toolchain is only for Parker's repo and `tools/simdbp_oracle`.

## Git
- Commit as the user only: `git -c user.name="Vishwath Shankar" -c
  user.email="vishwath@ucsb.edu" commit ...`. **No `Co-Authored-By` line.**
- Never push.

## Pre-commit checks (nairobi), all under `nice -n 10`
```
cargo fmt --check
cargo clippy --all-targets -- -D warnings
GRID_SMOKE_REQUIRE=1 cargo test
GRID_SMOKE_REQUIRE=1 cargo test --release
```
- `GRID_SMOKE_REQUIRE=1` turns a missing smoke grid into a failure instead of SKIPPED.
- **Laptop only:** `CODECS_REQUIRE_SIMD=1` (fails if SSE3/AVX2/AVX-512 aren't all
  available, so it fails on nairobi by design), and regenerating the SIMDBP golden
  vectors (`tools/simdbp_oracle/regenerate.sh`, needs AVX-512 for the 512 build). If
  the oracle, golden files, or a SIMDBP byte format change, run the reproducibility
  check in `tools/simdbp_oracle/README.md` on the laptop.
- Opt-in stress test: `cargo test --release --test boundaries -- --ignored`.

## Data (`~/data/`, from Parker's `export_data`)
Grid format: `u32 n_terms`, `u32 n_blocks`, then a u8 T×C matrix (one row per term).
Each prefix has `._blocks.bin` (8-doc blocks), `._superblocks.bin` (128-doc
superblocks), `._queries.bin`.
- `smoke_128_8.*`: small nfcorpus grid (15,572 terms; 455 blocks / 29 superblocks), made
  by `~/lsp-work/scripts/smoke_test.sh`. Used by the end-to-end tests in
  `tests/grid.rs` and `tests/stats.rs` (override with `GRID_SMOKE_PREFIX`) and for quick
  `grid_bench` dry runs. Not research data.
- `splade_128_8.*`: the real data, BP-ordered SPLADE v3 on MS MARCO (28,638 terms;
  blocks grid 1,105,228 columns, 31.65 GB; superblocks 69,077 columns, 1.98 GB).
- `shuffled_128_8.*`: same data in shuffled doc order. **Superblocks and queries only**
  (no blocks grid).

## grid_bench (`examples/grid_bench.rs`)
Build: `nice -n 10 cargo build --release --example grid_bench`.
- `--mode size`: encodes every row with every codec (round trips checked); exact totals,
  overall and per stratum.
- `--mode speed`: encode / full-row decode timings on a sampled set of rows (per
  stratum, top terms, query terms); median of `--reps` after a warm-up.
- `--mode stats`: value histogram, zero runs, gaps, nonzeros per group, and analytic
  size ESTIMATES for sparse/hybrid formats (`src/stats.rs`). SimdBp128 numbers are
  checked against the real encoder.
Rules for running it:
- Claude may run it itself, including full-grid scans:
  `nice -n 10 taskset -c 2 target/release/examples/grid_bench --mode ... --grid ... --out ...`
- One run at a time. Check `uptime` and `free -g` first; if other users' load or memory
  use is high, wait and tell the user.
- Outputs go to `~/lsp-work/results/<name>`; never overwrite (the program refuses to
  start if an output exists; pick a new name such as `_r2`).
- Check with the user first for: Parker's tools (only via the vetted scripts, see
  `~/lsp-work/CLAUDE.md`) and anything that may need more than ~1 GB of RAM.

## Crate facts
- `BlockCodec` (`src/lib.rs`): values are `u16 <= 255`. `encode(input, out)` appends a
  self-contained section and panics on a value > 255 before writing anything (B1).
  `decode(input, out) -> bytes consumed`; `n` comes from `out.len()` and isn't stored
  (a wrong `n` isn't detected, B3). Decoding is lenient on purpose (B2).
- `RandomAccessCodec` + `GroupIndex` (`src/random_access.rs`): decode one group of
  `GROUP_LEN` values. Raw, Bp128, SimdBp128/256/512 implement it; VByte doesn't.
  One-shot locate is O(g) for Bp128/SIMDBP; `GroupIndex` makes it O(1).
- Codecs: `Raw`, `VByte`, `Bp128` (bitpacking BitPacker4x, width bytes first), and
  `SimdBp128/256/512` (`src/simdbp/`): **byte-identical to Parker's `compress`**,
  checked against golden vectors from his code. Kernels in `src/simdbp/kernels/` are
  vendored by `scripts/vendor_simdbp_kernels.py`: never edit them by hand.
  Encode a whole term level as one row (every SIMDBP row pays a header).
- Registry (`src/registry.rs`, `codecs::Codec`): loop over `Codec::ALL`, never list
  codecs by hand. Adding a codec: implement the trait(s), add a variant to `Codec` and
  `Codec::ALL`, fill every `match` arm, update
  `registry_lists_exactly_the_implemented_codecs`.
- Real data: `src/grid.rs` (memmapped `Grid`, row release keeps RSS at a few MB,
  `QueryTerms`, strata S0–S4, `size_scan`), `src/stats.rs` (`Stats`, `stats_scan`).
- Bp128 uses a scalar packer in debug and SIMD in release: run tests in both.

## Where to read more
- `notes/code_map.md`: Parker's storage code, our codecs, test map (what each test
  file proves and whether it is independent of our decoder), loader and stats.
- `notes/design_decisions.md`: every design decision, Parker's SIMDBP issues P1–P5,
  research-validity rules (byte compatibility ≠ equal speed; synthetic results say
  nothing about real block maxes; encoder-vs-own-decoder tests don't verify a format).
- `~/lsp-work/NOTES.md` and `~/lsp-work/results/`: run logs and measured results.

## Status
Done (all on `main`): Raw, VByte, Bp128, codec registry, random access, the SIMDBP port
and its correctness audit, the real-data grid loader with `grid_bench` size/speed
modes, and the stats mode with analytic size estimates (incl. the hybrid estimate).

Next: prototype the **hybrid group format** (sparse groups stored as position + value
pairs, dense groups bit-packed). Measure decode speed first, then size. Don't start
without the user saying "continue".
