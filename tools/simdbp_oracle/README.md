# SIMDBP oracle

Runs **Parker's own SIMDBP code** (hierarchical_pruning, `src/index/compress_simdbp.rs`)
on deterministic inputs and writes the results as golden vectors, so the `codecs` tests
can check that `SimdBp128/256/512` are byte-identical to his implementation instead of
only checking that our encoder and decoder agree with each other.

This is a separate Cargo project (its own `[workspace]`): `cargo test` in the repo root
does not build or need it. The golden files it writes are committed, so the normal
tests work without Parker's repo, nightly Rust, or SIMD hardware.

## Regenerating the golden vectors

Requirements: Parker's repo at `../hierarchical_pruning` relative to this repo (or set
`PARKER_REPO`), the toolchain from `rust-toolchain.toml` (`nightly-2025-11-18`, which
his crate needs), and an x86_64 CPU with AVX-512 F+BW (for the 512 build).

```sh
tools/simdbp_oracle/regenerate.sh
```

It builds this crate three times, because his group size is chosen at compile time:

| build | RUSTFLAGS | his BLOCK_SIZE | writes |
|---|---|---|---|
| `sse` | (none) | 128 | `tests/data/simdbp/simdbp128.txt` |
| `avx2` | `-C target-feature=+avx2` | 256 | `tests/data/simdbp/simdbp256.txt` |
| `avx512` | `-C target-cpu=native` | 512 | `tests/data/simdbp/simdbp512.txt` |

and also writes `notes/simdbp_oracle_probes.txt` (the `probe` experiments). Build output
goes to `~/hp-target/oracle-<build>/` (override with `ORACLE_TARGET_DIR`). The script
stops if Parker's repo has local changes before or after the run, and the program
refuses to write a file if his `BLOCK_SIZE` doesn't match the build (for example, if
`target-cpu=native` picks AVX2 on another machine).

Then run `cargo test` in the repo root.

## Golden file format

Text, one line per case; lines starting with `#` are comments (they record his commit,
the build, RUSTFLAGS, and the rustc version). Each case line is `case` followed by
space-separated `key=value` fields:

| field | meaning |
|---|---|
| `name` | case name |
| `n` | number of values |
| `gen` | input generator spec, e.g. `width:3`, `random:255:11`, `sparse:18` (implemented identically in `src/main.rs` and `tests/simdbp.rs`) |
| `input_fnv` | FNV-1a 64 of the input values as little-endian u16 bytes, so a generator mismatch is caught |
| `parker_roundtrip` | `ok` if his `Decompressor` decoded every group back to the input (and the padding to 0) |
| `words` | length of his `compress` output in u16 |
| `widths` | his `Decompressor.offsets` (bit width per group) after `init` |
| `offsets` | his `Decompressor.accumulated_offsets` (group starts, in units of STRIDE u16) |
| `tc<size>` | his `transcode` of the output to another block size: `u16 length:FNV-1a of its bytes` |
| `tc_back` | `ok` if transcoding there and back gives his bytes again |
| `bytes_fnv` | FNV-1a 64 of his `compress` output bytes (every case) |
| `bytes` | his `compress` output as little-endian bytes, in hex; **only if it is at most 32 KiB** (`MAX_INLINE_BYTES`), otherwise left out |

The cases (sizes scale with the group size GL): n = 1 (0 and 255), one full group at
each width 0..=8, GL/2 + 3, GL + 1, GL − 1, 3·GL, all 255, 20 groups with widths
cycling 0..=8, an invented block-max-like mix, GL² (one full metablock),
GL² + 2·GL + 9 (second metablock), 2·GL² + 1 (third metablock with one value),
2·GL² + 5·GL + 7 with every header slot nonzero across three metablocks
(`headers:20`), GL² + 3·GL + 1 with random per-group widths (`randgroups:21`),
2·GL² + GL + 1 with width 8 in every group (`groupmax:8:22`: every slot of two whole
metablocks, then 2 groups in a third), and GL² + 1 with width 7 in every group
(`groupmax:7:23`: every slot of one metablock, then 1 value in a second). 25 cases.
With 32 KiB as the limit, 24 cases have full bytes at 128 (all but
`width8_every_slot`); at 256 and 512 the last four cases have only `words` +
`bytes_fnv`.

### Hash algorithm

FNV-1a, 64-bit: start from `0xcbf29ce484222325`; for each byte, XOR it in, then
multiply by `0x100000001b3` (wrapping). It is written out in both `src/main.rs` and
`tests/simdbp.rs`. It is a checksum, not a cryptographic hash: it detects accidental
differences, which is all these comparisons need.

### What the test does with each case (`tests/simdbp.rs`)

1. Regenerates the input from `gen` and requires its FNV-1a to equal `input_fnv`
   (proves our generator produced the oracle's exact input, before any output is
   compared).
2. With `bytes`: our encoding must equal them byte for byte. Without: our encoding
   must have length `2 × words` and FNV-1a `bytes_fnv`. (A hash match shows agreement
   for that input, with a negligible chance of coincidence; it is not a byte diff.)
3. Decodes his bytes (or ours, once they match) and checks widths, offsets, random
   access, and the transcode hashes.
4. `golden_cases_cover_every_header_bit` checks, from his widths, that the cases put a
   nonzero width in every header slot (every word, 4-bit position, and lane); that
   each of the 4 bits of every slot's field is 1 in some case and 0 in another; that
   one case has width 8 in every slot of at least two whole metablocks; and that
   nonzero widths appear in at least three metablocks.

### Checking that the committed files are reproducible

The program is deterministic, so regenerating on the same machine and toolchain must
give byte-identical files. To check without touching the working copy, regenerate in
a throwaway copy next to it (so `../../../hierarchical_pruning` still finds Parker's
repo), then compare:

```sh
rm -rf ~/codecs_audit
rsync -a --exclude target --exclude .git ~/codecs/ ~/codecs_audit/
~/codecs_audit/tools/simdbp_oracle/regenerate.sh
diff -r ~/codecs/tests/data/simdbp ~/codecs_audit/tests/data/simdbp
diff ~/codecs/notes/simdbp_oracle_probes.txt ~/codecs_audit/notes/simdbp_oracle_probes.txt
```

## Probes

`simdbp_oracle probe` runs small experiments on the concerns found while reading his
code (P1, P2, P3, P5 in `notes/design_decisions.md`), using only his public API, and
prints what happened.
