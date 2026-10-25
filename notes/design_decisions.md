# Design decisions

Every design decision in this crate that was found in the stage-5 review and audit
(Parker SIMDBP integration + correctness audit, branch `feature/simdbp-integration`),
including the arbitrary and unvalidated ones, with what happened to each. Update this
file when a decision changes.

Stages (same numbering as `CLAUDE.md`): 1 Raw + VByte, 2 Bp128, 3 codec registry,
4 random access, **5 SIMDBP integration + correctness audit (current)**, 6 seeded
synthetic data, 7 CLI and experiment configuration, 8 real-data loading (pending
Parker's guidance/data), 9 benchmarking (pending discussion with Parker).

**Classes**

- **A**: technically justified (there is a concrete reason or evidence).
- **B**: reasonable but not validated (no measurement or test says it's the best choice).
- **C**: arbitrary or legacy (no known reason).
- **D**: potentially problematic (can give wrong results or mislead).

**Status**: *kept* (unchanged), *changed in stage 5* (with the reason), or *deferred*
(left as is on purpose; the stage that should revisit it is named).

"Rationale: unknown" means nobody recorded a reason; none has been invented here.

## 1. Decisions that existed before stage 5

| # | Decision | Behavior | Rationale | Class | Evidence | Alternatives | Status |
|---|---|---|---|---|---|---|---|
| 1 | `u16` values, each <= 255 | `encode(&[u16])`, `decode(.., &mut [u16])` | Parker's query loop and SIMDBP work on u16 (room to multiply by a query weight) | A | `compress(Vec<u16>)`, `inverted_index.rs:300` widens u8 to u16 | u8 interface | kept |
| 2 | Range check with a real panic in every build | values > 255 panic in debug **and** release | never store a wrong value silently | A | Parker's SIMDBP stores 300 as 44 (P3 below, reproduced) | `debug_assert!` | kept; also applied to the SIMDBP codecs; since stage 5 done before any byte is written (B1) |
| 3 | VByte decode did not range-check | `[0x7F, 0xFF]` decoded to 16383 without a panic, against the trait contract | none (an oversight) | D | code reading, then a test | — | **changed in stage 5**: decode panics on a value > 255 (`vbyte_panics_on_a_two_byte_value_above_255`); valid bytes decode exactly as before |
| 4 | `n` is not stored | decoder gets it from `out.len()`, random access from an `n` argument | every term of a level has the same `n`, like `Decompressor::new(n)` | A | `lib.rs` docs; Parker stores one row per term per level with `n = ceil(num_docs / bsize)` (`inverted_index.rs:146-160`) | length prefix | kept. A wrong `n` gives garbage, zeros, or a panic, never UB; see B3 |
| 5 | Each `encode` call is a self-contained section | sections can be appended and decoded in order | convenient for tests and multi-row buffers | B | tests only | — | kept; SIMDBP rows follow it too |
| 6 | `BLOCK_SIZE` per codec (128 for Raw/VByte/Bp128) | "preferred values per call", a hint | Raw/VByte: unknown; Bp128: its chunk size | C (D for the old size report) | the old size report encoded every `block_size()` values as a separate row, which charged SIMDBP one header per 128 values | remove it; make it per-row | constant **kept** (SimdBp sets it to its group length, documented as a hint only); the size report **changed** (#21b) |
| 7 | Bp128 uses 128-value chunks | fixed | `BitPacker4x::BLOCK_LEN` is 128: chosen with the library, not measured | B | `bp128.rs` | 32 (BitPacker1x), 256/512 | deferred to stages 8–9 (real data, benchmarks) |
| 8 | Bp128 stores 1 byte per width | 0.0078 B/value of metadata | simplicity (4 bits would save 0.004 B/value) | B | size report: all zeros 0.008 vs SimdBp128 0.004 | 4-bit packed widths | deferred (real data first, stage 8) |
| 9 | All widths before all data (Bp128) | `[widths][data]` | needed for random access; mirrors SIMDBP | A | `bp128.rs`, random-access tests | interleaved | kept |
| 10 | Bp128 uses the `bitpacking` crate's 4-lane layout | crate-defined bytes | unknown (the crate was available) | B | debug = scalar `_simple`, release = SSE3 chosen at run time; since stage 5 checked against an **independent model** of the layout for widths 0..=8 in both builds (`tests/bp128_layout.rs`) | own packer | kept |
| 11 | The last group is zero-padded | up to `(GL−1) × w` padding bits per row | A for Bp128 (the crate needs 128 values) and for SIMDBP (part of Parker's format); the size cost is B | A/B | `partial_group_padding_is_zero_and_not_written_out` | separate tail encoding | kept; cost matters only for many tiny rows |
| 12 | Raw `GROUP_LEN` = 128 | 128-value groups | comment: "any size would work" | C | `raw.rs` | any | kept (harmless) |
| 13 | Random access is a separate trait | `RandomAccessCodec: BlockCodec`; VByte doesn't implement it | keeps `BlockCodec` and every format unchanged | A | SIMDBP fit without any trait change | `Option`-returning methods on `BlockCodec` | kept |
| 14 | `GroupIndex` stores `usize` byte offsets | 8 B × (G+1) RAM | simplicity | B | size report table 2 | `u32` offsets (half the RAM) | deferred to stage 9 |
| 15 | A group's width is recovered from its byte length | `decode_group_bytes(data, out)` | implicit in the trait | B | Bp128: `len / 16`; SimdBp: `len / (2 × STRIDE)` | pass the width in | kept; would not work for a codec whose group size isn't a function of one width (e.g. PFor) |
| 16 | No cached group inside the index | the caller owns the output buffer | caller can skip re-decoding; no hidden state | **A (strengthened)** | Parker's `Decompressor` caches and returns stale data after `init` (P1, reproduced); our same sequence is a regression test (`reused_index_never_returns_the_previous_rows_group`) | cache like Parker's | kept |
| 17 | Errors are panics, not `Result` | every invalid input panics with a message | encoding is offline, data is trusted; Parker panics too | B (D risk at stage 8) | — | `Result` | deferred to stage 8 (files read from disk) |
| 18 | `decode_group` doesn't touch `out[len..]` | no zero padding past `n` | documented contract | B | tests with sentinel buffers | zero-fill | kept; **differs from Parker**, whose `decompress_block` returns the full zero-padded group that his query loops read; revisit when wiring codecs into a query loop |
| 19 | Registry = enum + `match` | one variant per codec, `Codec::ALL` | exhaustiveness, stable names | A (dispatch cost B) | 6 codecs added without trait changes | trait objects | kept; time dispatch in stage 9 |
| 20 | Registry has only the one-shot `decode_group` | O(g) per call through the registry | none recorded | B | — | an indexed registry API | deferred (not needed yet) |
| 21a | Size-report datasets | uniform, 4-bit, zeros, "90/9/1" mix | placeholders until real data; the 90/9/1 proportions are invented | C | `examples/size_report.rs` | real `export_data` matrices | kept, **labelled synthetic** in the output and docs; seeded synthetic data is stage 6, real data stage 8 |
| 21b | Size report: one row per `block_size()` values | see #6 | unknown | D | old output: SIMDBP all zeros = 0.500 B/value | one row per dataset | **changed in stage 5**: each dataset is one row. Raw/VByte/Bp128 numbers are identical to before (checked: same output; `splitting_into_rows_changes_size_only_for_codecs_with_row_headers`) |
| 21c | Size report counted only encoded bytes | index RAM invisible | — | B | — | add a column | **changed in stage 5**: a second table shows `GroupIndex` RAM per value, kept separate from the encoded size |
| 22 | Test independence | most tests were round trips or compared with the same codec's `decode` | — | B | only `bp128_known_bytes`, VByte bytes, and hand-computed Bp128 offsets were independent | golden vectors | **changed in stage 5**: Parker-generated golden vectors, hand-computed SIMDBP bytes, an independent Bp128 layout model, independently computed sizes; see sections 4 and 6 |

## 2. New decisions in stage 5 (SIMDBP and the audit)

| # | Decision | Behavior | Rationale | Class | Evidence | Alternatives | Status |
|---|---|---|---|---|---|---|---|
| S1 | Three codecs, one per group size | `SimdBp<8/16/32>` = `SimdBp128/256/512`; other strides don't compile | in Parker's code the group size depends on compile flags; here the bytes never do | A | golden files from his 3 builds all match | one codec chosen by build flags (his way) | kept |
| S2 | u16 words stored as little-endian bytes | trait works on `&[u8]` | equals his `Vec<u16>` in x86 memory | A | golden: byte-identical | native-endian view (`align_to`) | kept |
| S3 | `n = 0` encodes to 0 bytes | empty rows work | the trait allows empty input; Parker's `compress` panics (P2) | B (our extension) | `empty_row_is_zero_bytes` | panic like Parker | kept; his code can't produce or read such a row, so there's no compatibility case to break |
| S4 | Vendored Parker's SIMD kernels, widths 0..=8 only | `scripts/vendor_simdbp_kernels.py` copies the function bodies unchanged and makes each an `unsafe fn` with `#[target_feature]` | values are <= 255, so widths 9..=16 are dead code; mechanical copy = auditable | B | `kernels_agree_*` tests; re-running the script against his repo reproduces the committed files exactly (checked in stage 5) | hand-written intrinsics; scalar only | kept |
| S5 | Kernel chosen at run time | `Kernel::best`: SSE3 / AVX2 / AVX-512 **F+BW**, else scalar | his AVX-512 code uses `_mm512_slli_epi16` (BW), his "SSE2" code uses `_mm_lddqu_si128` (SSE3) | A | intrinsic list of the vendored files; `kernel_selection_tests` show which kernel ran | compile-time `cfg` like Parker | kept |
| S6 | `Kernel::pack` rejects values wider than the width | panic | his SIMD pack doesn't mask, so a too-wide value would corrupt its neighbor | A | `packing_a_value_wider_than_the_width_panics` | mask like the scalar code | kept (the codec never hits it: width = bit width of the max) |
| S7 | Header widths are read with a direct formula, not an unpack kernel | `header_width`: lane `slot % STRIDE`, field `slot / STRIDE` | decoding headers stays independent of the code that packed them | B → **A** | golden widths/offsets match his `Decompressor` for **every header slot** (G1 closed) | unpack headers with the width-4 kernel (Parker's `init`) | kept |
| S8 | Group bytes are copied into a stack word buffer before unpacking | `from_le_bytes` per word, ≤ 512 bytes per group | works for any alignment and byte order, no `unsafe` | B | `decoding_works_at_any_byte_alignment` | zero-copy `align_to::<u16>` view | kept; cost not measured, revisit in stage 9 (see section 7) |
| S9 | One-shot `group_range` is O(g) | sums the 4-bit fields before group g | same contract as Bp128 | B | — | O(1) per-metablock prefix table stored in the row (would change the format) | kept; use `GroupIndex` |
| S10 | Decode doesn't check the unused header slots of the last metablock | they are ignored | Parker doesn't check them either | B | pinned by `simdbp_ignores_padding_bits_and_unused_header_slots` | reject nonzero padding | kept (part of B2) |
| S11 | A header width of 9..=15 in a used slot panics | "invalid bit width" | `encode` never writes it | A | `invalid_width_in_header_panics` | — | kept |
| S12 | `SimdBp::BLOCK_SIZE` = its group length | hint only | the report no longer uses it | C | — | — | kept (see #6) |
| S13 | `Kernel` and `encode_with`/`decode_with` are public | tests can force the scalar or a SIMD kernel | to prove SIMD = scalar | B | — | test-only feature flag | kept |
| S14 | Oracle = separate nightly crate with a path dependency on Parker's repo | `tools/simdbp_oracle`, run by `regenerate.sh`; not part of `cargo test` | normal tests must not need his repo, nightly, or SIMD | A | `cargo test` passes without it | call his crate from our tests | kept |
| S15 | Golden inputs are regenerated from a spec in the test, not stored | each case stores `gen=...` and `input_fnv` | keeps the files small; the hash checks that our generated input equals the oracle's, separately from the output check | B | `check_golden` asserts the input hash before comparing outputs | store the inputs | kept |
| S16 | Size-report index RAM is computed (8 × (G+1)), not measured | formula | the registry has no typed `GroupIndex` | B | `index_memory_is_one_usize_per_group_plus_one` checks `heap_bytes()` >= the formula | add a registry method | kept |
| S17 | Large golden cases store a hash, not the bytes | outputs > 32 KiB: `words` (length) + `bytes_fnv` (FNV-1a 64); smaller ones: the full `bytes` too | keeps the golden files about 200 KB in total; every original case keeps exact bytes | B | all 23 cases at 128 and 21 of 23 at 256/512 have exact bytes | store everything | kept (decided in stage 5) |
| S18 | Test-only kernel trace | `kernel::trace` counts packs/unpacks per kernel, per thread, only under `cfg(test)` | prove which kernel executed without out-of-range tricks, at zero cost to normal builds | A | `kernel_selection_tests` | cargo feature; always-on counters | kept |

## 3. Audit findings (stage 5): behavior at the edges of the contract

| # | Finding | Decision | Status | Tests |
|---|---|---|---|---|
| **B1** | Raw, VByte, and SimdBp appended bytes before discovering a value > 255, so a caught panic left a partial row in the caller's buffer (Bp128 already checked first) | every encoder calls `check_values` before writing anything | **changed in stage 5**; byte format unchanged (golden files identical) | `every_codec_rejects_bad_values_before_writing_anything`, `registry_…`, `every_simdbp_kernel_…`, `valid_rows_still_append_after_a_rejected_row` (256, 300, 1000, 0x8000, 65535; first value, middle, end of a partial group, second SIMDBP-128 metablock; buffer already holding bytes) |
| **B2** | Decoders accept byte patterns our encoders never write: VByte over-long values (`[0x05, 0x80]` = 5, 2 bytes), nonzero bits in SIMDBP/Bp128 last-group padding, any value in unused SIMDBP header slots | **kept lenient** (like Parker's decoder); no canonical-form check at this stage | kept, documented, pinned | `vbyte_accepts_an_over_long_encoding`, `simdbp_ignores_padding_bits_and_unused_header_slots`, `bp128_ignores_padding_bits` |
| **B3** | The row length `n` is not stored, so decoding with a wrong `n` is not detected in general: SIMDBP with a larger `n` (same metablock) returns extra zeros with no panic; Raw and Bp128 read the next row's bytes as data | **the caller must supply the exact logical row length** (for real data: from the matrix metadata, `n = ceil(num_docs / bsize)` per level); the format is not changed to store `n` | kept, documented, pinned | `simdbp_with_a_too_large_n_silently_returns_zeros`, `raw_and_bp128_with_a_too_large_n_read_the_next_rows_bytes` |

Test gaps found by the audit, and how each was closed:

| # | Gap | Closed by |
|---|---|---|
| G1 | Header word 1 never held a nonzero width in any golden file; SimdBp512 covered 21/32 lanes and 2/4 field positions; random tests had at most 6 SimdBp512 groups | new oracle cases `all_header_fields` (3 metablocks, slot s of metablock m has width (s + 3m) % 9) and `random_group_widths`, compared with Parker's bytes and widths; `golden_cases_cover_every_header_field` asserts every slot is covered, in ≥ 3 metablocks; hand-computed bytes for header word 1, lane 27 of 32, and a second-metablock header |
| G2 | Nothing proved the SIMD kernels executed (only `Kernel::best` + a one-time objdump); a CPU without AVX-512 silently tested scalar only | test-only trace (S18); `SKIPPED` lines; `CODECS_REQUIRE_SIMD=1` makes missing SIMD a failure; a deliberately broken AVX-512 kernel made the tests fail (section 6) |
| G3 | Bp128 bytes pinned for one width-1 case only | independent layout model, widths 0..=8, partial chunks, every value at both ends of each lane |
| G4 | No regression test for the P1 sequence | `reused_index_never_returns_the_previous_rows_group` (all random-access codecs) |
| G5 | `GroupIndex` "shorter than the row" panic untested | `group_index_rejects_truncated_input` |
| G6 | Range check untested for 65535, partial last group, later metablock | B1 tests above |
| G7 | 256/512 metablock crossings only in one fixed row | `simdbp{128,256,512}_metablock_boundary` (GL² − 1, GL², GL² + 1 for every codec) + the new golden cases |
| G8 | No MS MARCO-scale test | `stress_ms_marco_sized_row` (8,841,823 values, every codec), opt-in: `cargo test --release --test boundaries -- --ignored` |

## 4. Compatibility with Parker's SIMDBP (`fa0ede5`)

### Confirmed against output of his code

(Golden vectors from his own `compress`, `Decompressor`, and `transcode`, built three
times: `sse` = BLOCK_SIZE 128, `avx2` = 256, `avx512` = 512; 23 cases each.)

- Our bytes are **identical** to his `compress` output, viewed as little-endian bytes,
  with the scalar kernel and with the SIMD kernel. Exact byte comparison for every case
  up to 32 KiB of output (all 23 at 128; 21 at 256 and 512); for the two largest
  cases at 256 and 512 (`all_header_fields`, 132,359 / 526,855 values;
  `random_group_widths`, 66,305 / 263,681 values) the same length and FNV-1a 64 hash.
  Cases: n = 1 (0 and 255), one full group at every width 0..=8, partial groups (GL/2+3,
  GL−1, GL+1), exact multiples (3 groups), all 255, 20 groups with widths 0..=8 cycling,
  an invented block-max-like mix, exactly one full metablock (GL²), a second metablock
  (GL² + 2·GL + 9), a third metablock holding one 1-value group (2·GL² + 1), every header
  slot nonzero across three metablocks (2·GL² + 5·GL + 7), and random per-group widths
  across a metablock boundary (GL² + 3·GL + 1).
- Our generated inputs equal the oracle's inputs (`input_fnv`, checked before outputs).
- Our decoder reads his bytes back to the original values, consuming exactly all of them.
- Our per-group widths equal his `Decompressor.offsets`, and our `GroupIndex` byte starts
  equal his `accumulated_offsets × 2 × STRIDE`, for every group of every case; together
  the cases put a nonzero width in every header slot (all 4 header words, all 4 field
  positions, all lanes) of at least three metablocks, at all three sizes.
- His scalar `transcode` to the other two group sizes gives the same bytes (length and
  FNV-1a hash) as our encoder for those sizes. (Same author as `compress`, so this is a
  second piece of his code, not a fully independent implementation.)
- His own decoder round-trips his output, and transcoding there and back gives his
  bytes again (recorded as `parker_roundtrip=ok`, `tc_back=ok`).
- Reproducible: regenerating the golden files in a throwaway copy gives byte-identical
  files (section 6).

What a hash comparison does and doesn't show: equal length + equal FNV-1a 64 means our
bytes and his agree **for that input**, with a negligible chance of a coincidence for
non-adversarial data. It is not a byte-by-byte diff, and no finite set of cases proves
compatibility for every input.

### Verified only within our implementation

- Rows at sizes and value patterns the golden cases don't cover (random rows, all
  kernels give the same bytes and decode back; metablock-boundary rows; an 8.8M-value
  row in the opt-in stress test).
- Random access (one-shot and `GroupIndex`) on random rows equals sequential decode.
- Scalar and SIMD kernels agree for every width and stride on random and extreme
  groups, and unpack arbitrary words identically.
- Concatenated rows, misaligned input, empty rows, error handling, B1–B3.

### Not verified / limitations

- His `decompress_block` path in a real search (no index here); P1's effect on search
  results.
- The non-x86 build (the `cfg(not(target_arch = "x86_64"))` branches compile only on
  other targets, which aren't installed here).
- CPUs without SSE3 / AVX2 / AVX-512 BW: `Kernel::best` falling back to scalar on such
  a CPU was not run (the scalar path is tested here by forcing `Kernel::Scalar`, and
  the tests print `SKIPPED` instead of silently passing).
- Speed: nothing was benchmarked (stage 9).

## 5. Issues found in Parker's code (his repo was not modified)

All reproduced with his **public API** by `tools/simdbp_oracle probe`; output in
`notes/simdbp_oracle_probes.txt`.

- **P1. `Decompressor::init` doesn't reset the cached group. REPRODUCED (API level).**
  - Conditions: one `Decompressor` (same `n`) decodes group g of row A; then
    `init(B)` is called for another row B; then group g is requested again.
  - Expected: row B's group g. Actual: row A's group g (`[0, 1, 2, 3]` instead of
    `[100, 101, 102, 103]`), in all three builds. Controls behave: a fresh decompressor,
    or asking for a different group first, return row B.
  - Minimal reproduction: `d = Decompressor::new(n); d.init(&a); d.decompress_block(&a, 0);
    d.init(&b); d.decompress_block(&b, 0)` returns `a`'s values.
  - Where it could matter: `search.rs:65-74` creates 1000 decompressors per level before
    the query loop and reuses them (slot j = query term j). `compute_upper_bounds_simdbp`
    (`live_block.rs:121-155`) and `unified_simdbp_for_safe` (`:1218-1340`) call `init`
    per query, then `decompress_block`. If a slot's first group in query q+1 equals the
    last group it decoded in query q, the upper bounds use query q's term.
    Level 0 starts at group 0 and ends at the last group, so it collides only when the
    level has a single group; level 1 collides when the first filtered group of q+1 equals
    the last one of q for the same slot.
  - NOT reproduced: any wrong search result. That needs his index and queries.
  - Our design avoids it (#16), and a regression test runs the same sequence.
- **P2. Empty rows panic.** `compress(vec![])` and `Decompressor::new(0)` + `init(&[])`
  panic (`n_blocks - 1` underflows). Confirmed. Our codec defines `n = 0` as 0 bytes (S3).
- **P3. Values above 255 are silently truncated.** A 300 decodes as 44 (`bits_needed`
  returns 8 for any value > 255). Confirmed. Our encoders reject them before writing
  anything (#2, B1).
- **P4. His kernels are safe `pub fn`s with unchecked raw-pointer loads/stores.**
  Too-short slices would be undefined behavior. Not a bug in his callers (they pass
  correct lengths); our wrappers check lengths before every call.
- **P5. `single_unpack256_6`, register offset 10, is wrong. REPRODUCED.**
  `decompress_register` (AVX2 builds only) returns a wrong register for width 6 at
  offset 10 and is correct for every other width/offset tested (widths 1..=8, all 16
  offsets). By code reading it uses `data4` where `data5` is needed. The only caller,
  `unified_simdbp`, is never called, so no search result is affected. We don't port it.

## 6. Measured results vs. hypotheses

**Measured (this machine, i7-11800H, rustc 1.99.1 for this crate, nightly-2025-11-18 for
Parker's crate):**

- Byte-identical output to Parker's code (section 4).
- Stage-5 audit run (all with `CODECS_REQUIRE_SIMD=1`): 136 tests pass in debug and in
  release, plus the opt-in stress test in release. The kernel trace shows the SSE3,
  AVX2, and AVX-512 kernels executed, both when forced and through the default paths.
- Reproducibility: `regenerate.sh` run in a throwaway copy of the repo (Parker's repo
  checked clean before and after) wrote golden files and a probe log byte-identical to
  the committed ones.
- Mutation checks (in throwaway copies, never committed):
  - storing SIMDBP words big-endian (self-consistent, not Parker's format) passed every
    round-trip / random-access / end-to-end test and failed the golden, hand-computed,
    and header tests (earlier in stage 5);
  - breaking one vendored AVX-512 kernel (`pack512_3`, one shift `15 → 16`) made 21
    tests fail, including all three golden tests, the kernel agreement tests, and the
    forced-AVX-512 trace test, so the tests reach the SIMD code;
  - putting back one old encoder at a time (before B1) with the new tests: the B1 tests
    fail for Raw, VByte, and SimdBp, and pass for Bp128 (it already checked before
    writing), as expected.
- Vendored kernels: re-running `scripts/vendor_simdbp_kernels.py` against Parker's repo
  reproduces `src/simdbp/kernels/*.rs` exactly.
- Synthetic size report (one row per dataset; numbers in `notes/code_map.md`).

**Hypotheses (not measured on real data):**

- On block-max-like data, SimdBp512 is close to 1 byte/value because almost every
  512-value group contains an outlier (1 − 0.99^512 ≈ 99.4% for a 1% outlier rate). The
  invented 90/9/1 dataset gives 0.993, consistent with that, but real block maxes may
  have a very different outlier distribution. Check on real data (stage 8).
- Smaller groups (SimdBp128 0.862 vs SimdBp512 0.993 on the same invented data) trade
  compression for more metadata and more groups per lookup; the speed side is unknown.
- Copying group bytes before unpacking (S8) is cheap compared with unpacking; unmeasured.

## 7. Research-validity caveats

- **Binary compatibility is not performance equivalence.** Our SIMDBP writes Parker's
  bytes, but the decode path differs: our kernel is chosen at run time and called as a
  non-inlined `#[target_feature]` function through a `match` on the width, and each
  group's bytes are first copied into a stack buffer (S8). His is selected at compile
  time, inlined, and reads 64-byte-aligned `Vec<u16>` data in place
  (`align_block_data`, `posting_list.rs:58-101`). His `decompress_block` also returns
  the full zero-padded group (#18). So stage-9 speed numbers for our `SimdBp*` are not
  his numbers; compare like with like (e.g. time his decoder too, or document the gap).
- **Alignment assumptions.** The vendored kernels use only unaligned loads/stores
  (`_mm_lddqu_si128`, `_mm256_lddqu_si256`/`loadu`, `_mm512_loadu_si512`, `storeu`;
  checked by listing the intrinsics), so our 2-byte-aligned stack buffers are valid.
  A future zero-copy decode (S8) must keep using unaligned loads or guarantee alignment.
  Parker's `aligned_vec_u16` frees 64-byte-aligned memory as a 2-byte-aligned `Vec`
  (`notes/code_map.md`, section 1), which is UB by Rust's rules; not our code.
- **Index RAM.** Ours: 8 B × (G+1) per open row. His `Decompressor`: about 2 B per group
  for `offsets` (rounded up to a multiple of BLOCK_SIZE, plus BLOCK_SIZE) + 8 B per group
  for `accumulated_offsets` + a 1 KB group buffer. Neither is counted in encoded sizes;
  his "Block Scores Size" counts 2 × u16 words, which equals our byte count.
- **Synthetic success is not compression evidence.** Every size number so far is on
  synthetic data; it checks the codecs, not their value for real block maxes.
- **Self-consistency is not format verification.** Only the golden vectors, the
  hand-computed bytes, the Bp128 layout model, and the independent size formulas
  check the formats; round trips only check that encoder and decoder agree.

## 8. Concerns for real-data loading (stage 8, from reading Parker's code; not run)

- **Row length.** Per level, `n = ceil(num_docs / bsize)` for every term; the loader must
  pass exactly that (B3).
- **Group size of a stored index.** An index can be re-encoded to another SIMDBP group
  size when it is written (`target_simd_block_size` → `transcode`, `simd_block_size`
  field, `src/ciff/mod.rs` around lines 395-418). Read the group size from the index,
  don't assume 512 from the build.
- **ASC builds.** With the `asc_ablation` feature, some levels stored as
  `BlockData::SIMDBP` hold raw f16 bits, not SIMDBP data, and their values can exceed
  255 (`inverted_index.rs:223`, `:475`). Never decode those as SIMDBP.
- **Value provenance.** Block maxes are `(score as u8) >> quant` (`inverted_index.rs:151-156`).
  `score as u8` keeps only the low 8 bits of a u32 score, so a score above 255 would
  silently wrap; check the maximum CIFF score on real data before trusting the maxes.
  How `quant == 8` behaves (`u8 >> 8`) is question 2 in `notes/code_map.md`; settle it
  before using a quant-8 index.
- **Errors.** Real files need `Result` errors instead of panics (#17).
- `export_data` only works on a Raw index; real SIMDBP rows would be re-encoded by our
  encoder (byte-identical to his, section 4).
