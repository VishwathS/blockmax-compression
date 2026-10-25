# Design decisions

Every design decision in this crate that was found in the stage-6 review (SIMDBP
integration, branch `feature/simdbp-integration`), including the arbitrary and
unvalidated ones, with what happened to each. Update this file when a decision changes.

**Classes**

- **A**: technically justified (there is a concrete reason or evidence).
- **B**: reasonable but not validated (no measurement or test says it's the best choice).
- **C**: arbitrary or legacy (no known reason).
- **D**: potentially problematic (can give wrong results or mislead).

**Status**: *kept* (unchanged), *changed in stage 6* (with the reason), or *deferred*
(left as is on purpose; the stage that should revisit it is named).

"Rationale: unknown" means nobody recorded a reason; none has been invented here.

## 1. Decisions that existed before stage 6

| # | Decision | Behavior | Rationale | Class | Evidence | Alternatives | Status |
|---|---|---|---|---|---|---|---|
| 1 | `u16` values, each <= 255 | `encode(&[u16])`, `decode(.., &mut [u16])` | Parker's query loop and SIMDBP work on u16 (room to multiply by a query weight) | A | `compress(Vec<u16>)`, `inverted_index.rs:300` widens u8 to u16 | u8 interface | kept |
| 2 | Range check with `assert!` in every build | values > 255 panic in debug **and** release | never store a wrong value silently | A | Parker's SIMDBP stores 300 as 44 (P3 below, reproduced) | `debug_assert!` | kept; also applied to the SIMDBP codecs |
| 3 | VByte decode did not range-check | `[0x7F, 0xFF]` decoded to 16383 without a panic, against the trait contract | none (an oversight) | D | code reading, then a test | — | **changed in stage 6**: decode panics on a value > 255 (`vbyte_panics_on_a_two_byte_value_above_255`); valid bytes decode exactly as before |
| 4 | `n` is not stored | decoder gets it from `out.len()`, random access from an `n` argument | every term of a level has the same `n`, like `Decompressor::new(n)` | A | `lib.rs` docs | length prefix | kept. A wrong `n` gives garbage or a panic, never UB |
| 5 | Each `encode` call is a self-contained section | sections can be appended and decoded in order | convenient for tests and multi-row buffers | B | tests only | — | kept; SIMDBP rows follow it too |
| 6 | `BLOCK_SIZE` per codec (128 for Raw/VByte/Bp128) | "preferred values per call", a hint | Raw/VByte: unknown; Bp128: its chunk size | C (D for the old size report) | the old size report encoded every `block_size()` values as a separate row, which charged SIMDBP one header per 128 values | remove it; make it per-row | constant **kept** (SimdBp sets it to its group length, documented as a hint only); the size report **changed** (#21b) |
| 7 | Bp128 uses 128-value chunks | fixed | `BitPacker4x::BLOCK_LEN` is 128: chosen with the library, not measured | B | `bp128.rs` | 32 (BitPacker1x), 256/512 | deferred to stage 8 / real data |
| 8 | Bp128 stores 1 byte per width | 0.0078 B/value of metadata | simplicity (4 bits would save 0.004 B/value) | B | size report: all zeros 0.008 vs SimdBp128 0.004 | 4-bit packed widths | deferred (real data first) |
| 9 | All widths before all data (Bp128) | `[widths][data]` | needed for random access; mirrors SIMDBP | A | `bp128.rs`, random-access tests | interleaved | kept |
| 10 | Bp128 uses the `bitpacking` crate's 4-lane layout | crate-defined bytes | unknown (the crate was available) | B | debug = scalar `_simple`, release = SSE3 chosen at run time (read in the crate source); `bp128_known_bytes` pins the bytes in both | own packer | kept |
| 11 | The last group is zero-padded | up to `(GL−1) × w` padding bits per row | A for Bp128 (the crate needs 128 values) and for SIMDBP (part of Parker's format); the size cost is B | A/B | `partial_group_padding_is_zero_and_not_written_out` | separate tail encoding | kept; cost matters only for many tiny rows |
| 12 | Raw `GROUP_LEN` = 128 | 128-value groups | comment: "any size would work" | C | `raw.rs` | any | kept (harmless) |
| 13 | Random access is a separate trait | `RandomAccessCodec: BlockCodec`; VByte doesn't implement it | keeps `BlockCodec` and every format unchanged | A | SIMDBP fit without any trait change | `Option`-returning methods on `BlockCodec` | kept |
| 14 | `GroupIndex` stores `usize` byte offsets | 8 B × (G+1) RAM | simplicity | B | size report table 2 | `u32` offsets (half the RAM) | deferred to stage 8 |
| 15 | A group's width is recovered from its byte length | `decode_group_bytes(data, out)` | implicit in the trait | B | Bp128: `len / 16`; SimdBp: `len / (2 × STRIDE)` | pass the width in | kept; would not work for a codec whose group size isn't a function of one width (e.g. PFor) |
| 16 | No cached group inside the index | the caller owns the output buffer | caller can skip re-decoding; no hidden state | **A (strengthened)** | Parker's `Decompressor` caches and returns stale data after `init` (P1, reproduced) | cache like Parker's | kept |
| 17 | Errors are panics, not `Result` | every invalid input panics with a message | encoding is offline, data is trusted; Parker panics too | B (D risk at stage 5) | — | `Result` | deferred to stage 5 (files read from disk) |
| 18 | `decode_group` doesn't touch `out[len..]` | no zero padding past `n` | documented contract | B | tests with sentinel buffers | zero-fill | kept; Parker's query loops read a full zero-padded 512 buffer, so revisit when wiring codecs into a query loop |
| 19 | Registry = enum + `match` | one variant per codec, `Codec::ALL` | exhaustiveness, stable names | A (dispatch cost B) | 6 codecs added without trait changes | trait objects | kept; time dispatch in stage 8 |
| 20 | Registry has only the one-shot `decode_group` | O(g) per call through the registry | none recorded | B | — | an indexed registry API | deferred (not needed yet) |
| 21a | Size-report datasets | uniform, 4-bit, zeros, "90/9/1" mix | placeholders until real data; the 90/9/1 proportions are invented | C | `examples/size_report.rs` | real `export_data` matrices | kept, now **labelled synthetic** in the output and docs; replaced at stage 5 |
| 21b | Size report: one row per `block_size()` values | see #6 | unknown | D | old output: SIMDBP all zeros = 0.500 B/value | one row per dataset | **changed in stage 6**: each dataset is one row. Raw/VByte/Bp128 numbers are identical to before (checked: same output; `splitting_into_rows_changes_size_only_for_codecs_with_row_headers`) |
| 21c | Size report counted only encoded bytes | index RAM invisible | — | B | — | add a column | **changed in stage 6**: a second table shows `GroupIndex` RAM per value, kept separate from the encoded size |
| 22 | Test independence | most tests were round trips or compared with the same codec's `decode` | — | B | only `bp128_known_bytes`, VByte bytes, and hand-computed Bp128 offsets were independent | golden vectors | **changed in stage 6**: Parker-generated golden vectors, hand-computed SIMDBP bytes, independently computed sizes (`tests/end_to_end.rs`); see section 5 |

## 2. New decisions in stage 6 (SIMDBP)

| # | Decision | Behavior | Rationale | Class | Evidence | Alternatives | Status |
|---|---|---|---|---|---|---|---|
| S1 | Three codecs, one per group size | `SimdBp<8/16/32>` = `SimdBp128/256/512`; other strides don't compile | in Parker's code the group size depends on compile flags; here the bytes never do | A | golden files from his 3 builds all match | one codec chosen by build flags (his way) | kept |
| S2 | u16 words stored as little-endian bytes | trait works on `&[u8]` | equals his `Vec<u16>` in x86 memory | A | golden: byte-identical | native-endian view (`align_to`) | kept |
| S3 | `n = 0` encodes to 0 bytes | empty rows work | the trait allows empty input; Parker's `compress` panics (P2) | B (our extension) | `empty_row_is_zero_bytes` | panic like Parker | kept; his code can't produce or read such a row, so there's no compatibility case to break |
| S4 | Vendored Parker's SIMD kernels, widths 0..=8 only | `scripts/vendor_simdbp_kernels.py` copies the function bodies unchanged and makes each an `unsafe fn` with `#[target_feature]` | values are <= 255, so widths 9..=16 are dead code; mechanical copy = auditable | B | proof of concept + `kernels_agree_*` tests | hand-written intrinsics; scalar only | kept |
| S5 | Kernel chosen at run time | `Kernel::best`: SSE3 / AVX2 / AVX-512 **F+BW**, else scalar | his AVX-512 code uses `_mm512_slli_epi16` (BW), his "SSE2" code uses `_mm_lddqu_si128` (SSE3) | A | intrinsic list of the vendored files | compile-time `cfg` like Parker | kept |
| S6 | `Kernel::pack` rejects values wider than the width | panic | his SIMD pack doesn't mask, so a too-wide value would corrupt its neighbor | A | `packing_a_value_wider_than_the_width_panics` | mask like the scalar code | kept (the codec never hits it: width = bit width of the max) |
| S7 | Header widths are read with a direct formula, not an unpack kernel | `header_width`: lane `slot % STRIDE`, field `slot / STRIDE` | decoding headers stays independent of the code that packed them | B | golden widths/offsets match his `Decompressor` | unpack headers with the width-4 kernel (Parker's `init`) | kept |
| S8 | Group bytes are copied into a stack word buffer before unpacking | `from_le_bytes` per word, ≤ 512 bytes per group | works for any alignment and byte order, no `unsafe` | B | `decoding_works_at_any_byte_alignment` | zero-copy `align_to::<u16>` view (the approved plan's first idea) | kept; cost not measured, revisit in stage 8 |
| S9 | One-shot `group_range` is O(g) | sums the 4-bit fields before group g | same contract as Bp128 | B | — | O(1) per-metablock prefix table stored in the row (would change the format) | kept; use `GroupIndex` |
| S10 | Decode doesn't check the unused header slots of the last metablock | they are ignored | Parker doesn't check them either | B | — | reject nonzero padding | kept |
| S11 | A header width of 9..=15 panics | "invalid bit width" | `encode` never writes it | A | `invalid_width_in_header_panics` | — | kept |
| S12 | `SimdBp::BLOCK_SIZE` = its group length | hint only | the report no longer uses it | C | — | — | kept (see #6) |
| S13 | `Kernel` and `encode_with`/`decode_with` are public | tests can force the scalar or a SIMD kernel | to prove SIMD = scalar | B | — | test-only feature flag | kept |
| S14 | Oracle = separate nightly crate with a path dependency on Parker's repo | `tools/simdbp_oracle`, run by `regenerate.sh`; not part of `cargo test` | normal tests must not need his repo, nightly, or SIMD | A | `cargo test` passes without it | call his crate from our tests | kept |
| S15 | Golden inputs are regenerated from a spec in the test, not stored | each case stores `gen=...` and `input_fnv` | keeps the files small (126 KB total); the hash catches generator drift | B | `check_golden` asserts the hash | store the inputs | kept |
| S16 | Size-report index RAM is computed (8 × (G+1)), not measured | formula | the registry has no typed `GroupIndex` | B | `index_memory_is_one_usize_per_group_plus_one` checks `heap_bytes()` >= the formula | add a registry method | kept |

## 3. Compatibility with Parker's SIMDBP (`fa0ede5`)

### Confirmed against output of his code

(Golden vectors from his own `compress`, `Decompressor`, and `transcode`, built three
times: `sse` = BLOCK_SIZE 128, `avx2` = 256, `avx512` = 512; 21 cases each.)

- Our bytes are **identical** to his `compress` output, viewed as little-endian bytes,
  for every case at every group size, with the scalar kernel and with the SIMD kernel.
  Cases: n = 1 (0 and 255), one full group at every width 0..=8, partial groups (GL/2+3,
  GL−1, GL+1), exact multiples (3 groups), all 255, 20 groups with widths 0..=8 cycling,
  an invented block-max-like mix, exactly one full metablock (GL² values), a second
  metablock (GL² + 2·GL + 9), and a third metablock holding one 1-value group
  (2·GL² + 1; for 512 that's 524,289 values).
- Our decoder reads his bytes back to the original values, consuming exactly all of them.
- Our per-group widths equal his `Decompressor.offsets`, and our `GroupIndex` byte starts
  equal his `accumulated_offsets × 2 × STRIDE`, for every group of every case.
- His scalar `transcode` to the other two group sizes gives the same bytes (length and
  FNV-1a hash) as our encoder for those sizes: a second, independent piece of his code.
- His own decoder round-trips his output, and transcoding there and back gives his
  bytes again (recorded as `parker_roundtrip=ok`, `tc_back=ok`).

### Verified only within our implementation

- Rows at sizes and value patterns the golden cases don't cover (proptest: random rows
  up to 3,000 values, all kernels give the same bytes and decode back).
- Random access (one-shot and `GroupIndex`) on random rows equals sequential decode.
- Scalar and SIMD kernels agree for every width and stride on random and extreme
  groups, and unpack arbitrary words identically.
- Concatenated rows, misaligned input, empty rows, error handling.

### Not verified / limitations

- His `decompress_block` path in a real search (no index here); P1's effect on search
  results.
- The non-x86 build (the `cfg(not(target_arch = "x86_64"))` branches compile only on
  other targets, which aren't installed here).
- CPUs without AVX-512 BW running `SimdBp512` (the scalar fallback is tested on this
  machine by forcing `Kernel::Scalar`, but `Kernel::best` falling back on its own was not
  run on such a CPU).
- Speed: nothing was benchmarked (out of scope for stage 6).

## 4. Issues found in Parker's code (his repo was not modified)

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
  - Our design avoids it (#16: no cache inside the index).
- **P2. Empty rows panic.** `compress(vec![])` and `Decompressor::new(0)` + `init(&[])`
  panic (`n_blocks - 1` underflows). Confirmed. Our codec defines `n = 0` as 0 bytes (S3).
- **P3. Values above 255 are silently truncated.** A 300 decodes as 44 (`bits_needed`
  returns 8 for any value > 255). Confirmed. Our encoders reject them (#2).
- **P4. His kernels are safe `pub fn`s with unchecked raw-pointer loads/stores.**
  Too-short slices would be undefined behavior. Not a bug in his callers (they pass
  correct lengths); our wrappers check lengths before every call.
- **P5. `single_unpack256_6`, register offset 10, is wrong. REPRODUCED.**
  `decompress_register` (AVX2 builds only) returns a wrong register for width 6 at
  offset 10 and is correct for every other width/offset tested (widths 1..=8, all 16
  offsets). By code reading it uses `data4` where `data5` is needed. The only caller,
  `unified_simdbp`, is never called, so no search result is affected. We don't port it.

## 5. Measured results vs. hypotheses

**Measured (this machine, i7-11800H, rustc 1.99.1 for this crate, nightly-2025-11-18 for
Parker's crate):**

- Byte-identical output to Parker's code (section 3).
- All tests pass in debug and release builds (106 each), with `CODECS_REQUIRE_SIMD=1`
  so the SSE3, AVX2, and AVX-512 kernels were all required and used; their instructions
  (`zmm`, `ymm`, `lddqu`) are present in both test binaries (objdump).
- Mutation check: storing SIMDBP words big-endian (self-consistent but not Parker's
  format) still passed all 75 round-trip / random-access / end-to-end tests, and failed
  the 3 golden tests, 3 hand-computed byte tests, and 1 header test. So the golden tests
  catch format errors that round trips cannot.
- Synthetic size report (one row per dataset; numbers in `notes/code_map.md`).

**Hypotheses (not measured on real data):**

- On block-max-like data, SimdBp512 is close to 1 byte/value because almost every
  512-value group contains an outlier (1 − 0.99^512 ≈ 99.4% for a 1% outlier rate). The
  invented 90/9/1 dataset gives 0.993, consistent with that, but real block maxes may
  have a very different outlier distribution. Check on real data (stage 5).
- Smaller groups (SimdBp128 0.862 vs SimdBp512 0.993 on the same invented data) trade
  compression for more metadata and more groups per lookup; the speed side is unknown.
- Copying group bytes before unpacking (S8) is cheap compared with unpacking; unmeasured.
