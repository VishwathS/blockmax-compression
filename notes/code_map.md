# Code map: how block maxes are stored and read in hierarchical_pruning

Repo commit `fa0ede5`. All paths are relative to `~/hierarchical_pruning`.
Throughout, "AVX-512 build" means compiled with `target-cpu=native` on our i7-11800H.

Running example (from the README, `README.md:42-44`):
`ciff2bmp --bsize 128 8 --compress-range simdbp --quant 4`, MS MARCO with 8,841,823 docs.

| level | what | entries per term |
|---|---|---|
| 0 | superblock maxes (128 docs) | ceil(8,841,823/128) = 69,077 |
| 1 | block maxes (8 docs) | ceil(8,841,823/8) = 1,105,228 |

Each superblock holds `stride = bsize[0]/bsize[1] = 16` blocks (`src/search.rs:1333`).

---

## 1. `src/index/posting_list.rs`: the storage types

- **`BlockData`** (`posting_list.rs:9-13`): one term's maxes at one level. It is one of:
  - `Raw(Vec<u8>)`: dense, one byte per block, including zeros.
  - `Compressed(Vec<CompressedBlock>)`: the "Superblock" format. Each
    `CompressedBlock` (`:22-24`) holds `Vec<(usize, u8)>` (offset, max) pairs for the
    **non-zero** entries of a 256-entry chunk (built in `inverted_index.rs:282-298`).
    Note: `(usize, u8)` takes **16 bytes** in memory because of padding (8 + 1 rounded
    up to 8-byte alignment). On disk, bincode writes the usize as 8 bytes, so 9 bytes
    per non-zero.
  - `SIMDBP(Vec<u16>)`: the bit-packed format described in section 3.
  - The default is an empty `Raw` (`:15-19`).
- **`PostingList`** (`:27-30`):
  - `range_maxes: Vec<BlockData>`: one entry per level, in the order of `bsize`. The
    builder sorts bsize largest-first (`src/ciff/mod.rs:444-447`), so `[0]` = superblocks
    and `[1]` = blocks.
  - `kth_score: Vec<u8>`: the term's 10th/100th/1000th-largest impact plus a padding 0
    (`inverted_index.rs:164-170`). Used to seed the search threshold (`search.rs:181-207`).
  - No doc IDs or postings are stored here. This struct only holds upper bounds.
- **`align_block_data`** (`:58-101`) copies every `Raw`/`SIMDBP` vector into a
  64-byte-aligned buffer, so SIMD loads line up with cache lines. `Compressed` is skipped.
  - Possible issue: `aligned_vec_u16` (`:122-140`) allocates with 64-byte alignment
    but turns the memory into a `Vec<u16>`, which later frees it as if it were 2-byte
    aligned. Rust's rules call that undefined behavior. It works in practice with
    mimalloc. Ask Parker before relying on it.
- **`range_max_scores_iter` / `range_max_scores`** (`:199-235`) wrap each level in a
  borrowed `RangeMaxScore` enum. That is how the search code gets `&[u8]` / `&[u16]` /
  `&[CompressedBlock]` slices.

## 2. `src/index/inverted_index.rs`: computing block maxes (`insert_term_streaming`, `:138-172`)

For one term (`postings` = list of `(docid, score)`):
1. Loop over levels (`for b in self.bsize`, `:146`). `blocks_num = ceil(num_docs / b)`
   (`:148`). Every term has the same length at a level, so the length never needs to be
   stored per term.
2. Start a zeroed `Vec<u8>` (`:149`). For each posting, block index = `docid / b`, and
   keep the max (`:150-157`).
3. **Quant shift** (`:156`): `score as u8 >> self.quant`. Rust parses this as
   `(score as u8) >> quant`, because `as` binds tighter than `>>`. So:
   - `score as u8` keeps only the low 8 bits of the u32. The README (`README.md:35`)
     requires CIFF impacts to already be 8-bit, so nothing should be lost.
   - With `--quant 4`, values become 0..=15, which fits in 4 bits.
   - `quant` defaults to 8 (`bin/ciff2bmp.rs:24`, `inverted_index.rs:63`). There is a
     special branch for `quant == 8` (`:151-154`). See question 2 at the end.
   - The max is taken over shifted values. Because `>>` preserves order, this equals
     shifting the true max.
4. Encode the level (`:158-162`): `Superblock` → `compress` (`:282`), `SIMDBP` →
   `compress_simdbp` (`:300-303`, which widens u8 to u16 and then calls section 3's
   `compress`), `Raw` → store the `Vec<u8>` as is.
5. The query side shifts the threshold the same way so the two compare on the same
   scale: `search.rs:187-207` (initial threshold uses `kth >> quant`) and
   `search.rs:352-363` (early stopping uses `topk.threshold() >> quant`). Both have a
   `quant >= 8` branch that skips the shift.
- `:174-240` is the `asc_ablation` variant (feature flag). It also stores f16 per-segment
  averages as an extra `BlockData` per level. It's not the main path.
- `build_streamed` (`:244-269`) prints "Size of posting list" counting only Raw
  (1 B/entry) and SIMDBP (2 B/u16). `Compressed` is counted as 0.

## 3. `src/index/compress_simdbp.rs`: the SIMDBP format

### Constants chosen at compile time (`:157-199`)
| build | `BLOCK_SIZE` (values per group) | `STRIDE` (u16s per register) |
|---|---|---|
| AVX-512 (**ours**) | 512 | 32 |
| AVX2 | 256 | 16 |
| SSE2 | 128 | 8 |

`BLOCK_SIZE = 16 × STRIDE` in every case, and `METABLOCK_HEADER = BLOCK_SIZE/4` u16s
(`:203`). Naming warning: Parker's code calls a group a "block", which clashes with
"block" as in block max. These notes say **group** for a SIMDBP chunk.

### Layout produced by `compress` (`:205-321`)
```
[ header 0 | header 1 | ... | header M-1 ][ group 0 data ][ group 1 data ] ... [ group G-1 data ]
  each header = BLOCK_SIZE/4 u16s           each group = STRIDE × width_g u16s (0 if width 0)
```
- G = `n_blocks = ceil(n / BLOCK_SIZE)` groups. M = `n_metablocks = ceil(G / BLOCK_SIZE)`.
  So one **metablock** = BLOCK_SIZE groups, and its header stores those groups' widths.
- **All headers come first**, before any data (`write_i` starts at
  `n_metablocks * METABLOCK_HEADER`, `:213`). The scalar `transcode` writes the same
  layout (`:133-151`).
- For each group: `width = maxbits(group)` (`:215`), 0..=8 bits (`bits_needed`,
  `:489-501`). Width 0 writes **nothing**, so all-zero groups are free apart from 4 header
  bits. Otherwise `pack_<width>` writes exactly `STRIDE * width` u16s (`:217-253`).
- **Headers** (`:302-317`): the per-group widths are themselves bit-packed with
  `pack_4`, the same SIMD packer at width 4. BLOCK_SIZE widths × 4 bits = BLOCK_SIZE/4 u16s.
- `output.resize(write_i, 0)` (`:319`) trims the buffer to exactly what was written.
  `pad_amount` (`:207`) only over-allocates a scratch buffer beforehand. It equals a full
  BLOCK_SIZE when `n` is already a multiple of BLOCK_SIZE, which is harmless.

### Last partial group and last metablock (padding)
- **Last group** (`:256-300`): copied into a zeroed `BLOCK_SIZE` buffer
  (`padded_input`), then packed like the others. Zeros never raise the max, so the
  padding doesn't change the width. It does cost `width × (BLOCK_SIZE − leftover)` bits
  of stored zeros.
- **Last metablock header** (`:309-317`): the widths for groups that don't exist are
  zero-padded to BLOCK_SIZE before `pack_4`, so every header is full size.
- The format doesn't store `n`. The reader must already know it, which is why
  `Decompressor::new(n)` takes it (`search.rs:69,73`).

### Layout inside a group (`:5-35`, `pack_scalar`)
- There are STRIDE "lanes". Lane `i` holds the 16 elements `i, i+STRIDE, i+2·STRIDE, …,
  i+15·STRIDE`, packed one after another at `width` bits each, spread over `width`
  u16 words. Word `w` of lane `i` sits at `output[w*STRIDE + i]`.
- So one SIMD load (STRIDE consecutive u16s) grabs word `w` of **every lane at once**,
  and SIMD shifts/masks unpack 16 output registers. Which elements end up in which
  register is question 1.
- The comments at `:8` and `:11-15` describe this layout differently (also question 1).

### `Decompressor` (`:323-483`): how random access finds group i
- State (`:324-330`): `n`, `offsets` (one width per group, unpacked from the headers),
  `accumulated_offsets` (prefix sum), a `decompressed_buffer` of BLOCK_SIZE u16s, and
  `current_block` (which group is cached).
- **`init(compressed)`** (`:348-366`), run once per term per query:
  1. Unpack every metablock header with `unpack_4` into `offsets` (`:351-357`).
  2. `accumulated_offsets[0] = n_metablocks * 4` (`:361`). This counts in units of STRIDE
     u16s, and each header is `BLOCK_SIZE/4 = 4·STRIDE` u16s.
  3. `accumulated_offsets[i+1] = accumulated_offsets[i] + width[i]` (`:362-365`). This
     works because group i occupies `STRIDE × width[i]` u16s, i.e. `width[i]` units.
- **Finding group i afterwards** is O(1): data starts at
  `accumulated_offsets[i] * STRIDE` u16s, length `STRIDE × offsets[i]`, and the width is
  `offsets[i]`. The 4-bit headers make the widths compact. But "random access" here means
  "after an O(G) init, then O(1) per group", not "decode the header on demand". How
  much work that init is for a real query is question 3.
- **`decompress_block(compressed, idx)`** (`:432-482`): group = `idx / BLOCK_SIZE`. If
  that's the cached group, it returns the buffer immediately (`:435-437`). Otherwise it
  unpacks the **whole group** (BLOCK_SIZE values) into the buffer. Returns `&[u16]` of
  length BLOCK_SIZE, and the caller indexes it with `idx % BLOCK_SIZE`.
- **`decompress_register(compressed, idx)`** (`:380-430`): **AVX2 build only**
  (`#[cfg(all(avx2, not(avx512f)))]`, `:382`). Unpacks just the one 16-value register
  that holds `idx` (`stride_offset = (idx % 256) / 16`, `:386`) and returns a `__m256i`.
  **It doesn't exist in our AVX-512 build.**
- **`prefetch(compressed, idx)`** (`:368-378`): prefetches the first cache line of the
  group containing `idx`. A group with width `b` spans `b` cache lines on AVX-512
  (32·b u16 = 64·b bytes), so only the first one gets prefetched.

## 4. `src/query/live_block.rs`: call sites and access granularity

Search keeps 1000 preallocated `Decompressor`s per level (`search.rs:65-74`), one per
query term, created with the level's length. Nothing is allocated per query.

| call site | function | used by | what it decodes | granularity needed |
|---|---|---|---|---|
| `live_block.rs:133` | `init` | `compute_upper_bounds_simdbp` | all headers | whole term |
| `live_block.rs:136` | `decompress_block` | same (level 0, `search.rs:1272`) | every group in order | **full sequential scan**: all 69,077 superblock maxes, BLOCK_SIZE at a time |
| `live_block.rs:150` | `decompress_block` | same, last group | last group | tail of the vector, scalar loop `:151-154` |
| `live_block.rs:1232` | `init` | `unified_simdbp_for_safe` | all headers | whole term |
| `live_block.rs:1258` | `prefetch` | same | first line of a group 2 filter entries ahead | group |
| `live_block.rs:1261` | `decompress_block` | same (level 1, `search.rs:1347`) | the group containing superblock `fi` | **needs 16/32/64 consecutive block maxes** (`stride` = 16 in the example), but decodes the whole 512-value group. The filter is sorted (`search.rs:1329`), so neighbors reuse the cached group. One group covers 512/16 = 32 superblocks. |
| `live_block.rs:1334` | `decompress_block` | same, leftover tail | per index past the last full stride | single values, `range[idx % BLOCK_SIZE]` (`:1336`) |
| `live_block.rs:701-725` | `new`/`init`/`decompress_block` | `compute_asc_simdbp_intrinsic` (asc ablation, `search.rs:1127`) | steps of 32 values, a **fresh** Decompressor per term (twice: main loop + leftovers) | 32 consecutive |
| `live_block.rs:1071` | `init` | `unified_simdbp` | all headers | whole term |
| `live_block.rs:1091-1093` | `prefetch` + `decompress_register` | same, leftover tail | 16 values per call | one register (16) |
| `live_block.rs:1115-1150` | `prefetch` + `decompress_register` ×1/2/4 | same, main loop | 1, 2 or 4 registers for stride 16/32/64 | **exactly** the 16/32/64 values needed |

Notes:
- `unified_simdbp` (`:1045`), the only user of `decompress_register`, is AVX2-only
  **and isn't called anywhere** (a grep of all `.rs` files finds only its definition).
  So in practice every SIMDBP query, on any machine, goes through `decompress_block`.
  The comment at `:1217` ("decodes more data but faster for safe search") suggests the
  register version was tried and the full-group version won.
- Hot path summary for the default (non-ablation) search, `repeated_hierarchical_boundsum`
  (`search.rs:1234`):
  - Level 0: init plus decode **everything**.
  - Pick the superblocks above the threshold into a sorted `filter` (`search.rs:1283-1330`).
  - Level 1: init, then decode only the groups that hold filtered superblocks, using
    16 values out of each 512 decoded unless several filtered superblocks share a group.

---

## (a) What a generic codec trait must support to replace `BlockData`

A drop-in replacement for the SIMDBP variant would need:
1. **Values:** small unsigned ints, at most 8 bits (`0..=255 >> quant`). The query code
   wants them as `u16` so it can multiply by a weight without overflow
   (`live_block.rs:115,152`).
2. **Length known from outside:** `n` is the same for every term at a level, so the
   codec need not store it. Decoding takes `n` as a parameter.
3. **Encode once, offline:** `encode(&[u8 or u16]) -> bytes`, plus serde serialization
   and a size-in-bytes report.
4. **Two-phase decode with a reusable cursor:** a per-term, per-query `init`/`open` that
   builds whatever index it needs (like the prefix sum). It must work with preallocated
   state, since the hot path allocates nothing.
5. **Sequential full decode** (level 0): fill `&mut [u16]` of length `n`, or iterate
   group by group.
6. **Random-access decode of an aligned chunk** (level 1): "give me the values for
   positions `[k·s, k·s + s)`" for s = 16/32/64. Calls arrive in **increasing** order,
   so a codec may cache or decode forward. The current contract returns a whole group
   plus offset `idx % GROUP`, so the trait should expose the group size or return a slice
   already offset to `idx`.
7. **Zero padding past `n`**, so SIMD loops can read full registers. Optionally, 64-byte
   alignment (`align_block_data`).
8. **Optional `prefetch(idx)`.**
9. **Cheap all-zero regions:** most block maxes of most terms are 0, and SIMDBP spends
   0 data bits on them.
10. Nice to have: expose a group's width or max. A group of width `b` is bounded by
    `2^b − 1` without decoding, which might let the search skip groups. This is a design
    idea, not something the current code does.

The current `BlockCodec` trait (u16 values <= 255, encode/decode a whole block, no
cursor, no random access) covers only items 1, 3 and 5. Items 4 and 6 are what the
later random-access stage needs to add. `Bp128` (section (c)) also covers item 9: an
all-zero 128-value chunk costs one width byte and no data.

## (c) Codecs in this crate (`~/codecs`) and their status

| codec | file | status | bytes per value (synthetic, from `size_report`) |
|---|---|---|---|
| `Raw` | `src/raw.rs` | done | always 1.000 |
| `VByte` | `src/vbyte.rs` | done | 1 for 0..=127, 2 for 128..=255 |
| `Bp128` | `src/bp128.rs` | done | 16 × width + 1 bytes per 128 values |
| SIMDBP port | (not started) | stage 6 | |

Size report, 1M values per dataset (`cargo run --release --example size_report`):

| dataset | Raw | VByte | Bp128 |
|---|---|---|---|
| uniform 0..=255 | 1.000 | 1.500 | 1.008 |
| uniform 0..=15 (4-bit) | 1.000 | 1.000 | 0.508 |
| all zeros | 1.000 | 1.000 | 0.008 |
| block-max-like (90% zero, 9% 1..=20, 1% 21..=255) | 1.000 | 1.006 | 0.865 |

### `Bp128` format and how it compares to SIMDBP

```
[ w0 | w1 | ... | w(G-1) ][ chunk 0 data ][ chunk 1 data ] ... [ chunk G-1 data ]
  1 byte per chunk          16 × w0 bytes   16 × w1 bytes
```
- G = `ceil(n / 128)`. `n` is not stored (comes from `out.len()`), same as SIMDBP.
- Packing: `bitpacking::BitPacker4x` (4 interleaved lanes, SSE3 in release builds,
  a scalar fallback in debug builds; both write the same bytes). u16 values are widened
  to u32 for the crate and narrowed back after decoding.
- Similarities to SIMDBP: per-group max bit width, all headers before all data, width 0
  costs no data, last group zero-padded.
- Differences: groups are 128 (SIMDBP on AVX-512: 512); a width takes a whole byte
  (SIMDBP: 4 bits, packed with `pack_4`); no metablocks; the data layout is the crate's
  4-lane interleave, not SIMDBP's STRIDE-lane layout.
- Random access (not implemented yet): chunk k's data starts at
  `G + 16 × (w0 + ... + w(k-1))`. Like SIMDBP's `Decompressor::init`, a reader would
  build that prefix sum once per term, then unpack one 128-value chunk in O(1). The
  current trait can't express this, so it needs the trait discussion first (items 4
  and 6 in section (a)).

### Design decisions and limitations
- **Range check in every build.** All three `encode`s now `assert!` (not
  `debug_assert!`) that values are <= 255. Before, a release build silently truncated
  in `Raw` (256 → 0) and wrote 3-byte values in `VByte` that `decode` rejected.
- **Padding cost.** A short last chunk is padded to 128 values, which costs up to
  127 × width bits. That cost is negligible for long inputs encoded in 128-value pieces,
  but large if a caller encodes many tiny inputs. An alternative is to store the tail
  differently (raw bytes, VByte, or a scalar packer for exactly the leftover values).
- **One width byte per chunk.** Widths are 0..=8, so 4 bits would be enough (as in
  SIMDBP). That saves 0.004 B/value, which matters only for nearly all-zero data
  (all zeros: 0.008 → 0.004).
- **Outliers dominate.** On block-max-like data Bp128 gets only 0.865 B/value even though
  90% of values are 0: with 1% outliers, about 1 − 0.99^128 ≈ 72% of chunks contain at
  least one value >= 21 and are packed at up to 8 bits. Ideas to test on real data:
  smaller chunks (BitPacker1x, 32 values), PFor-style exceptions, or storing the
  outliers separately. Real block maxes may differ from this synthetic mix, so check on
  `export_data` output (stage 5) before optimizing.

### Next stages (see `CLAUDE.md`)
4. Codec registry, 5. real index loader (`export_data` matrices), 6. SIMDBP integration,
7. CLI, 8. benchmarking (criterion speed + bytes per value on real data).

## (b) Questions to verify yourself (not answered here)

1. **Register layout.** Using `pack_scalar`/`unpack_scalar` (`compress_simdbp.rs:17-59`)
   with STRIDE = 32: which 16 element indices does lane 0 hold? After unpacking, does
   output register `j` hold consecutive indices? Which of the two comments at `:8` and
   `:11` is correct?
2. **The quant = 8 case.** What does `(200u8) >> 8` evaluate to in a `--release` build
   vs. a debug build? Try it in a scratch program. How does that explain the
   `if self.quant == 8` branch at `inverted_index.rs:151` and `if quant >= 8` at
   `search.rs:187` and `:352`?
3. **Cost of `init`.** For the running example (level 1, AVX-512), how many u16s does
   `Decompressor::init` read, and how many prefix-sum entries does it write, per query
   term? Is it called for terms whose groups are never touched by the filter
   (`live_block.rs:1231-1233`)? Compare that with the data actually decoded for a query
   with a small filter.
4. **The unused register path.** In `unified_simdbp`, compare the condition on
   `live_block.rs:1102` with what gets pushed on `:1103`. Is the pushed (score, id) the
   one that was compared? Does it matter, given who calls this function?
5. **The ASC leftovers.** At `live_block.rs:726`, the loop runs up to
   `query_range_maxs[0].len()`. Is that the number of block maxes or the length of the
   compressed `Vec<u16>`? What would happen with each?
