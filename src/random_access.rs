//! Random access: decode one **group** of an encoded row without decoding the rest.
//!
//! A "row" is the output of one [`BlockCodec::encode`] call (e.g. one term's block
//! maxes at one level). A **group** is a run of [`RandomAccessCodec::GROUP_LEN`]
//! consecutive values: group `g` holds values `g × GROUP_LEN .. (g + 1) × GROUP_LEN`.
//! The last group is shorter when `n` isn't a multiple of `GROUP_LEN`.
//!
//! There are two ways to read a group:
//!
//! 1. **One-shot, no extra state:** [`RandomAccessCodec::decode_group`]. It finds the
//!    group from the row's own metadata each time. How long that takes depends on the
//!    codec (for `Bp128` it sums the width bytes before the group, so it is O(g)).
//! 2. **Index first, then O(1) per group:** build a [`GroupIndex`] once per row (O(G),
//!    one pass over the metadata), then call [`GroupIndex::decode_group`] as often as
//!    needed. This is what Parker's SIMDBP `Decompressor::init` does. The index lives in
//!    memory only; nothing is added to the encoded bytes.
//!
//! ```
//! use codecs::{BlockCodec, Bp128, GroupIndex, RandomAccessCodec};
//!
//! let values: Vec<u16> = (0..300).map(|i| (i % 7) as u16).collect();
//! let mut bytes = Vec::new();
//! Bp128::encode(&values, &mut bytes);
//!
//! // One-shot: group 1 holds values 128..256.
//! let mut group = [0u16; Bp128::GROUP_LEN];
//! let len = Bp128::decode_group(&bytes, values.len(), 1, &mut group);
//! assert_eq!(&group[..len], &values[128..256]);
//!
//! // With an index: build once, then each lookup is O(1).
//! let mut index = GroupIndex::<Bp128>::new();
//! index.build(&bytes, values.len());
//! let len = index.decode_group(&bytes, 2, &mut group); // the partial last group
//! assert_eq!(&group[..len], &values[256..300]);
//! ```

use std::marker::PhantomData;
use std::ops::Range;

use crate::BlockCodec;

/// A codec whose encoded rows can be decoded one group at a time.
///
/// It extends [`BlockCodec`] instead of changing it: `encode` and `decode` stay exactly
/// as they are, and a codec that can't seek cheaply (like `VByte`, where every value's
/// position depends on all earlier values) simply doesn't implement this trait.
///
/// Like `BlockCodec`, the codecs are stateless: every function takes the encoded row
/// and `n` (the number of values in it), and `n` is never stored in the bytes.
///
/// `input` always starts at the beginning of the row. Bytes after the row (e.g. the next
/// row in the same buffer) are ignored, the same as in `decode`.
pub trait RandomAccessCodec: BlockCodec {
    /// Values per group. Every group has this many values except possibly the last.
    const GROUP_LEN: usize;

    /// Number of groups in a row of `n` values (the last one may be partial).
    fn num_groups(n: usize) -> usize {
        n.div_ceil(Self::GROUP_LEN)
    }

    /// Number of values in group `group` of a row of `n` values: `GROUP_LEN`, except for
    /// a partial last group.
    ///
    /// # Panics
    ///
    /// Panics if `group >= num_groups(n)`.
    fn group_len(n: usize, group: usize) -> usize {
        let n_groups = Self::num_groups(n);
        assert!(
            group < n_groups,
            "group index {group} is out of range: a row of {n} values has {n_groups} groups"
        );
        // Every group before the last is full; the last one gets whatever is left.
        (n - group * Self::GROUP_LEN).min(Self::GROUP_LEN)
    }

    /// The byte range of group `group`'s data inside `input`, found from the row's
    /// metadata alone (no values are decoded). Pass `&input[range]` to
    /// [`decode_group_bytes`](Self::decode_group_bytes).
    ///
    /// Cost depends on the codec; see its docs.
    ///
    /// # Panics
    ///
    /// Panics if `group` is out of range, or if `input` is too short or invalid.
    fn group_range(input: &[u8], n: usize, group: usize) -> Range<usize>;

    /// Writes where every group starts into `starts`, in one pass over the metadata.
    ///
    /// Afterwards `starts` has `num_groups(n) + 1` entries: `starts[g]` is the first
    /// byte of group g's data, and the last entry is the end of the row (the number of
    /// bytes `decode` would consume). So group g's data is `starts[g]..starts[g + 1]`.
    /// `starts` is cleared first; its capacity is kept, so reusing one `Vec` for many
    /// rows doesn't allocate once it's big enough.
    ///
    /// # Panics
    ///
    /// Panics if `input` is too short or invalid.
    fn locate_groups(input: &[u8], n: usize, starts: &mut Vec<usize>);

    /// Decodes one group from its data bytes (`&input[group_range(..)]`) into `out`.
    /// `out.len()` must be the group's length ([`group_len`](Self::group_len)).
    ///
    /// # Panics
    ///
    /// Panics if `data` isn't a valid group for `out.len()` values.
    fn decode_group_bytes(data: &[u8], out: &mut [u16]);

    /// Decodes group `group` of the row in `input` (which has `n` values) into the
    /// start of `out`, and returns how many values it wrote (the group's length).
    /// Values of `out` past that are left unchanged.
    ///
    /// `out` needs room for the group, so a `[u16; GROUP_LEN]` buffer always works and
    /// can be reused for every call (no allocation).
    ///
    /// # Panics
    ///
    /// Panics if `group >= num_groups(n)`, if `out` is shorter than the group, or if
    /// `input` is too short or invalid.
    fn decode_group(input: &[u8], n: usize, group: usize, out: &mut [u16]) -> usize {
        let len = Self::group_len(n, group); // also checks the group index
        assert!(
            out.len() >= len,
            "output buffer has room for {} values, but group {group} has {len}",
            out.len()
        );
        let range = Self::group_range(input, n, group);
        Self::decode_group_bytes(&input[range], &mut out[..len]);
        len
    }
}

/// Where each group of one encoded row starts, so groups can be found in O(1).
///
/// Build it once per row with [`build`](GroupIndex::build) (one O(G) pass over the
/// metadata), then decode groups in any order with [`decode_group`](GroupIndex::decode_group).
/// The index is tied to one codec type `C` so it can't be used with the wrong codec.
///
/// Memory: one `usize` (8 bytes) per group plus one, held in RAM while the row is being
/// read. Nothing is stored in the encoded bytes. `build` reuses the existing allocation,
/// so one `GroupIndex` can be rebuilt for row after row without allocating (like
/// Parker's preallocated `Decompressor`s).
pub struct GroupIndex<C: RandomAccessCodec> {
    /// Number of values in the row this index was built for.
    n: usize,
    /// `starts[g]..starts[g + 1]` is group g's data; see `locate_groups`.
    starts: Vec<usize>,
    /// Remembers the codec type without storing anything (`PhantomData` is zero-sized).
    codec: PhantomData<C>,
}

// Written by hand instead of `#[derive(Default)]`, because the derive would also
// require `C: Default`, and codec types don't implement it.
impl<C: RandomAccessCodec> Default for GroupIndex<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: RandomAccessCodec> GroupIndex<C> {
    /// An empty index (zero groups). Call [`build`](Self::build) before use.
    pub fn new() -> Self {
        GroupIndex {
            n: 0,
            // An empty row still has one entry: the end of the row, at byte 0.
            starts: vec![0],
            codec: PhantomData,
        }
    }

    /// Rebuilds the index for the row in `input`, which has `n` values. O(G).
    ///
    /// # Panics
    ///
    /// Panics if `input` is too short or invalid.
    pub fn build(&mut self, input: &[u8], n: usize) {
        C::locate_groups(input, n, &mut self.starts);
        self.n = n;
        debug_assert_eq!(self.starts.len(), C::num_groups(n) + 1);
    }

    /// Number of values in the row.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of groups in the row.
    pub fn num_groups(&self) -> usize {
        self.starts.len() - 1
    }

    /// Total encoded size of the row in bytes, metadata included. Equals what
    /// [`BlockCodec::decode`] returns for this row.
    pub fn encoded_len(&self) -> usize {
        // `starts` always has at least one entry (see `new` and `locate_groups`).
        self.starts[self.starts.len() - 1]
    }

    /// Bytes of RAM the index itself uses (its `Vec`'s capacity). This is the memory
    /// cost of O(1) access; the encoded row doesn't grow.
    pub fn heap_bytes(&self) -> usize {
        self.starts.capacity() * std::mem::size_of::<usize>()
    }

    /// The byte range of group `group`'s data. O(1).
    ///
    /// # Panics
    ///
    /// Panics if `group >= num_groups()`.
    pub fn group_range(&self, group: usize) -> Range<usize> {
        assert!(
            group < self.num_groups(),
            "group index {group} is out of range: the row has {} groups",
            self.num_groups()
        );
        self.starts[group]..self.starts[group + 1]
    }

    /// Same as [`RandomAccessCodec::decode_group`], but finds the group in O(1) using
    /// the index. `input` must be the same row the index was built for.
    ///
    /// # Panics
    ///
    /// Panics if `group >= num_groups()`, if `out` is shorter than the group, or if
    /// `input` is shorter than the row.
    pub fn decode_group(&self, input: &[u8], group: usize, out: &mut [u16]) -> usize {
        let len = C::group_len(self.n, group); // also checks the group index
        assert!(
            out.len() >= len,
            "output buffer has room for {} values, but group {group} has {len}",
            out.len()
        );
        let range = self.group_range(group);
        let data = input
            .get(range)
            .expect("input is shorter than the row this GroupIndex was built for");
        C::decode_group_bytes(data, &mut out[..len]);
        len
    }
}
