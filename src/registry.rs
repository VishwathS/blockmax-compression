//! The codec registry: one list of every codec, so callers can loop over them.
//!
//! The codecs themselves are types with associated functions (`VByte::encode(...)`),
//! which is great for speed but means you can't put "a codec" in a variable or a
//! `Vec`. [`Codec`] fixes that: it's an enum with one variant per codec, and its
//! methods forward ("dispatch") to the matching type with a `match`.
//!
//! ```
//! use codecs::Codec;
//!
//! for codec in Codec::ALL {
//!     let mut bytes = Vec::new();
//!     codec.encode(&[1, 2, 3], &mut bytes);
//!     let mut back = [0u16; 3];
//!     let used = codec.decode(&bytes, &mut back);
//!     assert_eq!(back, [1, 2, 3]);
//!     assert_eq!(used, bytes.len());
//! }
//! ```
//!
//! # Adding a codec
//!
//! 1. Write the type and implement [`BlockCodec`] for it (e.g. `src/pfor.rs`).
//! 2. Add a variant to [`Codec`] and append it to [`Codec::ALL`].
//! 3. Add an arm to each `match` below. The compiler lists every `match` that is
//!    missing the new variant, so none can be forgotten.
//! 4. Update the expected list in the `registry_lists_exactly_the_implemented_codecs`
//!    test (`tests/roundtrip.rs`).

use crate::{BlockCodec, Bp128, RandomAccessCodec, Raw, VByte};

/// One variant per codec. `Copy` because it's just a tag (one byte), so passing it
/// around by value is free.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    Raw,
    VByte,
    Bp128,
}

impl Codec {
    /// Every codec, in the order reports print them (baseline first).
    pub const ALL: [Codec; 3] = [Codec::Raw, Codec::VByte, Codec::Bp128];

    /// The codec's name, as used in reports and (later) on the command line. Treat
    /// these as stable: changing one would break scripts and saved results.
    pub fn name(self) -> &'static str {
        match self {
            Codec::Raw => "Raw",
            Codec::VByte => "VByte",
            Codec::Bp128 => "Bp128",
        }
    }

    /// Looks a codec up by its [`name`](Codec::name), ignoring upper/lower case.
    /// Returns `None` if no codec has that name.
    pub fn from_name(name: &str) -> Option<Codec> {
        Codec::ALL
            .into_iter()
            .find(|codec| codec.name().eq_ignore_ascii_case(name))
    }

    /// The codec's preferred values per `encode` call ([`BlockCodec::BLOCK_SIZE`]).
    pub fn block_size(self) -> usize {
        match self {
            Codec::Raw => Raw::BLOCK_SIZE,
            Codec::VByte => VByte::BLOCK_SIZE,
            Codec::Bp128 => Bp128::BLOCK_SIZE,
        }
    }

    /// Same as [`BlockCodec::encode`] for this codec: appends the encoded bytes to
    /// `out`, and panics on a value above 255.
    pub fn encode(self, input: &[u16], out: &mut Vec<u8>) {
        match self {
            Codec::Raw => Raw::encode(input, out),
            Codec::VByte => VByte::encode(input, out),
            Codec::Bp128 => Bp128::encode(input, out),
        }
    }

    /// Same as [`BlockCodec::decode`] for this codec: fills `out` and returns how
    /// many bytes of `input` it consumed.
    pub fn decode(self, input: &[u8], out: &mut [u16]) -> usize {
        match self {
            Codec::Raw => Raw::decode(input, out),
            Codec::VByte => VByte::decode(input, out),
            Codec::Bp128 => Bp128::decode(input, out),
        }
    }

    /// The group size if this codec supports random access ([`RandomAccessCodec`]),
    /// or `None` if it doesn't (`VByte`).
    pub fn group_len(self) -> Option<usize> {
        match self {
            Codec::Raw => Some(Raw::GROUP_LEN),
            Codec::VByte => None,
            Codec::Bp128 => Some(Bp128::GROUP_LEN),
        }
    }

    /// Same as [`RandomAccessCodec::decode_group`] for this codec (one-shot, no index):
    /// returns `Some(number of values written)`, or `None` if the codec doesn't
    /// support random access. Panics on an invalid group index, like the trait.
    pub fn decode_group(
        self,
        input: &[u8],
        n: usize,
        group: usize,
        out: &mut [u16],
    ) -> Option<usize> {
        match self {
            Codec::Raw => Some(Raw::decode_group(input, n, group, out)),
            Codec::VByte => None,
            Codec::Bp128 => Some(Bp128::decode_group(input, n, group, out)),
        }
    }
}
