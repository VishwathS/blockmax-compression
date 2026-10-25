//! Compression codecs for block-max data.
//!
//! Every codec implements [`BlockCodec`]: it turns a slice of `u16` values into bytes,
//! and turns those bytes back into the same values.
//!
//! Why `u16` values that are at most 255? Block maxes are 8-bit impact scores (often
//! quantized further, e.g. to 4 bits), so every value fits in a byte. Parker's SIMDBP
//! code and the query loop still work on `u16`, because that leaves room to multiply a
//! max by a query weight without overflowing. Using `u16` here too means a codec can be
//! dropped into that code without converting.
//!
//! To loop over every codec (reports, benchmarks), use [`Codec::ALL`] from the
//! [`registry`] module.
//!
//! To decode one group of values without decoding a whole row, use
//! [`RandomAccessCodec`] (implemented by `Raw` and `Bp128`) and optionally a
//! [`GroupIndex`]; see the [`random_access`] module.

pub mod bp128;
pub mod random_access;
pub mod raw;
pub mod registry;
pub mod vbyte;

// Re-export the codec types so users can write `codecs::Raw` instead of `codecs::raw::Raw`.
pub use bp128::Bp128;
pub use random_access::{GroupIndex, RandomAccessCodec};
pub use raw::Raw;
pub use registry::Codec;
pub use vbyte::VByte;

/// The largest value a codec has to handle. Inputs above this break the contract.
pub const MAX_VALUE: u16 = 255;

/// A codec that encodes a block of `u16` values to bytes and decodes them back.
///
/// # Contract
///
/// - Every input value is a `u16` that is **at most [`MAX_VALUE`] (255)**. `encode`
///   checks this with `assert!` and panics on a bad value, in debug **and** release
///   builds. (Encoding happens once, offline, so the check costs nothing at query time,
///   and a release build never silently writes a wrong value.)
/// - `encode(input, out)` appends the encoded bytes to `out`.
/// - `decode(input, out)` fills `out` and returns how many bytes it consumed.
///
/// The codecs are "stateless": a codec is just a type with associated functions, and
/// it is never instantiated. You call them as `VByte::encode(...)`, not `codec.encode(...)`.
pub trait BlockCodec {
    /// How many values the codec prefers to handle per call.
    ///
    /// Every codec accepts any input length, so this is only a suggestion to callers
    /// that split long inputs into blocks. For `Bp128` it matters for size: it packs
    /// 128 values at a time and zero-pads a shorter last chunk, so encoding in
    /// multiples of 128 avoids storing padding.
    const BLOCK_SIZE: usize;

    /// Encodes `input` and **appends** the bytes to `out`. Bytes that are already in
    /// `out` are left alone, so several blocks can be written into one buffer.
    ///
    /// # Panics
    ///
    /// Panics if any value in `input` is above `MAX_VALUE` (in every build).
    fn encode(input: &[u16], out: &mut Vec<u8>);

    /// Decodes exactly `out.len()` values from the start of `input` into `out`.
    ///
    /// The format doesn't store the number of values, so the caller must already know
    /// it (that's why it comes from `out.len()`). Bytes after the decoded values are
    /// ignored. Returns how many bytes were consumed, so the caller can find where the
    /// next block starts.
    ///
    /// # Panics
    ///
    /// Panics if `input` is too short or isn't valid output of `encode`.
    fn decode(input: &[u8], out: &mut [u16]) -> usize;
}
