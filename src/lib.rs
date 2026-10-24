//! Compression codecs for block-max data.
//!
//! Every codec implements [`BlockCodec`]: it turns a slice of `u32` values into bytes,
//! and turns those bytes back into the same values.

pub mod raw;
pub mod vbyte;

// Re-export the codec types so users can write `codecs::Raw` instead of `codecs::raw::Raw`.
pub use raw::Raw;
pub use vbyte::VByte;

/// A codec that encodes a block of `u32` values to bytes and decodes them back.
///
/// The codecs are "stateless": a codec is just a type with associated functions, and
/// it is never instantiated. You call them as `VByte::encode(...)`, not `codec.encode(...)`.
pub trait BlockCodec {
    /// How many values the codec prefers to handle per call.
    ///
    /// `Raw` and `VByte` work on any length, so for them this is only a suggestion
    /// to callers that split long inputs into blocks.
    const BLOCK_SIZE: usize;

    /// Encodes `input` and **appends** the bytes to `out`. Bytes that are already in
    /// `out` are left alone, so several blocks can be written into one buffer.
    fn encode(input: &[u32], out: &mut Vec<u8>);

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
    fn decode(input: &[u8], out: &mut [u32]) -> usize;
}
