//! Parker's generated SIMD pack/unpack functions (x86_64 only), plus one small
//! dispatcher per instruction set that picks the function for a bit width.
//!
//! The three files next to this one are copied from his repo by
//! `scripts/vendor_simdbp_kernels.py`; see the notice at the top of each. They are
//! generated code, so formatting and lints are switched off for them below, and they
//! must not be edited by hand (re-run the script instead).
//!
//! Everything here is `unsafe`: the functions use raw-pointer SIMD loads and stores
//! with no bounds checks, and they use instructions the CPU might not have. The only
//! caller is [`super::Kernel`], which checks both things first.

// `rustfmt::skip` and the `allow`s apply to the whole file each `mod` line pulls in.
// `unused_mut`: the generated code declares some variables `mut` that it never changes.
#[rustfmt::skip]
#[allow(unused_mut, clippy::all)]
mod avx2;
#[rustfmt::skip]
#[allow(unused_mut, clippy::all)]
mod avx512;
#[rustfmt::skip]
#[allow(unused_mut, clippy::all)]
mod sse3;

/// Expands to a `match` on the bit width that calls `$module::$function(input, output)`
/// for widths 0..=8. Written as a macro so the nine arms aren't repeated six times.
macro_rules! by_width {
    ($width:expr, $module:ident, $input:expr, $output:expr,
     [$($w:literal => $function:ident),* $(,)?]) => {
        match $width {
            $($w => $module::$function($input, $output),)*
            _ => unreachable!("the caller checks that the width is at most 8"),
        }
    };
}

// The six dispatchers. Each has the same safety contract:
//
// # Safety
//
// - The CPU supports the instruction set in the function's name (SSE3, AVX2, or
//   AVX-512 F + BW). Calling it otherwise is undefined behavior (illegal instruction).
// - `width <= 8`.
// - pack: `input.len() >= 16 × STRIDE` and `output.len() >= width × STRIDE`.
//   unpack: `input.len() >= width × STRIDE` and `output.len() >= 16 × STRIDE`.
//   (STRIDE is 8 for SSE3, 16 for AVX2, 32 for AVX-512.) The kernels don't check.

pub(super) unsafe fn pack_sse3(width: usize, input: &[u16], output: &mut [u16]) {
    // SAFETY: forwarded from this function's contract (see above).
    unsafe {
        by_width!(width, sse3, input, output, [
            0 => pack128_0, 1 => pack128_1, 2 => pack128_2, 3 => pack128_3, 4 => pack128_4,
            5 => pack128_5, 6 => pack128_6, 7 => pack128_7, 8 => pack128_8,
        ])
    }
}

pub(super) unsafe fn unpack_sse3(width: usize, input: &[u16], output: &mut [u16]) {
    // SAFETY: forwarded from this function's contract (see above).
    unsafe {
        by_width!(width, sse3, input, output, [
            0 => unpack128_0, 1 => unpack128_1, 2 => unpack128_2, 3 => unpack128_3,
            4 => unpack128_4, 5 => unpack128_5, 6 => unpack128_6, 7 => unpack128_7,
            8 => unpack128_8,
        ])
    }
}

pub(super) unsafe fn pack_avx2(width: usize, input: &[u16], output: &mut [u16]) {
    // SAFETY: forwarded from this function's contract (see above).
    unsafe {
        by_width!(width, avx2, input, output, [
            0 => pack256_0, 1 => pack256_1, 2 => pack256_2, 3 => pack256_3, 4 => pack256_4,
            5 => pack256_5, 6 => pack256_6, 7 => pack256_7, 8 => pack256_8,
        ])
    }
}

pub(super) unsafe fn unpack_avx2(width: usize, input: &[u16], output: &mut [u16]) {
    // SAFETY: forwarded from this function's contract (see above).
    unsafe {
        by_width!(width, avx2, input, output, [
            0 => unpack256_0, 1 => unpack256_1, 2 => unpack256_2, 3 => unpack256_3,
            4 => unpack256_4, 5 => unpack256_5, 6 => unpack256_6, 7 => unpack256_7,
            8 => unpack256_8,
        ])
    }
}

pub(super) unsafe fn pack_avx512(width: usize, input: &[u16], output: &mut [u16]) {
    // SAFETY: forwarded from this function's contract (see above).
    unsafe {
        by_width!(width, avx512, input, output, [
            0 => pack512_0, 1 => pack512_1, 2 => pack512_2, 3 => pack512_3, 4 => pack512_4,
            5 => pack512_5, 6 => pack512_6, 7 => pack512_7, 8 => pack512_8,
        ])
    }
}

pub(super) unsafe fn unpack_avx512(width: usize, input: &[u16], output: &mut [u16]) {
    // SAFETY: forwarded from this function's contract (see above).
    unsafe {
        by_width!(width, avx512, input, output, [
            0 => unpack512_0, 1 => unpack512_1, 2 => unpack512_2, 3 => unpack512_3,
            4 => unpack512_4, 5 => unpack512_5, 6 => unpack512_6, 7 => unpack512_7,
            8 => unpack512_8,
        ])
    }
}
