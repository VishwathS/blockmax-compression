//! Packing and unpacking one SIMDBP group: a portable scalar version (the reference)
//! and Parker's SIMD versions, chosen at run time.
//!
//! # Layout of one packed group
//!
//! A group has `16 × STRIDE` values, split into `STRIDE` **lanes**: value
//! `i + j × STRIDE` belongs to lane `i` (so lane `i` holds values `i`, `i + STRIDE`,
//! `i + 2 × STRIDE`, ..., 16 values in all). At bit width `w`, lane `i` is a little
//! stream of `16 × w` bits: its `j`-th value sits at bits `j × w .. (j + 1) × w`.
//! That stream is stored as `w` u16 words, and word `k` of lane `i` goes to position
//! `k × STRIDE + i` of the output. So a packed group is `w × STRIDE` u16 words.
//!
//! This is the layout one SIMD register per word produces: a register holds one u16
//! from each of the `STRIDE` lanes, and shifting/OR-ing whole registers packs all
//! lanes at once. With width 0 the group takes no words at all.
//!
//! # Provenance
//!
//! `pack_scalar` and `unpack_scalar` below are ports of `pack_scalar` /
//! `unpack_scalar` in `src/index/compress_simdbp.rs` of Parker Carlson's
//! hierarchical_pruning (<https://github.com/thefxperson/hierarchical_pruning>, commit
//! fa0ede5241ea913ec9e293bdbef0d0b1012e7f07), Apache License 2.0 (copy in
//! `third_party/hierarchical_pruning/LICENSE`). MODIFIED: they write into a slice the
//! caller provides instead of returning a new `Vec`, handle width 0 explicitly, and
//! their arguments are checked by [`Kernel::pack`] / [`Kernel::unpack`]. The bit
//! layout is the same.

/// Largest bit width: values are at most 255, which fits in 8 bits.
pub const MAX_WIDTH: usize = 8;

/// Largest number of lanes (AVX-512: 32 u16s per register).
pub const MAX_STRIDE: usize = 32;

/// Largest group: 16 values per lane × 32 lanes = 512.
pub const MAX_GROUP_LEN: usize = 16 * MAX_STRIDE;

/// Which implementation packs and unpacks groups.
///
/// All of them produce exactly the same bits; the tests check that for every width
/// and stride. The codec normally uses [`Kernel::best`]. Picking one by hand is for
/// tests (e.g. "does the scalar version give the same bytes as the SIMD one?").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// Plain Rust, any CPU, any stride. The reference implementation.
    Scalar,
    /// Parker's SSE3 kernels: stride 8 (groups of 128).
    Sse3,
    /// Parker's AVX2 kernels: stride 16 (groups of 256).
    Avx2,
    /// Parker's AVX-512 (F + BW) kernels: stride 32 (groups of 512).
    Avx512,
}

impl Kernel {
    /// Every kernel, scalar first.
    pub const ALL: [Kernel; 4] = [Kernel::Scalar, Kernel::Sse3, Kernel::Avx2, Kernel::Avx512];

    /// The SIMD kernel written for groups with `stride` lanes, whether or not this CPU
    /// can run it. `None` for a stride no SIMD kernel handles.
    pub fn simd_for_stride(stride: usize) -> Option<Kernel> {
        match stride {
            8 => Some(Kernel::Sse3),
            16 => Some(Kernel::Avx2),
            32 => Some(Kernel::Avx512),
            _ => None,
        }
    }

    /// The fastest kernel this CPU can run for `stride` lanes: the SIMD one if the CPU
    /// has its instructions, otherwise [`Kernel::Scalar`].
    pub fn best(stride: usize) -> Kernel {
        match Kernel::simd_for_stride(stride) {
            Some(simd) if simd.is_available() => simd,
            _ => Kernel::Scalar,
        }
    }

    /// Whether this CPU can run the kernel. Checked at run time (the answer is cached
    /// by the standard library, so asking again is cheap). `Scalar` always can.
    pub fn is_available(self) -> bool {
        match self {
            Kernel::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Kernel::Sse3 => std::arch::is_x86_feature_detected!("sse3"),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => std::arch::is_x86_feature_detected!("avx2"),
            // Parker's AVX-512 kernels shift 16-bit lanes (`_mm512_slli_epi16`), which
            // is part of AVX512BW, not just the AVX512F foundation.
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => {
                std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw")
            }
            // The SIMD kernels are x86_64 code; on other CPUs only Scalar exists.
            #[cfg(not(target_arch = "x86_64"))]
            _ => false,
        }
    }

    /// Packs one group of `16 × stride` values at `width` bits into `output`
    /// (`width × stride` u16 words). Every value must fit in `width` bits.
    ///
    /// # Panics
    ///
    /// Panics if a length is wrong, `width > 8`, a value doesn't fit in `width` bits,
    /// this is a SIMD kernel for a different stride, or the CPU can't run it.
    pub fn pack(self, stride: usize, width: usize, input: &[u16], output: &mut [u16]) {
        check_args(stride, width, input.len(), output.len(), "pack");
        // The SIMD kernels don't mask their inputs: a value with more than `width` bits
        // would spill into the next value's bits. The scalar version masks instead. To
        // keep every kernel giving the same answer, reject such input up front.
        let limit = 1u32 << width; // values must be < 2^width
        assert!(
            input.iter().all(|&v| u32::from(v) < limit),
            "SimdBp: a value doesn't fit in {width} bits"
        );
        self.check_runnable(stride);
        // Test builds only: note that `self` is the kernel the `match` below runs.
        #[cfg(test)]
        trace::record(self, trace::Op::Pack);
        match self {
            Kernel::Scalar => pack_scalar(input, stride, width, output),
            // SAFETY (all three arms): `check_runnable` confirmed this kernel's stride
            // and that the CPU has its instructions; `check_args` confirmed width <= 8
            // and both lengths, which is the dispatchers' whole contract.
            #[cfg(target_arch = "x86_64")]
            Kernel::Sse3 => unsafe { super::kernels::pack_sse3(width, input, output) },
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => unsafe { super::kernels::pack_avx2(width, input, output) },
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => unsafe { super::kernels::pack_avx512(width, input, output) },
            // `check_runnable` already panicked for SIMD kernels on other CPUs.
            #[cfg(not(target_arch = "x86_64"))]
            _ => unreachable!(),
        }
    }

    /// Unpacks one group packed at `width` bits (`width × stride` u16 words) into
    /// `output` (`16 × stride` values). Any input bits are accepted.
    ///
    /// # Panics
    ///
    /// Panics if a length is wrong, `width > 8`, this is a SIMD kernel for a different
    /// stride, or the CPU can't run it.
    pub fn unpack(self, stride: usize, width: usize, input: &[u16], output: &mut [u16]) {
        // `check_args` takes (values, words); for unpack the values are the output.
        check_args(stride, width, output.len(), input.len(), "unpack");
        self.check_runnable(stride);
        #[cfg(test)]
        trace::record(self, trace::Op::Unpack);
        match self {
            Kernel::Scalar => unpack_scalar(input, stride, width, output),
            // SAFETY: same as in `pack`.
            #[cfg(target_arch = "x86_64")]
            Kernel::Sse3 => unsafe { super::kernels::unpack_sse3(width, input, output) },
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => unsafe { super::kernels::unpack_avx2(width, input, output) },
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => unsafe { super::kernels::unpack_avx512(width, input, output) },
            #[cfg(not(target_arch = "x86_64"))]
            _ => unreachable!(),
        }
    }

    /// Panics unless this kernel handles `stride` and the CPU can run it.
    fn check_runnable(self, stride: usize) {
        if self != Kernel::Scalar {
            assert_eq!(
                Kernel::simd_for_stride(stride),
                Some(self),
                "SimdBp: kernel {self:?} doesn't handle stride {stride}"
            );
            assert!(
                self.is_available(),
                "SimdBp: this CPU can't run kernel {self:?}"
            );
        }
    }
}

/// Checks the arguments every pack/unpack shares. `n_values` is the unpacked side
/// (`16 × stride`), `n_words` the packed side (`width × stride`).
fn check_args(stride: usize, width: usize, n_values: usize, n_words: usize, what: &str) {
    assert!(
        (1..=MAX_STRIDE).contains(&stride),
        "SimdBp {what}: invalid stride {stride}"
    );
    assert!(
        width <= MAX_WIDTH,
        "SimdBp {what}: invalid bit width {width}"
    );
    assert_eq!(
        n_values,
        16 * stride,
        "SimdBp {what}: a group has {} values",
        16 * stride
    );
    assert_eq!(
        n_words,
        width * stride,
        "SimdBp {what}: a group at width {width} has {} words",
        width * stride
    );
}

/// The reference packer (see the module docs for the layout). Lengths are checked by
/// [`Kernel::pack`].
fn pack_scalar(input: &[u16], stride: usize, width: usize, output: &mut [u16]) {
    output.fill(0); // bits are OR-ed in below, so start from all zeros
    if width == 0 {
        return; // a width-0 group stores nothing
    }
    let mask = (1u32 << width) - 1;
    for lane in 0..stride {
        for j in 0..16 {
            let value = u32::from(input[lane + j * stride]) & mask;
            let bit = j * width; // where this value starts in the lane's bit stream
            let (word, shift) = (bit / 16, bit % 16);
            // Low part: as many of the value's bits as fit in this word. The `as u16`
            // drops any bits that went past bit 15; they're written just below.
            output[word * stride + lane] |= (value << shift) as u16;
            // High part: if the value crosses into the next word, its remaining bits go
            // to the bottom of that word.
            let room = 16 - shift; // bits left in this word
            if room < width {
                output[(word + 1) * stride + lane] |= (value >> room) as u16;
            }
        }
    }
}

/// The reference unpacker, the exact inverse of `pack_scalar`. Lengths are checked
/// by [`Kernel::unpack`].
fn unpack_scalar(input: &[u16], stride: usize, width: usize, output: &mut [u16]) {
    if width == 0 {
        output.fill(0); // a width-0 group is all zeros
        return;
    }
    let mask = (1u32 << width) - 1;
    for lane in 0..stride {
        for j in 0..16 {
            let bit = j * width;
            let (word, shift) = (bit / 16, bit % 16);
            let mut value = u32::from(input[word * stride + lane]) >> shift;
            let room = 16 - shift;
            if room < width {
                // The value's top bits are at the bottom of the next word.
                value |= u32::from(input[(word + 1) * stride + lane]) << room;
            }
            output[lane + j * stride] = (value & mask) as u16;
        }
    }
}

/// Records which kernel really ran, so tests can prove a SIMD kernel was executed (not
/// just compiled) and that forcing a kernel works. Compiled only into this crate's own
/// unit tests (`cfg(test)`); normal and benchmark builds don't contain it at all.
#[cfg(test)]
pub(crate) mod trace {
    use std::cell::Cell;

    use super::Kernel;

    /// Which operation a call was.
    #[derive(Clone, Copy)]
    pub(crate) enum Op {
        Pack,
        Unpack,
    }

    /// How many groups (or headers) each kernel packed and unpacked, in the order of
    /// [`Kernel::ALL`] (Scalar, Sse3, Avx2, Avx512).
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub(crate) struct Calls {
        pub(crate) pack: [usize; 4],
        pub(crate) unpack: [usize; 4],
    }

    thread_local! {
        // One counter per thread: the test harness runs tests on several threads at
        // once, and each test must only see its own calls.
        static CALLS: Cell<Calls> = Cell::new(Calls::default());
    }

    /// Position of `kernel` in `Kernel::ALL`.
    pub(crate) fn slot(kernel: Kernel) -> usize {
        Kernel::ALL.iter().position(|&k| k == kernel).unwrap()
    }

    pub(super) fn record(kernel: Kernel, op: Op) {
        CALLS.with(|cell| {
            let mut calls = cell.get();
            match op {
                Op::Pack => calls.pack[slot(kernel)] += 1,
                Op::Unpack => calls.unpack[slot(kernel)] += 1,
            }
            cell.set(calls);
        });
    }

    /// The counts recorded on this thread so far; resets them to zero.
    pub(crate) fn take() -> Calls {
        CALLS.with(|cell| cell.replace(Calls::default()))
    }
}
