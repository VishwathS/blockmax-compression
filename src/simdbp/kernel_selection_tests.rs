//! Which kernel really runs. These are unit tests (inside the crate) because they read
//! the test-only call counter in `kernel::trace`, which integration tests can't see.
//!
//! A SIMD kernel is tested only where this CPU supports it. Otherwise the test prints
//! a `SKIPPED` line (visible with `cargo test -- --nocapture`), and fails instead if
//! `CODECS_REQUIRE_SIMD=1` is set.

use super::kernel::trace::{self, slot};
use super::{Kernel, SimdBp};
use crate::{BlockCodec, GroupIndex, RandomAccessCodec};

/// The SIMD kernel for `stride` if this CPU can run it. If it can't: prints SKIPPED and
/// returns `None`, or panics when `CODECS_REQUIRE_SIMD=1`.
fn simd_or_skip(stride: usize, test: &str) -> Option<Kernel> {
    let simd = Kernel::simd_for_stride(stride).unwrap();
    if simd.is_available() {
        return Some(simd);
    }
    assert!(
        std::env::var("CODECS_REQUIRE_SIMD").as_deref() != Ok("1"),
        "{test}: {simd:?} is required (CODECS_REQUIRE_SIMD=1) but this CPU can't run it"
    );
    println!("SKIPPED {test}: this CPU can't run {simd:?} (stride {stride})");
    None
}

/// A row with 3 full groups and a partial one, with different widths (including a
/// width-0 group), so every code path that calls a kernel is used.
fn input<const S: usize>() -> Vec<u16> {
    let gl = 16 * S;
    (0..3 * gl + gl / 2)
        .map(|i| match i / gl {
            0 => 0,
            1 => (i % 7) as u16,
            2 => (i % 256) as u16,
            _ => (i % 30) as u16,
        })
        .collect()
}

/// Calls where only `kernel` did any work: `pack` packs and `unpack` unpacks.
fn only(kernel: Kernel, pack: usize, unpack: usize) -> trace::Calls {
    let mut calls = trace::Calls::default();
    calls.pack[slot(kernel)] = pack;
    calls.unpack[slot(kernel)] = unpack;
    calls
}

/// Encodes, decodes, and reads one group with the forced `kernel`, and checks from the
/// trace that `kernel` (and no other) did every pack and unpack.
fn check_forced<const S: usize>(kernel: Kernel) {
    let input = input::<S>();
    let n_groups = SimdBp::<S>::num_groups(input.len());
    let n_headers = SimdBp::<S>::num_metablocks(input.len());

    trace::take(); // start from zero
    let mut bytes = Vec::new();
    SimdBp::<S>::encode_with(kernel, &input, &mut bytes);
    // One pack per group and one per metablock header.
    assert_eq!(
        trace::take(),
        only(kernel, n_groups + n_headers, 0),
        "encode_with({kernel:?}), stride {S}"
    );

    let mut decoded = vec![0u16; input.len()];
    SimdBp::<S>::decode_with(kernel, &bytes, &mut decoded);
    assert_eq!(decoded, input);
    // One unpack per group; the headers are read without a kernel.
    assert_eq!(
        trace::take(),
        only(kernel, 0, n_groups),
        "decode_with({kernel:?}), stride {S}"
    );

    let range = SimdBp::<S>::group_range(&bytes, input.len(), 2);
    let mut out = vec![0u16; 16 * S];
    SimdBp::<S>::decode_group_bytes_with(kernel, &bytes[range], &mut out);
    assert_eq!(out, input[2 * 16 * S..3 * 16 * S]);
    assert_eq!(
        trace::take(),
        only(kernel, 0, 1),
        "decode_group_bytes_with({kernel:?}), stride {S}"
    );
}

fn check_forced_all<const S: usize>(test: &str) {
    check_forced::<S>(Kernel::Scalar); // runs on every CPU
    if let Some(simd) = simd_or_skip(S, test) {
        check_forced::<S>(simd);
        println!("{test}: {simd:?} executed");
    }
}

#[test]
fn forcing_a_kernel_runs_exactly_that_kernel_128() {
    check_forced_all::<8>("forcing_a_kernel_runs_exactly_that_kernel_128");
}

#[test]
fn forcing_a_kernel_runs_exactly_that_kernel_256() {
    check_forced_all::<16>("forcing_a_kernel_runs_exactly_that_kernel_256");
}

#[test]
fn forcing_a_kernel_runs_exactly_that_kernel_512() {
    check_forced_all::<32>("forcing_a_kernel_runs_exactly_that_kernel_512");
}

/// The normal entry points (the trait methods and `GroupIndex`, which pick a kernel
/// themselves) run the SIMD kernel when the CPU has it, and the scalar one otherwise.
fn check_default_paths<const S: usize>(test: &str) {
    let input = input::<S>();
    let n = input.len();
    let n_groups = SimdBp::<S>::num_groups(n);
    let n_headers = SimdBp::<S>::num_metablocks(n);
    let kernel = simd_or_skip(S, test).unwrap_or(Kernel::Scalar);
    assert_eq!(Kernel::best(S), kernel);

    trace::take();
    let mut bytes = Vec::new();
    SimdBp::<S>::encode(&input, &mut bytes);
    assert_eq!(
        trace::take(),
        only(kernel, n_groups + n_headers, 0),
        "encode"
    );

    let mut decoded = vec![0u16; n];
    SimdBp::<S>::decode(&bytes, &mut decoded);
    assert_eq!(trace::take(), only(kernel, 0, n_groups), "decode");

    let mut out = vec![0u16; 16 * S];
    SimdBp::<S>::decode_group(&bytes, n, 3, &mut out);
    assert_eq!(trace::take(), only(kernel, 0, 1), "decode_group");

    let mut index = GroupIndex::<SimdBp<S>>::new();
    index.build(&bytes, n);
    // Building the index reads the 4-bit header fields directly: no kernel at all.
    assert_eq!(trace::take(), trace::Calls::default(), "GroupIndex::build");
    index.decode_group(&bytes, 1, &mut out);
    assert_eq!(
        trace::take(),
        only(kernel, 0, 1),
        "GroupIndex::decode_group"
    );
    println!("{test}: default paths executed {kernel:?}");
}

#[test]
fn default_paths_use_the_best_kernel_128() {
    check_default_paths::<8>("default_paths_use_the_best_kernel_128");
}

#[test]
fn default_paths_use_the_best_kernel_256() {
    check_default_paths::<16>("default_paths_use_the_best_kernel_256");
}

#[test]
fn default_paths_use_the_best_kernel_512() {
    check_default_paths::<32>("default_paths_use_the_best_kernel_512");
}
