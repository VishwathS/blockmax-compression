#!/usr/bin/env bash
# Regenerates the SIMDBP golden vectors (tests/data/simdbp/simdbp{128,256,512}.txt)
# and the probe log (notes/simdbp_oracle_probes.txt) by running Parker's own code.
#
# Needs: Parker's repo (default: ../hierarchical_pruning next to this repo, override
# with PARKER_REPO), the nightly toolchain in rust-toolchain.toml, and an x86_64
# CPU with AVX-512 F+BW (for the 512 build). Build output goes to ~/hp-target/
# (override with ORACLE_TARGET_DIR), never into Parker's repo. His repo is only read;
# this script stops if it has local changes before or after.
#
# Run from anywhere:  tools/simdbp_oracle/regenerate.sh
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PARKER=${PARKER_REPO:-$(cd "$REPO/.." && pwd)/hierarchical_pruning}
TARGET=${ORACLE_TARGET_DIR:-$HOME/hp-target}
OUT=$REPO/tests/data/simdbp
PROBES=$REPO/notes/simdbp_oracle_probes.txt

check_parker_clean() {
    if [ -n "$(git -C "$PARKER" status --porcelain)" ]; then
        echo "error: $PARKER has local changes ($1); refusing to continue" >&2
        exit 1
    fi
}

check_parker_clean "before"
COMMIT=$(git -C "$PARKER" rev-parse HEAD)
cd "$HERE" # so rustup picks the toolchain from rust-toolchain.toml
RUSTC=$(rustc --version)
echo "Parker's repo: $PARKER at $COMMIT; toolchain: $RUSTC"
mkdir -p "$OUT"

# label, Parker's BLOCK_SIZE that build must select, RUSTFLAGS.
# Parker's code picks the block size from cfg(target_feature):
# avx512f -> 512, avx2 -> 256, otherwise 128.
BUILDS=(
    "sse 128 "
    "avx2 256 -C target-feature=+avx2"
    "avx512 512 -C target-cpu=native"
)

: > "$PROBES.tmp"
for build in "${BUILDS[@]}"; do
    read -r label bs flags <<< "$build"
    echo "== build $label (expects BLOCK_SIZE=$bs, RUSTFLAGS='$flags')"
    export RUSTFLAGS="$flags"
    cargo run --release --quiet --target-dir "$TARGET/oracle-$label" -- \
        golden "$OUT/simdbp$bs.txt" "$label" "$bs" "$COMMIT" "$flags" "$RUSTC"
    {
        # A blank line between builds (not after the last one).
        if [ -s "$PROBES.tmp" ]; then echo; fi
        echo "# build=$label RUSTFLAGS='$flags' parker_commit=$COMMIT rustc=\"$RUSTC\""
        cargo run --release --quiet --target-dir "$TARGET/oracle-$label" -- probe
    } >> "$PROBES.tmp"
done
mv "$PROBES.tmp" "$PROBES"

check_parker_clean "after"
echo "wrote $OUT/simdbp{128,256,512}.txt and $PROBES"
