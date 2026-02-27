#!/bin/bash
# Performance breakdown: dense vs sparse with detailed counters
# Must be run from the build/ directory
# Single RTL build supports all types dynamically via fmt_s

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../../../build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: build/ directory not found at $BUILD_DIR"
    exit 1
fi

cd "$BUILD_DIR"

# Use int8/int32 with two sizes: small (overhead-dominated) and large (compute-dominated)
ITYPE=int8; OTYPE=int32
SIZES=("8 8 32" "32 32 128")

# Build RTL once (supports all sparse input types)
echo "Building rtlsim (all types) ..." >&2
make -C sim/rtlsim clean > /dev/null 2>&1
rm -f runtime/librtlsim.so runtime/libvortex-rtlsim.so
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_TYPE_DPI -DPERF_ENABLE" make -C runtime/rtlsim > /dev/null 2>&1

echo "Building dense test ..." >&2
make -C tests/regression/sgemm_tcu clean > /dev/null 2>&1
CONFIGS="-DNUM_THREADS=8 -DITYPE=$ITYPE -DOTYPE=$OTYPE" make -C tests/regression/sgemm_tcu > /dev/null 2>&1

echo "Building sparse test ..." >&2
make -C tests/regression/sgemm_tcu_struct_sparse clean > /dev/null 2>&1
CONFIGS="-DNUM_THREADS=8 -DITYPE=$ITYPE -DOTYPE=$OTYPE" make -C tests/regression/sgemm_tcu_struct_sparse > /dev/null 2>&1

for size in "${SIZES[@]}"; do
    read -r M N K <<< "$size"
    label="m${M}n${N}k${K}"

    echo ""
    echo "================================================================"
    echo "  int8/int32 ${label}"
    echo "================================================================"

    echo ""
    echo "--- DENSE ---"
    CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_TYPE_DPI -DPERF_ENABLE" \
        ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu --perf=1 --args="-m${M} -n${N} -k${K}" 2>&1 \
        | grep -E "^PERF:"

    echo ""
    echo "--- SPARSE ---"
    CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_TYPE_DPI -DPERF_ENABLE" \
        ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_struct_sparse --perf=1 --args="-m${M} -n${N} -k${K}" 2>&1 \
        | grep -E "^PERF:"
done

echo ""
