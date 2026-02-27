#!/bin/bash
# Sweep sparse TCU with larger M/N/K: 3 types x 5 sizes = 15 configs
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

TYPES=("int8:int32" "fp16:fp32" "int4:int32")
SIZES=("16 16 64" "32 32 64" "32 32 128" "64 64 128" "64 64 256")

PASS=0
FAIL=0
TOTAL=0

# Build RTL once (supports all sparse input types)
echo "============================================"
echo "Building rtlsim (all types)"
echo "============================================"
make -C sim/rtlsim clean > /dev/null 2>&1
rm -f runtime/librtlsim.so runtime/libvortex-rtlsim.so
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_TYPE_DPI" make -C runtime/rtlsim > /dev/null 2>&1

for type_spec in "${TYPES[@]}"; do
    IFS=: read -r ITYPE OTYPE <<< "$type_spec"

    echo "============================================"
    echo "Building test with ITYPE=$ITYPE OTYPE=$OTYPE"
    echo "============================================"
    make -C tests/regression/sgemm_tcu_struct_sparse clean > /dev/null 2>&1
    CONFIGS="-DNUM_THREADS=8 -DITYPE=$ITYPE -DOTYPE=$OTYPE" make -C tests/regression/sgemm_tcu_struct_sparse > /dev/null 2>&1

    for size in "${SIZES[@]}"; do
        read -r M N K <<< "$size"
        TOTAL=$((TOTAL + 1))
        echo -n "  ${ITYPE}/${OTYPE} m${M}n${N}k${K} ... "
        if CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_TYPE_DPI" \
            ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_struct_sparse --args="-m${M} -n${N} -k${K}" > /dev/null 2>&1; then
            echo "PASS"
            PASS=$((PASS + 1))
        else
            echo "FAIL"
            FAIL=$((FAIL + 1))
        fi
    done
done

echo "============================================"
echo "Results: $PASS passed, $FAIL failed (out of $TOTAL)"
echo "============================================"

[ "$FAIL" -eq 0 ]
