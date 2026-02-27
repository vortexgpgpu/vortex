#!/bin/bash
# Compare dense vs sparse TCU performance (cycles, instrs, IPC)
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
SIZES=("8 8 32" "8 8 64" "16 16 32" "16 16 64" "32 32 64" "32 32 128")

extract_perf() {
    local output="$1"
    echo "$output" | grep "^PERF:" | head -1
}

printf "\n"
printf "%-12s  %-14s  %12s %12s %10s  %12s %12s %10s  %8s\n" \
    "Type" "Size" "Dense_Cyc" "Dense_Inst" "Dense_IPC" "Sparse_Cyc" "Sparse_Inst" "Sparse_IPC" "Speedup"
printf "%s\n" "-------------------------------------------------------------------------------------------------------------------------------"

# Build RTL once (supports all sparse input types)
echo "Building rtlsim (all types) ..." >&2
make -C sim/rtlsim clean > /dev/null 2>&1
rm -f runtime/librtlsim.so runtime/libvortex-rtlsim.so
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_TYPE_DPI" make -C runtime/rtlsim > /dev/null 2>&1

for type_spec in "${TYPES[@]}"; do
    IFS=: read -r ITYPE OTYPE <<< "$type_spec"

    # Build dense test
    echo "  Building dense test (sgemm_tcu) for ${ITYPE}/${OTYPE} ..." >&2
    make -C tests/regression/sgemm_tcu clean > /dev/null 2>&1
    CONFIGS="-DNUM_THREADS=8 -DITYPE=$ITYPE -DOTYPE=$OTYPE" make -C tests/regression/sgemm_tcu > /dev/null 2>&1

    # Build sparse test
    echo "  Building sparse test (sgemm_tcu_struct_sparse) for ${ITYPE}/${OTYPE} ..." >&2
    make -C tests/regression/sgemm_tcu_struct_sparse clean > /dev/null 2>&1
    CONFIGS="-DNUM_THREADS=8 -DITYPE=$ITYPE -DOTYPE=$OTYPE" make -C tests/regression/sgemm_tcu_struct_sparse > /dev/null 2>&1

    for size in "${SIZES[@]}"; do
        read -r M N K <<< "$size"
        label="m${M}n${N}k${K}"

        # Run dense
        echo "  Running dense  ${ITYPE}/${OTYPE} ${label} ..." >&2
        dense_out=$(CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_TYPE_DPI" \
            ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu --args="-m${M} -n${N} -k${K}" 2>&1) || true
        dense_perf=$(extract_perf "$dense_out")
        dense_cycles=$(echo "$dense_perf" | grep -oP 'cycles=\K[0-9]+')
        dense_instrs=$(echo "$dense_perf" | grep -oP 'instrs=\K[0-9]+')
        dense_ipc=$(echo "$dense_perf" | grep -oP 'IPC=\K[0-9.]+')

        # Run sparse
        echo "  Running sparse ${ITYPE}/${OTYPE} ${label} ..." >&2
        sparse_out=$(CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_TYPE_DPI" \
            ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_struct_sparse --args="-m${M} -n${N} -k${K}" 2>&1) || true
        sparse_perf=$(extract_perf "$sparse_out")
        sparse_cycles=$(echo "$sparse_perf" | grep -oP 'cycles=\K[0-9]+')
        sparse_instrs=$(echo "$sparse_perf" | grep -oP 'instrs=\K[0-9]+')
        sparse_ipc=$(echo "$sparse_perf" | grep -oP 'IPC=\K[0-9.]+')

        # Compute speedup
        if [ -n "$dense_cycles" ] && [ -n "$sparse_cycles" ] && [ "$sparse_cycles" -gt 0 ]; then
            speedup=$(python3 -c "print(f'{${dense_cycles}/${sparse_cycles}:.2f}x')")
        else
            speedup="N/A"
        fi

        printf "%-12s  %-14s  %12s %12s %10s  %12s %12s %10s  %8s\n" \
            "${ITYPE}/${OTYPE}" "$label" \
            "${dense_cycles:-N/A}" "${dense_instrs:-N/A}" "${dense_ipc:-N/A}" \
            "${sparse_cycles:-N/A}" "${sparse_instrs:-N/A}" "${sparse_ipc:-N/A}" \
            "$speedup"
    done
done

printf "\n"
