#!/usr/bin/env bash
#
# compare_4way_deepbuf.sh — build + run four sgemm TCU configs under RTLSim,
# ALL WITH THE SAME CONFIGS (NUM_WARPS=16, FEDP2K, WGMMA_SS, sparsity, DEEP_BUF
# all enabled in hardware), and diff their PERF (instrs/cycles/IPC):
#
#   1. regular WMMA        (sgemm_tcu)              -- kernel never issues
#   2. sparse  WMMA        (sgemm_tcu_sp)               sparse/WGMMA ops even
#   3. regular WGMMA+DXA, DEEP_BUF (sgemm_tcu_wg_dxa_deepbuf)  though the
#   4. sparse  WGMMA+DXA, DEEP_BUF (sgemm_tcu_wg_sp_dxa)       hardware for it
#                                                               is present
#
# Because all four share one hardware config, sw/runtime/libvortex.so (and
# the RTLSim shared lib) only needs building ONCE: common.mk's rule for it
# has no prerequisites, so once it exists the sub-make that owns it is never
# re-entered -- every subsequent test just links against it. No `make clean`
# or lib removal between legs; each test dir's own config.stamp still forces
# a rebuild of ITS kernel.vxbin (cheap) but not the shared hardware model.
#
# NT=8 (not changed): FEDP2K's actual constraint is a bbuf bank-row
# divisibility invariant that NT=16 violates; NT=8 is the validated-passing
# thread count. This is about NUM_THREADS, independent of NUM_WARPS=16.
#
# WGMMA_SS requires tests/regression/sgemm_tcu_wg_dxa_deepbuf/Makefile's
# WGMMA_RS append to be guarded (fixed alongside this script -- it was
# unconditional, so -DWGMMA_SS on the command line was silently overridden).
#
# The sparse WGMMA leg runs in FULLK mode (VX_CFG_TCU_SPARSE_FULLK): a sparse
# uop drives both FEDP2K halves and covers 2x the dense K, like Hopper's
# wgmma.sp k32 vs dense k16. FULLK requires K-major B (B_KMAJOR) and A from
# smem (WGMMA_SS), so both WGMMA legs use K-major B. KB (K-tiles per pipeline
# stage) is set per leg so both stage the same K per stage: dense tileK=8,
# KB=4 -> 32; sparse FULLK tileK=16, KB=2 -> 32.
#
# DEEP_BUF is passed BOTH ways since the two WG test Makefiles gate it
# differently: sgemm_tcu_wg_dxa_deepbuf reads it as a MAKE variable
# (DEEP_BUF=1, not in CONFIGS); sgemm_tcu_wg_sp_dxa's kernel/main check
# `#if defined(DEEP_BUF)` directly from CONFIGS. Passing both to all four is
# a no-op for sgemm_tcu/sgemm_tcu_sp, whose Makefiles reference neither.
#
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VORTEX_HOME="$(cd "$SCRIPT_DIR/.." && pwd)"

M=256
N=256
K=256
while getopts "m:n:k:" opt; do
  case "$opt" in
    m) M="$OPTARG" ;;
    n) N="$OPTARG" ;;
    k) K="$OPTARG" ;;
    *) echo "usage: $0 [-m M] [-n N] [-k K]"; exit 2 ;;
  esac
done

NT=8
OPTS="-m $M -n $N -k $K"

CONFIGS="-DVX_CFG_NUM_THREADS=$NT -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=4 \
  -DVX_CFG_NUM_LSU_BLOCKS=1 \
  -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_TCU_SPARSE_ENABLE \
  -DVX_CFG_TCU_FEDP2K -DVX_CFG_TCU_USE_DSP=1 -DVX_CFG_TCU_SPARSE_FULLK \
  -DWGMMA_SS -DITYPE=fp16 -DOTYPE=fp32 -DWGMMA_NRC=32 -DPIPE_N=2 -DDEEP_BUF -DB_KMAJOR -DFLOAT_ULP=32"

DENSE_WG_KB=4
SPARSE_WG_KB=2

WORKDIR="$(mktemp -d /tmp/compare_4way_deepbuf.XXXXXX)"

echo "==================================================================="
echo "Shared CONFIGS for all four legs:"
echo "  $CONFIGS"
echo "  + DEEP_BUF=1 (make var, for the dense-deepbuf test's own gate)"
echo "==================================================================="

# Regenerate the two no-regen-rule generated config headers ONCE, matching
# the single shared config every leg below will build/run against.
python3 "$VORTEX_HOME/ci/gen_config.py" --config "$VORTEX_HOME/VX_config.toml" \
  --format verilog --cflags="$CONFIGS -DVX_CFG_XLEN=32" -o "$VORTEX_HOME/hw/VX_config.vh" \
  || { echo "ERROR: hw/VX_config.vh regen failed"; exit 1; }
python3 "$VORTEX_HOME/ci/gen_config.py" --config "$VORTEX_HOME/VX_config.toml" \
  --format cpp --cflags="$CONFIGS -DVX_CFG_XLEN=32" -o "$VORTEX_HOME/sw/VX_config.h" \
  || { echo "ERROR: sw/VX_config.h regen failed"; exit 1; }

# Clean the shared runtime lib ONCE so the first leg below builds it fresh
# against this config; the remaining three reuse it unchanged.
rm -f "$VORTEX_HOME/sw/runtime/librtlsim.so" "$VORTEX_HOME/sw/runtime/libvortex-rtlsim.so" \
      "$VORTEX_HOME/sw/runtime/libvortex.so"

run_one() {
  local label="$1" testdir="$2" extra="${3:-}"
  local dir="$VORTEX_HOME/tests/regression/$testdir"
  local log="$WORKDIR/${testdir}.log"
  echo "==================================================================="
  echo ">>> $label   (dir=tests/regression/$testdir, OPTS=\"$OPTS\", extra=\"$extra\")"
  echo "==================================================================="
  (
    cd "$dir" || exit 1
    env CONFIGS="$CONFIGS $extra" DEEP_BUF=1 OPTS="$OPTS" timeout 3600 make run-rtlsim
  ) > "$log" 2>&1
  local rc=$?
  echo "[$label] exit=$rc" >> "$log"
  tail -n 8 "$log" | sed 's/^/  /'
  echo
}

run_one "regular WMMA"              "sgemm_tcu"
run_one "sparse  WMMA"              "sgemm_tcu_sp"
run_one "regular WGMMA+DXA deepbuf" "sgemm_tcu_wg_dxa_deepbuf" "-DKB=$DENSE_WG_KB"
run_one "sparse  WGMMA+DXA deepbuf" "sgemm_tcu_wg_sp_dxa"      "-DKB=$SPARSE_WG_KB"

extract() {
  local log="$1" field="$2"
  grep -aE "^PERF:" "$log" | tail -1 | sed -nE "s/.*${field}=([0-9.]+).*/\1/p"
}
status_of() {
  local log="$1"
  grep -qa "PASSED!" "$log" && echo "PASS" || echo "FAIL"
}

declare -A STATUS INSTRS CYCLES IPC
for t in sgemm_tcu sgemm_tcu_sp sgemm_tcu_wg_dxa_deepbuf sgemm_tcu_wg_sp_dxa; do
  log="$WORKDIR/${t}.log"
  STATUS[$t]="$(status_of "$log")"
  INSTRS[$t]="$(extract "$log" instrs)"
  CYCLES[$t]="$(extract "$log" cycles)"
  IPC[$t]="$(extract "$log" IPC)"
done

echo "==================================================================="
echo "SUMMARY — 4-way comparison, same CONFIGS, RTLSim, m=$M n=$N k=$K, NT=$NT"
echo "==================================================================="
printf '%-32s %-6s %10s %12s %8s\n' "mode" "status" "instrs" "cycles" "IPC"
printf '%-32s %-6s %10s %12s %8s\n' "regular WMMA"              "${STATUS[sgemm_tcu]}"                 "${INSTRS[sgemm_tcu]:--}"                 "${CYCLES[sgemm_tcu]:--}"                 "${IPC[sgemm_tcu]:--}"
printf '%-32s %-6s %10s %12s %8s\n' "sparse WMMA"               "${STATUS[sgemm_tcu_sp]}"              "${INSTRS[sgemm_tcu_sp]:--}"              "${CYCLES[sgemm_tcu_sp]:--}"              "${IPC[sgemm_tcu_sp]:--}"
printf '%-32s %-6s %10s %12s %8s\n' "regular WGMMA+DXA deepbuf" "${STATUS[sgemm_tcu_wg_dxa_deepbuf]}" "${INSTRS[sgemm_tcu_wg_dxa_deepbuf]:--}" "${CYCLES[sgemm_tcu_wg_dxa_deepbuf]:--}" "${IPC[sgemm_tcu_wg_dxa_deepbuf]:--}"
printf '%-32s %-6s %10s %12s %8s\n' "sparse WGMMA+DXA deepbuf"  "${STATUS[sgemm_tcu_wg_sp_dxa]}"       "${INSTRS[sgemm_tcu_wg_sp_dxa]:--}"       "${CYCLES[sgemm_tcu_wg_sp_dxa]:--}"       "${IPC[sgemm_tcu_wg_sp_dxa]:--}"

echo
echo "per-run logs: $WORKDIR"
echo "SWEEP_DONE"
