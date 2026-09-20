#!/bin/bash
# FEDP2K investigation stages 2+3 (runs after perf-counter stage 1):
#  E1/E2: DXA_MAX_INFLIGHT=32 for base and FEDP2K  -> tests gmem-latency bound
#  F0-F4: A-multicast + double-buffer kernel variant (sgemm_tcu_wg_dxa_mcast
#         with -DDXA_MCAST_A -DDXA_DOUBLE_BUFFER), +/- inflight32, +/- FEDP2K
#  E3/E4: debug-trace runs (DBG_TRACE_DXA+DBG_TRACE_PIPELINE, k=64) last
set -u
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
STATUS=$OUT/status_stage2.txt
cd "$BUILD"
: > "$STATUS"

# Wait for stage-1 perf runs to finish (shared build tree).
while ! grep -q "PERF-RUNS-COMPLETE" $OUT/status_rtl_perf.txt 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] stage1 complete, starting stage2" >> "$STATUS"

CFGBASE="-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DITYPE=fp16 -DOTYPE=fp32 -DWGMMA_NRC=16"

run(){
  local APP=$1 EXTRA=$2 TAG=$3 DRV=$4 ARGS=$5 DBGARG=$6
  local CFG="$CFGBASE $EXTRA"
  echo "[$(date +%H:%M:%S)] START $TAG" >> "$STATUS"
  make -C tests/regression/$APP clean >/dev/null 2>&1
  if ! CONFIGS="$CFG" make -C tests/regression/$APP > $OUT/${TAG}.log 2>&1; then
    echo "[$(date +%H:%M:%S)] BUILD-FAIL $TAG" >> "$STATUS"; return
  fi
  t0=$(date +%s)
  CONFIGS="$CFG" ./ci/blackbox.sh --driver=$DRV --app=$APP \
    --args="$ARGS" $DBGARG >> $OUT/${TAG}.log 2>&1
  rc=$?; t1=$(date +%s)
  echo "[$(date +%H:%M:%S)] DONE $TAG rc=$rc wall=$((t1-t0))s" >> "$STATUS"
}

ARGS128="-m 128 -n 128 -k 128"
MC="sgemm_tcu_wg_dxa_mcast"
# Double-buffered smem is 6KB/CTA at NT=32; a 4-CTA cluster needs 24KB, so
# the mcast+dbuf runs scale LMEM to 32KB (LMEM_LOG_SIZE=15).
MCDEF="-DDXA_MCAST_A -DDXA_DOUBLE_BUFFER -DVX_CFG_LMEM_LOG_SIZE=15"

# F0: quick correctness gate on simx (fast) before spending rtlsim time
run $MC "$MCDEF"                                            mcA_dbuf_simx     simx   "$ARGS128" "--perf=1"

# E1/E2: deeper gmem pipelining on the unmodified kernel
run sgemm_tcu_wg_dxa "-DVX_CFG_DXA_MAX_INFLIGHT=32"                     inflight32_base   rtlsim "$ARGS128" "--perf=1"
run sgemm_tcu_wg_dxa "-DVX_CFG_DXA_MAX_INFLIGHT=32 -DVX_CFG_TCU_FEDP2K" inflight32_fedp2k rtlsim "$ARGS128" "--perf=1"

# G1: double buffering alone on the unmodified kernel (LMEM 32KB keeps 4 CTAs)
run sgemm_tcu_wg_dxa "-DDXA_DOUBLE_BUFFER -DVX_CFG_LMEM_LOG_SIZE=15"    dbuf_only         rtlsim "$ARGS128" "--perf=1"

# F1-F4: the fix variant on rtlsim
run $MC "$MCDEF"                                                        mcA_dbuf          rtlsim "$ARGS128" "--perf=1"
run $MC "$MCDEF -DVX_CFG_TCU_FEDP2K"                                    mcA_dbuf_2k       rtlsim "$ARGS128" "--perf=1"
run $MC "$MCDEF -DVX_CFG_DXA_MAX_INFLIGHT=32"                           mcA_dbuf_if32     rtlsim "$ARGS128" "--perf=1"
run $MC "$MCDEF -DVX_CFG_DXA_MAX_INFLIGHT=32 -DVX_CFG_TCU_FEDP2K"       mcA_dbuf_if32_2k  rtlsim "$ARGS128" "--perf=1"

# E3/E4: debug traces at stock config (timeline ground truth), k=64
run sgemm_tcu_wg_dxa "-DDBG_TRACE_DXA -DDBG_TRACE_PIPELINE"                     trace_base   rtlsim "-m 128 -n 128 -k 64" "--debug=3"
run sgemm_tcu_wg_dxa "-DDBG_TRACE_DXA -DDBG_TRACE_PIPELINE -DVX_CFG_TCU_FEDP2K" trace_fedp2k rtlsim "-m 128 -n 128 -k 64" "--debug=3"

echo "STAGE2-COMPLETE" >> "$STATUS"
