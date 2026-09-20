#!/bin/bash
# Stage 6: P2.1 / P2.2 -- the design's actual thesis, never exercised on this tree.
#
# Sizing constraint (learned the hard way on the first attempt): the app caps a
# sparse tile's LMEM payload at a hardcoded 4KB (tile_payload_bytes), which does
# NOT scale with VX_CFG_LMEM_LOG_SIZE. For a 32xtile_K tile the bitmap costs
# 4*tile_K bytes and the compressed operand must fit in the remainder:
#
#   tile_K=128 -> bitmap 512B, compressed budget 3584B = 1792 fp16 elements,
#                 but a 128x32 B tile at 2:4 has 2048 nonzeros. Impossible.
#   tile_K=64  -> bitmap 256B, tile is 2048 elements, 2:4 gives 1024 nonzeros
#                 = 2048B. Fits with headroom.
#
# So sparse runs use SGEMM_TILE_K_MULT=2 (tile_K=64). K=128 then also gives two
# k-chunks per tile, exercising the continuation path alongside sparsity.
#
# Crossbar queue depth 4 per P5.2: depth 2 is what the dense runs inherited and
# is irrelevant there (dense never queues); the exploration settled on 4 for
# sparse, where the queues actually fill.
set -u
R=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
S=$R/status_stage6.txt
: > $S

# $1=M $2=N $3=K $4=sparsity $5=runtime-args $6=tag
sp(){
  local M=$1 N=$2 K=$3 SP=$4 AR=$5 TAG=$6
  local BUILD=~/dev/vortex_spg/build
  local CFG="-DTCU_OP -DVX_CFG_TCU_TYPE=DPI -DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 \
-DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DITYPE=fp16 -DOTYPE=fp32 \
-DSGEMM_CONST_M=$M -DSGEMM_CONST_N=$N -DSGEMM_CONST_K=$K -DSGEMM_CONST_SPARSITY=$SP \
-DTCU_FEOP_BLOCK_M_OVERRIDE=2 -DTCU_FEOP_BLOCK_N_OVERRIDE=16 \
-DTCU_FEOP_XBAR_QUEUE_DEPTH_OVERRIDE=4 -DVX_CFG_LMEM_LOG_SIZE=16 -DSGEMM_TILE_K_MULT=2 \
-DVX_DBG_DEBUG_LEVEL=0"
  cd "$BUILD"
  echo "[$(date +%H:%M:%S)] START $TAG (s=$SP $AR)" >> $S
  : > $R/${TAG}.log
  make -C tests/regression/sgemm_tcu_op clean >/dev/null 2>&1
  if ! CONFIGS="$CFG" make -C tests/regression/sgemm_tcu_op >> $R/${TAG}.log 2>&1; then
    echo "RESULT $TAG BUILD-FAIL" >> $S
    grep -aiE "error:" $R/${TAG}.log | tail -3 >> $S
    return 1
  fi
  local t0=$(date +%s)
  CONFIGS="$CFG" timeout 5400 ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_op \
    --args="-m $M -n $N -k $K -s $SP $AR" --perf=1 >> $R/${TAG}.log 2>&1
  local rc=$? t1=$(date +%s)
  local cyc=$(grep -aoE "PERF: instrs=[0-9]+, cycles=[0-9]+" $R/${TAG}.log | tail -1 | grep -oE "cycles=[0-9]+" | cut -d= -f2)
  local ins=$(grep -aoE "PERF: instrs=[0-9]+" $R/${TAG}.log | tail -1 | cut -d= -f2)
  local v=$(grep -aoE "PASSED|FAILED" $R/${TAG}.log | tail -1)
  [ $rc -eq 124 ] && v=TIMEOUT
  # distinguish a host-side capacity/config rejection from an RTL hang
  if grep -qa "exceeding the .* byte tile budget" $R/${TAG}.log; then v="CONFIG-REJECT(tile budget)"; fi
  [ -z "$v" ] && v="DEADLOCK/ABORT(rc=$rc)"
  echo "RESULT $TAG ${M}x${N}x${K} s=$SP [$AR] rc=$rc wall=$((t1-t0))s cycles=${cyc:-NA} instrs=${ins:-NA} $v" >> $S
}

# --- P2.2: S1, uncompressed A x compressed B (the realistic DNN case) ---
sp 32  32  128 1 "-b 0.5 -p n"  s6_s1_32_24         # 2:4 structured B, single tile
sp 128 128 128 1 "-b 0.5 -p n"  s6_s1_128_24        # multi-tile under DXA staging
sp 128 128 128 1 "-b 0.75 -p u" s6_s1_128_u75       # 75% sparse B, unstructured

# --- P2.1: S2, compressed x compressed (the core contribution) ---
sp 32  32  128 2 "-a 0.5 -b 0.5 -p u"   s6_s2_32_u
sp 128 128 128 2 "-a 0.5 -b 0.5 -p u"   s6_s2_128_u
sp 128 128 128 2 "-a 0.75 -b 0.75 -p u" s6_s2_128_u75

# --- dense control at the same tile_K, so sparse speedup is measured against a
#     matched baseline rather than against the kmult=4 dense numbers ---
sp 128 128 128 0 "" s6_dense_128_k2

echo "[$(date +%H:%M:%S)] STAGE6-COMPLETE" >> $S
