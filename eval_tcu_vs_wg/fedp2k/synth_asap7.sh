#!/bin/bash
# Yosys + OpenSTA ASAP7 synthesis, NW=16 NT=32 IW=4, two TCU configurations.
#
# DUT is the full Vortex, not a TCU unit wrapper. That is forced, not preferred:
#   - hw/unittest/tcu_unit/VX_tcu_unit_top.sv states "Only ISSUE_WIDTH == 1 is
#     supported by this wrapper" and declares no TCU_OP ports at all.
#   - hw/unittest/tensor/VX_tensor_top.sv has zero TCU_OP references; it binds
#     VX_tcu_unit without tcu_lsu_mem_if / txbar_bus_if.
#   - Synthesizing VX_tcu_unit bare would leave its interface ports unloaded and
#     let yosys optimize away whatever is not observed, which is precisely what
#     those wrappers exist to prevent. The area number would be fiction.
# Everything outside the TCU is identical between the two builds, so the delta
# is attributable to the tensor configuration.
#
# TCU datapath is BHF for BOTH builds. TCU_OP supports only DPI or BHF
# (VX_tcu_op_core.sv:37-50) and DPI is simulation-only (dpi_fadd/dpi_fmadd in
# always_comb), so BHF is the only synthesizable choice; using it for WGMMA too
# keeps the comparison datapath-matched rather than TFR-vs-BHF. This run is also
# the first time the BHF path has been elaborated at all (review item P2.3).
#
# NOTE ON "RS": WGMMA RS vs SS is a per-instruction runtime bit (rs2[3], decoded
# to op_args.tcu.a_from_smem). Both operand paths exist in hardware regardless,
# so RS/SS does not change the synthesized netlist. Recorded here so the
# distinction is not silently dropped.
set -u
SRC=~/dev/vortex_spg
BUILD=$SRC/build
R=$SRC/eval_tcu_vs_wg/fedp2k
S=$R/status_synth.txt
: > $S

COMMON="-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=4 \
-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_TCU_TYPE=BHF \
-DVX_CFG_TCU_FP16_ENABLE"

WGMMA="$COMMON -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_TCU_FEDP2K"
TCUOP="$COMMON -DTCU_OP"

# $1=tag $2=configs
run(){
  local TAG=$1 CFG=$2
  echo "[$(date +%H:%M:%S)] START $TAG" >> $S
  local t0=$(date +%s)
  cd "$BUILD"
  # hw/syn/yosys/Makefile caches BUILD_DIR/src and does NOT regenerate it when
  # the config or include set changes (the dut/Makefile says so explicitly), so
  # a build that aborted part-way leaves an incomplete source set behind and the
  # next run fails on whatever it is missing. Always start from a clean tree.
  rm -rf "$BUILD/hw/syn/yosys/synth_${TAG}_Vortex"
  # YOSYS_FLATTEN=0: the flattened full-Vortex run was OOM-killed (SIGKILL,
  # make Error 137) after ~2h, with yosys growing from 24GB through the
  # machine's 125GB during flatten/techmap. Hierarchical synthesis keeps the
  # memory bounded, and for an A/B area comparison it is if anything better --
  # per-module area survives in the stat report instead of dissolving into one
  # flat netlist.
  timeout 36000 make -C hw/syn/yosys/dut vortex \
      PREFIX="synth_${TAG}" CONFIGS="$CFG" YOSYS_FLATTEN=0 > $R/synth_${TAG}.log 2>&1
  local rc=$? t1=$(date +%s)
  local summary="$BUILD/hw/syn/yosys/synth_${TAG}_Vortex/synth_summary.csv"
  echo "[$(date +%H:%M:%S)] DONE $TAG rc=$rc wall=$((t1-t0))s" >> $S
  if [ -f "$summary" ]; then
    echo "--- $TAG synth_summary.csv ---" >> $S
    cat "$summary" >> $S
  else
    echo "RESULT $TAG NO-SUMMARY rc=$rc" >> $S
    grep -aiE "^ERROR|error:|Killed|out of memory" $R/synth_${TAG}.log | tail -5 >> $S
  fi
}

run wgmma "$WGMMA"
run tcuop "$TCUOP"
echo "[$(date +%H:%M:%S)] SYNTH-COMPLETE" >> $S
