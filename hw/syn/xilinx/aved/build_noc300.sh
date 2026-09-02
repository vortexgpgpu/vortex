#!/usr/bin/env bash
# Move the RM's vNOC memory ingress onto SLR2, where all the HBM is, so the
# whole 300 MHz domain can sit on one die and stop paying ~0.456 ns of
# inter-SLR clock compensation on every path.
#
# Single variable against the fast300 baseline (WNS -0.260): the NMU sites.
# No pblock on the AFU -- five previous builds show that constraining logic
# placement makes this design worse. The anchors move; the placer follows.
#
# The hook aborts the build at ~12 min if the NoC solution turns out to be
# locked at pre-opt, so a negative result is cheap.
set -eo pipefail
source ~/dev/xilinx_setup_aved.sh
cd ~/dev/vortex_gfxw_v2/build/hw/syn/xilinx/aved

export VX_DFX_SLR_CONFINE=0     # pblocks: disproven, stay off
export VX_NOC_STEER=1
export VX_NOC_STEER_SLR=SLR2
export VX_NOC_STEER_ABORT=1

CONFIGS="-DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=1 -DVX_CFG_SOCKET_SIZE=1 \
-DVX_CFG_NUM_WARPS=2 -DVX_CFG_NUM_THREADS=2 \
-DVX_CFG_ICACHE_ENABLE=0 -DVX_CFG_DCACHE_ENABLE=0 -DVX_CFG_LMEM_ENABLE=0"

exec make all PREFIX=noc300 TARGET=hw CONFIGS="$CONFIGS"
