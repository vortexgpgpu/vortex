#!/usr/bin/env bash
# vnoc2 + post-route physical optimization.
#
# vnoc2 closed at WNS -0.034 (297.0 MHz) with 265 failing endpoints averaging
# -13 ps, all in vortex_afu_0 and concentrated in cp_core/g_cpe[0].u_fetch.
# The worst path is a replica-to-replica hop on offset_r -- a high-fanout
# artifact, not a placement or an RTL-depth problem. Post-route phys_opt is the
# pass aimed exactly at that.
#
# Single variable against vnoc2: VX_POST_PHYSOPT. Same shell, same RTL.
set -eo pipefail
source ~/dev/xilinx_setup_aved.sh
cd ~/dev/vortex_gfxw_v2/build/hw/syn/xilinx/aved

export VX_DFX_SLR_CONFINE=0
export VX_NOC_STEER=0
export VX_POST_PHYSOPT=1

CONFIGS="-DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=1 -DVX_CFG_SOCKET_SIZE=1 \
-DVX_CFG_NUM_WARPS=2 -DVX_CFG_NUM_THREADS=2 \
-DVX_CFG_ICACHE_ENABLE=0 -DVX_CFG_DCACHE_ENABLE=0 -DVX_CFG_LMEM_ENABLE=0"

exec make all PREFIX=vnoc3 TARGET=hw CONFIGS="$CONFIGS"
