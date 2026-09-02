#!/usr/bin/env bash
# Rebuild the fast300 RM, byte-for-byte the same design, against the rebuilt
# static shell whose 8 hbm_vnoc_* NoC ingress units are pinned to SLR2.
#
# THE POINT. Vortex RTL is unchanged and every placement hook is OFF, so any
# difference against fast300 (WNS -0.260, 278.3 MHz) is attributable solely to
# the shell's NoC anchors moving onto the die that holds the HBM. That is the
# one variable nine previous builds could not touch, because DRC HDPR-122
# refuses to relocate a locked NoC port from inside the RM build.
#
# Requires promote_shell_vnoc2.sh to have been run first.
set -eo pipefail
source ~/dev/xilinx_setup_aved.sh
cd ~/dev/vortex_gfxw_v2/build/hw/syn/xilinx/aved

# Every placement intervention is disproven; the anchors move, the placer follows.
export VX_DFX_SLR_CONFINE=0
export VX_NOC_STEER=0

CONFIGS="-DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=1 -DVX_CFG_SOCKET_SIZE=1 \
-DVX_CFG_NUM_WARPS=2 -DVX_CFG_NUM_THREADS=2 \
-DVX_CFG_ICACHE_ENABLE=0 -DVX_CFG_DCACHE_ENABLE=0 -DVX_CFG_LMEM_ENABLE=0"

exec make all PREFIX=vnoc2 TARGET=hw CONFIGS="$CONFIGS"
