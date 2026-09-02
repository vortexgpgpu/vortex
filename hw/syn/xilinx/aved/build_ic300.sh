#!/usr/bin/env bash
# Make the two shell SmartConnects CONTIGUOUS instead of split across the
# SLR1/SLR2 boundary.
#
# Measured on fast300 (WNS -0.260, 942 failing endpoints):
#
#   block            SLR1    SLR2    failing   worst    worst path
#   smartconnect_0   10.7%   89.3%       121   -0.260   internal: FIFO full -> its own AXI-Lite conv
#   hbm_sc_01        14.3%   85.7%       388   -0.183   internal: FIFO empty -> its own rd-addr counter
#
# Both blocks' worst paths are INSIDE themselves, and both blocks straddle the
# interposer, so those handshake loops are crossing dies inside a 579- and a
# 1775-LUT block. Confining each to the die it is already ~87% on removes the
# split. 509 of 942 failing endpoints are in scope.
#
# Direction matters: they go to SLR2, where their NoC anchors are -- NOT to
# SLR1 where the AFU is. Every HBM channel on this device is in SLR2
# (NOC_NMU_HBM2E: SLR0 0, SLR1 0, SLR2 64), so pulling them to the AFU is what
# lost three earlier builds.
#
# NOT changed here: connectivity tags, RTL, address map. Single variable
# against fast300.
#
# No `set -u`: the Vitis settings64 script reads unset vars (PYTHONPATH).
set -eo pipefail

source ~/dev/xilinx_setup_aved.sh

cd ~/dev/vortex_gfxw_v2/build/hw/syn/xilinx/aved

export VX_DFX_SLR_CONFINE=1     # enable the (default) 'ic' scope
export VX_DFX_SLR_SCOPE=ic      # interconnect blocks, each its own pblock
export VX_DFX_SLR=SLR2          # pin to the anchors' die, not the AFU's

# Byte-identical to the fast300 baseline. The three _ENABLE=0 flags are known
# no-ops -- kept only so this is the same string fast300 ran.
CONFIGS="-DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=1 -DVX_CFG_SOCKET_SIZE=1 \
-DVX_CFG_NUM_WARPS=2 -DVX_CFG_NUM_THREADS=2 \
-DVX_CFG_ICACHE_ENABLE=0 -DVX_CFG_DCACHE_ENABLE=0 -DVX_CFG_LMEM_ENABLE=0"

exec make all PREFIX=ic300 TARGET=hw CONFIGS="$CONFIGS"
