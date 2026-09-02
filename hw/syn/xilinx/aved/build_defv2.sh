#!/usr/bin/env bash
# The GPU default configuration at 300 MHz, against the vNOC-pinned shell.
#
# Stock VX_config.toml defaults (1 cluster / 1 core / NW=4 / NT=4, L1+LMEM on,
# L2/L3 off, no extensions). No CONFIGS override -- this is the milestone after
# tinyGPU.
#
# WHY THIS IS THE REAL TEST. tinyGPU is ~3% utilization, so pinning every
# memory anchor into SLR2 cost it nothing: the logic followed onto one die and
# vnoc3 closed at exactly 300.000 MHz. The default config is substantially
# larger (NW/NT doubled, L1 + LMEM enabled). If it no longer fits comfortably
# in SLR2 the placer must spread it again and some crossings come back. That is
# the open question this build answers.
#
# Settings carried over from vnoc3, which closed 300 MHz:
#   - no pblock confinement (five builds measured it harmful)
#   - no NoC steering from the RM (HDPR-122 forbids it; fixed in the shell)
#   - post-route phys_opt, which recovered the last 34 ps on tinyGPU
set -eo pipefail
source ~/dev/xilinx_setup_aved.sh
cd ~/dev/vortex_gfxw_v2/build/hw/syn/xilinx/aved

export VX_DFX_SLR_CONFINE=0
export VX_NOC_STEER=0
export VX_POST_PHYSOPT=1

exec make all PREFIX=defv2 TARGET=hw
