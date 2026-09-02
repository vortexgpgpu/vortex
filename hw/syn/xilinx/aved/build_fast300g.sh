#!/usr/bin/env bash
# Stripped timing-iteration config (L1 caches + LMEM off) WITH the DFX SLR
# confinement fix active. A/B partner to the fast300 baseline, which ran
# opt_design before the fix landed.
# No `set -u`: the Vitis settings64 script reads unset vars (PYTHONPATH).
set -eo pipefail

source ~/dev/xilinx_setup_aved.sh

# Builds run out-of-tree: the configured build/ tree is what carries config.mk
# (and therefore VORTEX_HOME). Running from the source dir leaves VORTEX_HOME
# empty and gen_config.py goes looking for /VX_config.toml.
cd ~/dev/vortex_gfxw_v2/build/hw/syn/xilinx/aved

CONFIGS="-DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=1 -DVX_CFG_SOCKET_SIZE=1 \
-DVX_CFG_NUM_WARPS=2 -DVX_CFG_NUM_THREADS=2 \
-DVX_CFG_ICACHE_ENABLE=0 -DVX_CFG_DCACHE_ENABLE=0 -DVX_CFG_LMEM_ENABLE=0"

# NoC-site counts across SLR1/SLR2 are a near-tie (42 vs 44), so the automatic
# pick is unreliable. Measurement decides it: the HBM NMU placed at
# NOC_NMU512_X2Y9 in SLR1, and the unconstrained baseline put the plurality of
# logic (30,829 LUT) in SLR1. Confining to SLR2 measured WNS -1.599 vs -0.260.
export VX_DFX_SLR_CONFINE=0   # confinement measured harmful; strategy override is the fix

exec make all PREFIX=fast300g TARGET=hw CONFIGS="$CONFIGS"
