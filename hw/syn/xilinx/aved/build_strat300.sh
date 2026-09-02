#!/usr/bin/env bash
# Phase 1a: the implementation-strategy override, on its own.
#
# slash_project_build.tcl:190 forces Congestion_SSI_SpreadLogic_high on every
# build, which applies Floorplan.BalancedSLR.high (balance logic ACROSS SLRs)
# and NET_DELAY_WEIGHT low (deprioritise net delay). Both are wrong for a
# 3.5%-utilisation design, and 68% of this build's failing paths cross an SLR.
# pre_synth_hook.tcl clears them; that override has been written and validated
# against list_property_value since 2026-08-30 but has never run to completion.
#
# Deliberately ONE change against the fast300 baseline: same CONFIGS, same
# connectivity, confinement still off. fast300 closed at WNS -0.260 (278.3 MHz),
# so any delta here belongs to the strategy and nothing else.
#
# The pblock co-location half of Phase 1 is NOT in this build. It is new Tcl,
# and two builds have already been lost ~13 min in to untested Tcl; it gets
# written and dry-run while this one is in flight.
#
# No `set -u`: the Vitis settings64 script reads unset vars (PYTHONPATH).
set -eo pipefail

source ~/dev/xilinx_setup_aved.sh

# Builds run out-of-tree: the configured build/ tree carries config.mk and
# therefore VORTEX_HOME. From the source dir gen_config.py looks for
# /VX_config.toml and dies.
cd ~/dev/vortex_gfxw_v2/build/hw/syn/xilinx/aved

# SLR confinement measured harmful (free -0.260, SLR1 -0.801, SLR2 -1.599).
# Explicit rather than relying on the script's own default.
export VX_DFX_SLR_CONFINE=0

# Byte-identical to the fast300 baseline. The three _ENABLE=0 flags are known
# no-ops (gen_config.py emits a presence guard, so the correct off-switch is
# _DISABLE) -- they are kept only so this is the same string fast300 ran.
CONFIGS="-DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=1 -DVX_CFG_SOCKET_SIZE=1 \
-DVX_CFG_NUM_WARPS=2 -DVX_CFG_NUM_THREADS=2 \
-DVX_CFG_ICACHE_ENABLE=0 -DVX_CFG_DCACHE_ENABLE=0 -DVX_CFG_LMEM_ENABLE=0"

exec make all PREFIX=strat300 TARGET=hw CONFIGS="$CONFIGS"
