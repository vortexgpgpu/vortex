#!/bin/bash
# Run one regression test against the AVED HARDWARE vbin on the V80.
#   usage: run_hw_test.sh <test> [make args...]
#
# Bounded by a hard timeout: a wedged AFU otherwise hangs until the board is
# JTAG-recovered, and there is no reason to discover that an hour later.
#
# SIGKILL runs no destructors, so a timed-out run leaves the CP enabled with no
# teardown at all. That is safe only because the runtime disables the queue and
# waits for the drain on the way IN as well as on the way out -- without the
# entry-side quiesce, the next run's device reset would land on a live AXI
# master and take the card off the bus.
set -o pipefail

TEST="${1:?usage: run_hw_test.sh <test> [opts]}"
shift
TIMEOUT="${HW_TIMEOUT:-300}"

source /home/blaise/dev/xilinx_setup_aved.sh >/dev/null || exit 1

VORTEX_HOME=/home/blaise/dev/vortex_gfxw_v2
BUILD=$VORTEX_HOME/build

# SLASH is installed from its Debian packages, so libvrt/libslash live on the
# default library path and the runtime resolves them with no help. Only the
# in-tree driver .so needs locating.
#
# There used to be a /opt/xilinx/slash prefix here plus a CPATH include-shim
# (~/dev/v80/inc-shim) holding symlinks for the jsoncpp/libxml2 headers VRT's
# public headers pull in transitively. The aved Makefile now asks pkg-config
# for those include paths, so both are gone.
export LD_LIBRARY_PATH="$BUILD/sw/runtime:${LD_LIBRARY_PATH:-}"

# Set VORTEX_AVED_MMIO_TRACE=<path> to record every register access; the
# transition to 0xFFFFFFFF is the last thing the card does before it drops off
# the bus. VORTEX_AVED_TRACE is a different knob and traces only the simulation
# host-memory sync, so it emits nothing here.

# Which synthesis output to run. Overridable because the HOST_TAG variants are
# separate build trees rather than rebuilds of one: build32_aved_hw is the
# original (m_axi_host -> HOST, the QDMA slave bridge, which does not work on
# this shell) and hbm1_aved_hw is the one that routes the CP's command ring to
# HBM1 instead. Keeping both on disk means the two can be compared without a
# 37-minute resynthesis, and the runtime picks its behaviour from whichever
# vbin it is handed -- staged_probe() reads the target back out of
# system_map.xml -- so this one variable selects the whole configuration.
VBIN_DIR="${VBIN_DIR:-$BUILD/hw/syn/xilinx/aved/hbm1_aved_hw/bin}"
[ -f "$VBIN_DIR/vortex_afu.vbin" ] || {
    echo "run_hw_test.sh: no vbin at $VBIN_DIR/vortex_afu.vbin" >&2
    exit 1
}
echo "run_hw_test.sh: using $VBIN_DIR/vortex_afu.vbin" >&2

cd "$BUILD/tests/regression/$TEST" || exit 1
exec timeout --signal=KILL "$TIMEOUT" \
  make run-aved TARGET=hw VRT_DEVICE_BDF=01:00 \
    FPGA_BIN_DIR="$VBIN_DIR" "$@"
