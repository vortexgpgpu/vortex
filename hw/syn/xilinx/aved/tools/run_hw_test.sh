#!/bin/bash
# Run one regression test against the AVED HARDWARE vbin on the V80.
#   usage: run_hw_test.sh <test> [make args...]
#
# Bounded by a hard timeout: a wedged AFU otherwise hangs until the board is
# JTAG-recovered, and there is no reason to discover that an hour later.
set -o pipefail

TEST="${1:?usage: run_hw_test.sh <test> [opts]}"
shift
TIMEOUT="${HW_TIMEOUT:-300}"

source /home/blaise/dev/xilinx_setup_aved.sh >/dev/null || exit 1

VORTEX_HOME=/home/blaise/dev/vortex_gfxw_v2
BUILD=$VORTEX_HOME/build

export LD_LIBRARY_PATH="/opt/xilinx/slash/lib:$BUILD/sw/runtime:$LD_LIBRARY_PATH"
# VRT's public headers pull in zmq/CLI/inih/jsoncpp/libxml2 transitively; the
# shim holds symlinks to just those, so the no-root prefix cannot also shadow
# the installed vrt/ and vrtd/ trees.
export CPATH="/home/blaise/dev/v80/inc-shim:$CPATH"
export VORTEX_AVED_TRACE=1

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
  make run-aved TARGET=hw VRT_HOME=/opt/xilinx/slash VRT_DEVICE_BDF=01:00 \
    FPGA_BIN_DIR="$VBIN_DIR" "$@"
