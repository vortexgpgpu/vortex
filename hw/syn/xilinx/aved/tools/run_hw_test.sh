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

# Which synthesis output to run -- and it must be the one actually resident in
# the fabric.
#
# The vbin is programmed on open by the runtime, so it
# is tempting to treat it as a formality. It is not: portMemoryConfig() reads
# its system_map.xml to decide whether the CP's memory is staged in HBM or
# reached through the QDMA slave bridge. Hand over a vbin from a different
# build tree and the runtime configures itself for hardware that is not there.
# This script used to default to a fixed build directory, which meant it
# quietly did that whenever a newer AFU had been loaded.
#
# jtag_load_vortex.sh records what it loaded, so use that. VBIN_DIR still
# overrides, for comparing two builds against one resident design on purpose.
STAMP=/tmp/v80_resident_afu.path
if [ -z "${VBIN_DIR:-}" ] && [ -r "$STAMP" ]; then
    VBIN_DIR="$(dirname "$(cat "$STAMP")")"
fi
if [ -z "${VBIN_DIR:-}" ]; then
    echo "run_hw_test.sh: don't know which AFU is resident." >&2
    echo "  $STAMP is missing -- the card has not been loaded this boot, or it" >&2
    echo "  was loaded by something other than jtag_load_vortex.sh." >&2
    echo "  Load it:  bash hw/syn/xilinx/aved/tools/jtag_load_vortex.sh <vbin>" >&2
    echo "  Or state it explicitly:  VBIN_DIR=<path>/bin $0 $TEST" >&2
    exit 1
fi
[ -f "$VBIN_DIR/vortex_afu.vbin" ] || {
    echo "run_hw_test.sh: no vbin at $VBIN_DIR/vortex_afu.vbin" >&2
    exit 1
}
echo "run_hw_test.sh: using $VBIN_DIR/vortex_afu.vbin" >&2

# Say out loud where m_axi_host points. A vbin tagged HOST routes the CP's
# command ring to the QDMA slave bridge, whose reads never complete on this
# shell: the CP then hangs on its first fetch with every register reading back
# armed and correct and no error bit set anywhere. The build refuses to produce
# one now, but old vbins are still on disk and this costs a few milliseconds.
HOSTMAP=$(tar xzOf "$VBIN_DIR/vortex_afu.vbin" --wildcards '*system_map.xml' 2>/dev/null \
          | grep -oE 'm_axi_host[^>]*' | head -1)
case "$HOSTMAP" in
  *HBM*) echo "run_hw_test.sh: m_axi_host -> HBM (staged CP memory)" >&2 ;;
  "")    echo "run_hw_test.sh: could not read m_axi_host mapping from the vbin" >&2 ;;
  *)     echo >&2
         echo "run_hw_test.sh: WARNING -- m_axi_host is not on HBM: $HOSTMAP" >&2
         echo "  If this is the QDMA slave bridge (HOST), the CP will hang on its" >&2
         echo "  first ring fetch and report nothing. Rebuild with HOST_TAG=HBM1." >&2
         echo >&2 ;;
esac

cd "$BUILD/tests/regression/$TEST" || exit 1
exec timeout --signal=KILL "$TIMEOUT" \
  make run-aved TARGET=hw VRT_DEVICE_BDF=01:00 \
    FPGA_BIN_DIR="$VBIN_DIR" "$@"
