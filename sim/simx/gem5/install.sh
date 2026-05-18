#!/bin/bash
# Install Vortex gem5 SimObjects into a pinned gem5 tree.
#
# Phase 3+: installs the real VortexGPGPU device. The Phase-0 dummy/
# scaffolding is intentionally removed from $GEM5_HOME during the
# transition — its job (proving the install path works) is done.
#
# Idempotent: re-running just refreshes the files. Caller must
# re-run `scons build/{X86,ARM}/gem5.opt` after this script to pick
# up changes.
#
# Usage:
#   GEM5_HOME=$HOME/tools/gem5 sim/simx/gem5/install.sh
# or
#   sim/simx/gem5/install.sh           # uses $GEM5_HOME from env

set -e

GEM5_HOME=${GEM5_HOME:-$HOME/tools/gem5}
SOURCE_DIR=$(dirname "$(readlink -f "$0")")

if [ ! -d "$GEM5_HOME/src/dev" ]; then
    echo "ERROR: GEM5_HOME=$GEM5_HOME does not look like a gem5 tree" >&2
    echo "       (expected $GEM5_HOME/src/dev/)" >&2
    exit 1
fi

DEST_DIR="$GEM5_HOME/src/dev/vortex"
mkdir -p "$DEST_DIR"

# Phase 0 scaffolding cleanup: the dummy SimObject existed only to
# prove the install path; remove it now that the real device is in
# place so `gem5.opt --list-sim-objects` is not polluted by it.
if [ -d "$DEST_DIR/dummy" ]; then
    rm -rf "$DEST_DIR/dummy"
fi

# Install the real device: header, source, Python binding, SConscript.
install -m 0644 "$SOURCE_DIR/vortex_gpgpu_dev.hh" "$DEST_DIR/"
install -m 0644 "$SOURCE_DIR/vortex_gpgpu_dev.cc" "$DEST_DIR/"
install -m 0644 "$SOURCE_DIR/VortexGPGPU.py"      "$DEST_DIR/"
install -m 0644 "$SOURCE_DIR/SConscript"          "$DEST_DIR/"

echo "Vortex SimObjects installed at $DEST_DIR"
echo "Files:"
ls -1 "$DEST_DIR" | sed 's/^/  /'
echo ""
echo "Re-build gem5 with one or both of:"
echo "  scons -C $GEM5_HOME build/X86/gem5.opt -j\$(nproc)"
echo "  scons -C $GEM5_HOME build/ARM/gem5.opt -j\$(nproc)"
