#!/bin/bash
# Install the Vortex gem5 SimObject sources into a pinned gem5 tree.
#
# Copies vortex_gpgpu_dev.{cc,hh}, VortexGPGPU.py, and SConscript into
# $GEM5_HOME/src/dev/vortex/ so gem5's scons can build them. The
# source-of-truth lives in the Vortex tree (this directory); any
# change has to re-run this script before `scons build/<ISA>/gem5.opt`
# picks it up.
#
# Idempotent: re-running just refreshes the files.
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
