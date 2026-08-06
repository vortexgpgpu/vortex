#!/bin/bash
# Register (or remove) the Vortex PoCL OpenCL platform with the *system* ICD
# loader, so it is discovered by any OpenCL application and coexists with other
# installed platforms WITHOUT setting OCL_ICD_VENDORS per process.
#
# This is the portable, cross-loader registration path: it uses the standard
# /etc/OpenCL/vendors/*.icd convention honored by both ocl-icd and the Khronos
# reference loader. Because it writes /etc, it must run as root.
#
# This script is OPTIONAL and is intended for real deployments. It is NOT used
# by the test harness or CI -- those use the no-sudo, ocl-icd-specific
# OCL_ICD_VENDORS path (see tests/opencl/common.mk and tests/hip/common.mk),
# so the CI setup never requires sudo.
#
# Usage:
#   sudo POCL_PATH=$TOOLDIR/pocl ci/register_icd.sh            # install
#   sudo ci/register_icd.sh --remove                           # uninstall
#
# POCL_PATH defaults to $TOOLDIR/pocl when set.

set -e

VENDOR_DIR=/etc/OpenCL/vendors
VENDOR_FILE=$VENDOR_DIR/pocl-vortex.icd

if [ "$1" = "--remove" ]; then
    [ "$(id -u)" = 0 ] || { echo "error: removing $VENDOR_FILE requires root; re-run with sudo" >&2; exit 1; }
    rm -f "$VENDOR_FILE" && echo "removed $VENDOR_FILE"
    exit 0
fi

POCL_PATH=${POCL_PATH:-${TOOLDIR:+$TOOLDIR/pocl}}
[ -n "$POCL_PATH" ] || { echo "error: set POCL_PATH (or TOOLDIR) to the PoCL install root" >&2; exit 1; }

libpocl=$(ls "$POCL_PATH"/lib/libpocl.so.*.* 2>/dev/null | head -1)
[ -n "$libpocl" ] || { echo "error: libpocl not found under $POCL_PATH/lib -- is PoCL built ICD-only?" >&2; exit 1; }

[ "$(id -u)" = 0 ] || { echo "error: writing $VENDOR_FILE requires root; re-run with: sudo POCL_PATH=$POCL_PATH $0" >&2; exit 1; }

mkdir -p "$VENDOR_DIR"
echo "$libpocl" > "$VENDOR_FILE"
echo "registered Vortex OpenCL platform: $VENDOR_FILE -> $libpocl"
