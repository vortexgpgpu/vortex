#!/bin/bash
# CI guard for the HW/SW config layering boundary.
#
# VX_config.h is the hardware *build configuration* — cache sizes, buffer
# depths, pipeline knobs, microarchitecture. It is private to the RTL and
# the simulators (hw/, sim/). No software or test file may include it.
#
# Software obtains what it needs from the right place instead:
#   - the ISA/ABI contract        -> VX_types.h
#   - device properties           -> vx_device_query() (VX_CAPS_*)
#   - build parameters            -> config.mk
#   - compile-time HW config      -> gen_config.py --cflags (build -D flags)
#
# See docs/proposals/config_hw_sw_layering_proposal.md.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# sw/ and tests/ are the software side; VX_config.h must not appear there.
hits=$(grep -rnE '#[[:space:]]*include[[:space:]]*[<"]VX_config\.h[>"]' \
         "$ROOT/sw" "$ROOT/tests" \
         --include='*.c' --include='*.cpp' --include='*.cc' \
         --include='*.h' --include='*.hpp' --include='*.S' \
         2>/dev/null || true)

if [ -n "$hits" ]; then
  echo "ERROR: software/test file(s) include VX_config.h (HW/sim-private):"
  echo
  echo "$hits"
  echo
  echo "VX_config.h is the hardware build configuration — private to RTL and"
  echo "the simulators. Use VX_types.h for the ISA/ABI contract,"
  echo "vx_device_query() for device properties, config.mk for build"
  echo "parameters, or the gen_config.py --cflags -D injection for"
  echo "compile-time hardware config."
  echo "See docs/proposals/config_hw_sw_layering_proposal.md"
  exit 1
fi

echo "check_config_boundary: OK — no sw/ or tests/ file includes VX_config.h"
