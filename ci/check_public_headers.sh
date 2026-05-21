#!/bin/bash
# Guard: the public runtime headers must stay independent of the Vortex
# build configuration. No header under sw/runtime/include/ may include
# VX_config.h (directly or transitively) — otherwise every host TU that
# uses the runtime inherits the build-internal VX_CFG_*/VX_DBG_* macros.
# See docs/proposals/config_macro_namespace_proposal.md.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INC_DIR="$ROOT/sw/runtime/include"

hits=$(grep -rnE '#[[:space:]]*include[[:space:]]*[<"]VX_config\.' "$INC_DIR" || true)
if [ -n "$hits" ]; then
  echo "ERROR: a public runtime header includes a build-config header:"
  echo "$hits"
  echo
  echo "Public headers (sw/runtime/include/) must not depend on VX_config.h."
  echo "Hardware configuration is discovered at runtime via vx_device_query()."
  exit 1
fi

echo "check_public_headers: OK — no public header reaches VX_config.h"
