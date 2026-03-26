#!/usr/bin/env bash
set -euo pipefail

# Optional arg: 32 or 64 to run in build32/build64.
XLEN="${1:-}"

if [[ -n "${XLEN}" ]]; then
  cd "build${XLEN}"
fi

if [[ -f ci/toolchain_env.sh ]]; then
  # Build artifacts in CI expect this environment.
  # shellcheck disable=SC1091
  source ci/toolchain_env.sh
fi

commands=(
  'make -C tests/regression/sgemm_tcu clean && CONFIGS="-DNUM_THREADS=2 -DITYPE=int8 -DOTYPE=int32" make -C tests/regression/sgemm_tcu'
  'CONFIGS="-DNUM_THREADS=2 -DEXT_TCU_ENABLE" ./ci/blackbox.sh --driver=simx --app=sgemm_tcu --debug=3 --log=run_simx.log'
)

for cmd in "${commands[@]}"; do
  echo ">>> ${cmd}"
  bash -lc "${cmd}"
done

echo "tcu_regression.sh completed successfully"
