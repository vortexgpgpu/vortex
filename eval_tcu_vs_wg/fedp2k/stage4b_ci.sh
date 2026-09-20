#!/bin/bash
# Stage 4b: run every case in ci/testcases/tensor_op.yaml through the real CI
# harness, so the suite is not authored on untested configurations.
#
# NOTE: blackbox.sh must run from the BUILD tree (it needs config.mk and the
# generated VX_config.toml); running it from the source tree fails instantly in
# gen_config.py with FileNotFoundError: '/VX_config.toml'.
set -u
SRC=~/dev/vortex_spg
BUILD=$SRC/build
R=$SRC/eval_tcu_vs_wg/fedp2k
S=$R/status_stage4b.txt
: > $S

# enumerate the cases from the source tree (where ci/ lives)
cd "$SRC"
python3 - <<'PY' > $R/ci_cases.txt
import sys, shlex
sys.path.insert(0, 'ci')
import testcase
for c in testcase.load_all():
    if c.category != 'tensor_op':
        continue
    argv, env = c.run_command(32)
    print(c.id + '\t' + env.get('CONFIGS','') + '\t' + ' '.join(shlex.quote(a) for a in argv))
PY

# execute them from the build tree
cd "$BUILD"
while IFS=$'\t' read -r id cfg cmd; do
  [ -z "${id:-}" ] && continue
  tag=$(echo "$id" | tr ':' '_')
  echo "[$(date +%H:%M:%S)] START $tag" >> $S
  t0=$(date +%s)
  CONFIGS="$cfg" timeout 5400 bash -c "$cmd" > $R/ci_${tag}.log 2>&1
  rc=$?; t1=$(date +%s)
  v=$(grep -aoE "PASSED|FAILED" $R/ci_${tag}.log | tail -1)
  [ $rc -eq 124 ] && v=TIMEOUT
  [ -z "$v" ] && v="ABORT(rc=$rc)"
  echo "RESULT $tag rc=$rc wall=$((t1-t0))s $v" >> $S
done < $R/ci_cases.txt
echo "[$(date +%H:%M:%S)] STAGE4B-COMPLETE" >> $S
