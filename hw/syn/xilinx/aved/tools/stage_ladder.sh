#!/bin/bash
# Run the regression ladder under TARGET=sim with the staged CP path FORCED ON.
#
#   bash ~/dev/v80/stage_ladder.sh
#   TESTS="demo sgemv" bash ~/dev/v80/stage_ladder.sh
#
# NO ROOT. NO BOARD. This is the only way to execute the staged CP code without
# silicon.
#
# WHY THIS EXISTS
# ---------------
# The staged path -- publish/refresh ordering, the ring exclusion, the 4096-byte
# allocator floor, and the deferred-free lifetime across cp_batch_begin/end --
# is hardware-only by construction:
#
#   TARGET=avedsim   the code is not compiled (CPP_API is defined iff !AVEDSIM)
#   TARGET=sim       staged_probe() returns early on sim_mode_
#   TARGET=hw        the only route
#
# So the riskiest code in sw/runtime/aved/vortex.cpp would otherwise execute for
# the first time on a board, where a run costs a device reset and has twice cost
# a host reset. VORTEX_AVED_FORCE_STAGE makes staging run against the simulated
# device memory instead: vrt::Buffer and sync() carry the same API on the
# simulation platform, so the ordering and lifetime rules all get exercised.
#
# Sizes are reduced from the hardware defaults (demo -n64, sgemv -n32) because
# this is xsim, not silicon. The point is to cover the code path, not to match
# the hardware workload; hw_ladder.sh uses the real sizes.

# Source the Xilinx env BEFORE `set -u` -- it references unset variables and
# would otherwise take the whole shell down with it.
source /home/blaise/dev/xilinx_setup_aved.sh >/dev/null 2>&1
set -u

export CPATH="/home/blaise/dev/v80/inc-shim:${CPATH:-}"
# run-aved does not put xsim's own runtime on the path; without this the
# simulator dies with "libxv_simulator_kernel.so: cannot open shared object".
export LD_LIBRARY_PATH="$XILINX_VIVADO/lib/lnx64.o:${LD_LIBRARY_PATH:-}"
export VORTEX_AVED_FORCE_STAGE=1
export VORTEX_CP_POLL_TIMEOUT_S=3000

VBIN=/home/blaise/dev/vortex_gfxw_v2/build/hw/syn/xilinx/aved/build32_aved_sim/bin
RT=/home/blaise/dev/vortex_gfxw_v2/build/tests/regression
LOG=/tmp/v80_stage_ladder.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "=========== staged-path ladder (TARGET=sim, FORCE_STAGE=1) $(date) ==========="
echo "log: $LOG"
[ -f "$VBIN/vortex_afu.vbin" ] || { echo "STOP: no sim vbin at $VBIN"; exit 1; }

run() {
    local test=$1 opts=$2 cap=$3
    echo
    echo "############ rung: $test $opts (cap ${cap}s) ############"
    cd "$RT/$test" || return 1
    timeout "$cap" make run-aved TARGET=sim VRT_HOME=/opt/xilinx/slash \
        FPGA_BIN_DIR="$VBIN" OPTS="$opts" 2>&1 \
      | grep -vE '^(make|CONFIGS=|SCOPE_JSON_PATH)' \
      | tail -25
    local rc=${PIPESTATUS[0]}
    echo "  >>> $test rc=$rc"
    return $rc
}

declare -A RC
TESTS="${TESTS:-demo sgemv sgemm}"
for t in $TESTS; do
    case $t in
        demo)  run demo  "-n16" 2400 ;;
        sgemv) run sgemv "-n16" 2400 ;;
        sgemm) run sgemm "-n8"  3000 ;;
        *)     run "$t"  "-n4"  2400 ;;
    esac
    RC[$t]=$?
done

echo
echo "=========== SUMMARY ==========="
fail=0
for t in $TESTS; do
    printf "  %-8s rc=%-3s %s\n" "$t" "${RC[$t]}" \
        "$([ "${RC[$t]}" -eq 0 ] && echo PASS || echo FAIL)"
    [ "${RC[$t]}" -eq 0 ] || fail=1
done
exit $fail
