#!/bin/bash
# Run the full TARGET=hw regression ladder against ONE device reset.
#
#   bash ~/dev/v80/hw_ladder.sh                 # default ladder
#   TESTS="minimal demo sgemm" bash ~/dev/v80/hw_ladder.sh
#
# NO ROOT. Run it as blaise, on a board that is already healthy (AMC READY).
#
# WHY A LADDER IS POSSIBLE NOW
# ----------------------------
# It used to cost one sudo recovery per test. Every vrt::Device open reprograms
# the PL, and a design write only succeeds on a freshly reset device -- vrtd
# runs reset_with_ami only when the requested shell differs from the current
# one, so once the shell reads "compute" the load fails with
# "Input/output error" and takes the AMC to NO_AMC, needing JTAG to recover.
#
# So: program ONCE, on the first rung, then set VORTEX_AVED_NO_PROGRAM=1 for
# every rung after it. The design stays resident, so the later rungs reuse it
# and never touch the design writer.
#
# ORDER MATTERS. Rungs run cheapest-first so a break is localised: -l exercises
# the CP command path (MEM_WRITE/MEM_READ/seqnum) without launching a kernel,
# so if it fails the core never ran and nothing after it would mean anything.

set -u

TESTS="${TESTS:-minimal demo sgemv sgemm}"
LOG=/tmp/v80_hw_ladder.log
: > "$LOG"

exec > >(tee -a "$LOG") 2>&1
echo "=========== V80 hw ladder ($(date)) -- log: $LOG ==========="

export LD_LIBRARY_PATH=/opt/xilinx/slash/lib:${LD_LIBRARY_PATH:-}
if ! timeout 60 /opt/xilinx/slash/bin/ami_tool overview 2>&1 | grep -q READY; then
    echo "STOP: AMC is not READY, so the first rung cannot load its PDI."
    echo "Recover first:  sudo bash ~/dev/v80/v80_oneshot.sh"
    exit 1
fi
timeout 60 /opt/xilinx/slash/bin/v80-smi list 2>&1 | tail -2

declare -A RC
first=1
for t in $TESTS; do
    echo
    echo "############ rung: $t ############"
    # PROGRAM_FIRST=0 when the vortex design is ALREADY resident from a previous
    # run -- then not even the first rung needs the design writer, which is the
    # safest way to run: a load that fails takes the AMC to NO_AMC and costs a
    # JTAG recovery, and there is nothing to gain from reloading a design that
    # is already there.
    if [ "$first" = "1" ] && [ "${PROGRAM_FIRST:-1}" != "0" ]; then
        # The one rung that programs. -l launches no kernel, so it isolates the
        # CP command path from anything the core does.
        echo "  (programs the PDI; loopback only)"
        HW_TIMEOUT=600 VORTEX_CP_POLL_TIMEOUT_S=60 \
            bash /home/blaise/dev/v80/run_hw_test.sh "$t" OPTS="-n4 -l"
        RC["$t-loopback"]=$?
        first=0
        echo "  $t -l rc=${RC[$t-loopback]}"
        if [ "${RC[$t-loopback]}" -ne 0 ]; then
            echo "STOP: the CP command path failed; later rungs would be noise."
            break
        fi
    fi
    HW_TIMEOUT=900 VORTEX_CP_POLL_TIMEOUT_S=60 VORTEX_AVED_NO_PROGRAM=1 \
        bash /home/blaise/dev/v80/run_hw_test.sh "$t"
    RC["$t"]=$?
    echo "  $t rc=${RC[$t]}"
done

echo
echo "=========== SUMMARY ==========="
fail=0
for k in "${!RC[@]}"; do
    printf "  %-20s rc=%s%s\n" "$k" "${RC[$k]}" \
        "$([ "${RC[$k]}" -eq 0 ] && echo "  PASS" || echo "  FAIL")"
    [ "${RC[$k]}" -eq 0 ] || fail=1
done
echo
echo "Cross-check: sgemm must report instrs=336912 -- the same count sim and"
echo "avedsim produce. A different count means the hardware executed a"
echo "different program, which a PASSED line alone would not catch."
exit $fail
