#!/bin/bash
# Run the full TARGET=hw ladder against a design that is ALREADY resident,
# with programming disabled from the very first rung.
#
#   bash ~/dev/v80/hw_ladder_noprogram.sh
#   TESTS="minimal demo" bash ~/dev/v80/hw_ladder_noprogram.sh
#
# NO ROOT. Assumes:
#   1. The Vortex partial PDI was loaded over JTAG (jtag_load_vortex.sh), and
#   2. slash.ko is loaded WITHOUT ami.ko (slash_only_load.sh).
#
# WHY NO PROGRAMMING AT ALL
# -------------------------
# hw_ladder.sh programs on its first rung. That is what triggers vrtd's design
# writer, which runs reset_with_ami when the requested shell differs from the
# current one, which toggles a secondary bus reset -- and an SBR on root port
# 0000:00:01.1 hard-reset this host on 2026-08-19 09:40.
#
# Loading the design over JTAG instead removes that entirely: the fabric already
# holds the AFU, so every rung can run with VORTEX_AVED_NO_PROGRAM=1 and the
# design writer is never invoked. Combined with ami being absent, both known
# crash mechanisms are out of the loop.
#
# ORDER MATTERS: cheapest first, so a break is localised. -l exercises the CP
# command path (MEM_WRITE/MEM_READ/seqnum + the staged publish/refresh) without
# launching a kernel; if that fails the core never ran and later rungs are noise.

source /home/blaise/dev/xilinx_setup_aved.sh >/dev/null 2>&1
set -u

TESTS="${TESTS:-minimal demo sgemv sgemm}"
LOG=/tmp/v80_hw_ladder_noprog.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
echo "=========== V80 hw ladder, no-program ($(date)) -- log: $LOG ==========="

export LD_LIBRARY_PATH=/opt/xilinx/slash/lib:${LD_LIBRARY_PATH:-}

if lsmod | grep -q '^ami '; then
    echo "WARNING: ami is loaded. It heartbeats the AMC over GCQ, which is the"
    echo "         suspected trigger of the 2026-08-19 15:15 idle crash."
    echo "         Prefer: sudo bash ~/dev/v80/slash_only_load.sh"
fi
timeout 60 /opt/xilinx/slash/bin/v80-smi list 2>&1 | tail -2

declare -A RC
for t in $TESTS; do
    echo
    echo "############ rung: $t ############"
    if [ "$t" = "minimal" ]; then
        # -l first: CP command path only, no kernel launch.
        HW_TIMEOUT=600 VORTEX_CP_POLL_TIMEOUT_S=120 VORTEX_AVED_NO_PROGRAM=1 \
            bash /home/blaise/dev/v80/run_hw_test.sh minimal OPTS="-n4 -l"
        RC[minimal-loopback]=$?
        echo "  minimal -l rc=${RC[minimal-loopback]}"
        if [ "${RC[minimal-loopback]}" -ne 0 ]; then
            echo "STOP: the CP command path failed; later rungs would be noise."
            break
        fi
    fi
    HW_TIMEOUT=1800 VORTEX_CP_POLL_TIMEOUT_S=300 VORTEX_AVED_NO_PROGRAM=1 \
        bash /home/blaise/dev/v80/run_hw_test.sh "$t"
    RC[$t]=$?
    echo "  $t rc=${RC[$t]}"
done

echo
echo "=========== SUMMARY ==========="
fail=0
for k in "${!RC[@]}"; do
    printf "  %-20s rc=%-3s %s\n" "$k" "${RC[$k]}" \
        "$([ "${RC[$k]}" -eq 0 ] && echo PASS || echo FAIL)"
    [ "${RC[$k]}" -eq 0 ] || fail=1
done
echo
echo "Cross-check: sgemm must report instrs=336912 -- the same count sim and"
echo "avedsim produce. A different count means the hardware executed a"
echo "different program, which a PASSED line alone would not catch."
exit $fail
