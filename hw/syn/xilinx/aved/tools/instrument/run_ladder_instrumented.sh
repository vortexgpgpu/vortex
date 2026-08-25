#!/bin/bash
# Run the hardware ladder under full forensic instrumentation.
#
#   bash ~/dev/v80/instrument/run_ladder_instrumented.sh
#
# NO ROOT. NO design write. NO reset. NO secondary bus reset.
#
# Every rung runs with VORTEX_AVED_NO_PROGRAM=1, so vrt::Device is constructed
# with program=false and programDevice() is never called -- which means vrtd
# never evaluates shell_reset_required() and the SBR branch is unreachable.
# The AFU is already in the fabric from jtag_load_vortex.sh.
#
# The device reset is off by default in the runtime -- writing CTL_AP_RESET is
# measurably fatal on this shell -- so nothing needs to be set here for it.
#
# EVERYTHING PERSISTS. Logs go to ~/dev/v80/logs, not /tmp: /tmp lost every
# previous run's results across the reboots that followed each crash.

set -u
LOGDIR=/home/blaise/dev/v80/logs
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
RUNLOG="$LOGDIR/ladder_$STAMP.log"
ln -sf "$RUNLOG" "$LOGDIR/ladder_latest.log"

# --- breadcrumb: fsync'd, so it survives a hard reset --------------------
crumb() {
    printf '%s' "$1" > "$LOGDIR/breadcrumb"
    printf '%s  %s\n' "$(date +%H:%M:%S)" "$1" >> "$LOGDIR/breadcrumb.log"
    # force both to stable storage before the risky operation begins
    sync "$LOGDIR/breadcrumb" "$LOGDIR/breadcrumb.log" 2>/dev/null || sync
}

exec > >(python3 /home/blaise/dev/v80/instrument/fsync_tee.py "$RUNLOG") 2>&1
echo "=========== INSTRUMENTED LADDER  ($(date)) ==========="
echo "run log : $RUNLOG   (persistent)"
echo "samples : $LOGDIR/sample.tsv"

# --- start the 1 Hz fsync'd sampler --------------------------------------
crumb "startup"
python3 /home/blaise/dev/v80/instrument/sampler.py "$LOGDIR" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; crumb "exited"' EXIT
echo "sampler pid $SAMPLER"

# --- preflight ------------------------------------------------------------
crumb "preflight"
export LD_LIBRARY_PATH=/opt/xilinx/slash/lib:${LD_LIBRARY_PATH:-}
echo
echo "--- preflight ---"
if [ ! -e /sys/bus/pci/devices/0000:01:00.0 ]; then
    echo "STOP: V80 not on the PCIe bus."; exit 1
fi
/opt/xilinx/slash/bin/v80-smi list 2>&1 | sed 's/^/  /'
before_sbr=$(journalctl -b 0 --no-pager 2>/dev/null | grep -c 'toggle_sbr')
echo "  SBRs so far this boot: $before_sbr  (must not increase)"

# --- the ladder -----------------------------------------------------------
TESTS="${TESTS:-minimal demo sgemv sgemm}"
declare -A RC
for t in $TESTS; do
    echo
    echo "############ rung: $t ############"

    if [ "$t" = "minimal" ]; then
        crumb "rung:minimal-loopback"
        echo "  (loopback: exercises the CP command path, launches no kernel)"
        MMIO="$LOGDIR/mmio_${t}_loopback.tsv"; : > "$MMIO"
        stdbuf -oL -eL env HW_TIMEOUT=600 VORTEX_CP_POLL_TIMEOUT_S=120 \
            VORTEX_AVED_NO_PROGRAM=1  \
        VORTEX_AVED_MMIO_TRACE="$MMIO" \
            bash /home/blaise/dev/v80/run_hw_test.sh "$t" OPTS="-n4 -l"
        RC["$t-l"]=$?
        echo "  == $t -l rc=${RC[$t-l]}"
        crumb "rung:minimal-loopback:done rc=${RC[$t-l]}"
        [ "${RC[$t-l]}" -ne 0 ] && { echo "  loopback failed; the core never ran, so later rungs are meaningless"; break; }
    fi

    crumb "rung:$t"
    MMIO="$LOGDIR/mmio_${t}.tsv"; : > "$MMIO"
    stdbuf -oL -eL env HW_TIMEOUT=1800 VORTEX_CP_POLL_TIMEOUT_S=300 \
        VORTEX_AVED_NO_PROGRAM=1  \
        VORTEX_AVED_MMIO_TRACE="$MMIO" \
        bash /home/blaise/dev/v80/run_hw_test.sh "$t"
    RC["$t"]=$?
    echo "  == $t rc=${RC[$t]}"
    crumb "rung:$t:done rc=${RC[$t]}"
done

# --- verdict --------------------------------------------------------------
crumb "summary"
echo
echo "=========== SUMMARY ==========="
for k in "${!RC[@]}"; do printf "  %-22s rc=%s  %s\n" "$k" "${RC[$k]}" "$([ "${RC[$k]}" -eq 0 ] && echo PASS || echo FAIL)"; done
after_sbr=$(journalctl -b 0 --no-pager 2>/dev/null | grep -c 'toggle_sbr')
echo
echo "  SBRs before=$before_sbr after=$after_sbr"
[ "$before_sbr" -eq "$after_sbr" ] \
  && echo "  *** ZERO secondary bus resets during the run -- as designed ***" \
  || echo "  !!! AN SBR OCCURRED -- the no-program path did not hold, investigate !!!"
echo
echo "  --- MMIO traces (last access before any wedge) ---"
for m in "$LOGDIR"/mmio_*.tsv; do
    [ -s "$m" ] || continue
    n=$(grep -vc '^#' "$m")
    w=$(grep -c 'CARD STOPPED ANSWERING' "$m")
    printf "    %-34s %5s records  wedge=%s\n" "$(basename "$m")" "$n" "$([ "$w" -gt 0 ] && echo YES || echo no)"
    [ "$w" -gt 0 ] && { echo "      last 5 accesses before/at the wedge:"; grep -v '^#' "$m" | tail -5 | sed 's/^/        /'; }
done
echo
echo "  persistent log: $RUNLOG"
crumb "complete"
