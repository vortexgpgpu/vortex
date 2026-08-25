#!/bin/bash
# Run every regression test against the V80 and record a pass/fail table.
#
#   bash hw_sweep.sh [test ...]
#
# NO ROOT. The AFU must already be resident (jtag_load_vortex.sh) and the
# drivers loaded (slash_only_load.sh).
#
# Each test gets its own bounded timeout and its own MMIO trace, so a test that
# wedges the card is attributable to itself rather than to whatever ran next.
# The sweep stops early if the card leaves the bus -- everything after that
# point would fail for the same reason and the results would be noise.
#
# RECOVERY BETWEEN TESTS IS MANDATORY, not a nicety. A test whose kernel never
# completes leaves a command unretired in the CP, and every later device_open
# then refuses with "the CP is still busy from a previous run". Without a
# reload in between, one stalling test turns the whole remaining sweep into
# identical failures that say nothing about the tests they are attributed to.
# Reloading the AFU reconfigures the partition, which resets the CP; it costs a
# few minutes, so it runs only after a verdict that can leave the CP dirty.

set -u
LOGDIR=/home/blaise/dev/v80/logs/sweep_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.tsv"
printf "test\trc\tverdict\tcycles\tnote\n" > "$SUMMARY"

TESTS="${*:-$(ls /home/blaise/dev/vortex_gfxw_v2/tests/regression | grep -vE '^(common.mk|Makefile|run_parallel.sh.in)$')}"

card_present() { [ "$(lspci -d 10ee:50c1 -nn 2>/dev/null | wc -l)" -gt 0 ]; }

for t in $TESTS; do
    [ -d "/home/blaise/dev/vortex_gfxw_v2/build/tests/regression/$t" ] || continue
    if ! card_present; then
        printf "%s\t-\tSKIPPED\t-\tcard left the bus; sweep aborted\n" "$t" >> "$SUMMARY"
        continue
    fi
    MMIO="$LOGDIR/mmio_$t.tsv"; : > "$MMIO"
    OUT="$LOGDIR/$t.log"
    stdbuf -oL -eL env HW_TIMEOUT=300 VORTEX_CP_POLL_TIMEOUT_S=90 \
        VORTEX_AVED_NO_PROGRAM=1 VORTEX_AVED_MMIO_TRACE="$MMIO" \
        timeout 400 bash /home/blaise/dev/v80/run_hw_test.sh "$t" > "$OUT" 2>&1
    rc=$?
    cycles=$(grep -oE "cycles=[0-9]+" "$OUT" | tail -1 | cut -d= -f2)
    if grep -q "PASSED!" "$OUT"; then verdict=PASS
    elif grep -q "CP poll timed out" "$OUT"; then verdict=CP_STALL
    elif grep -q "CARD STOPPED ANSWERING" "$MMIO" 2>/dev/null; then verdict=WEDGE
    elif [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then verdict=TIMEOUT
    else verdict=FAIL; fi
    note=$(grep -oE "not supported|unsupported|Unsupported|no such file|error: [^\"]{0,40}" "$OUT" | head -1)
    if grep -q "CP is still busy from a previous run" "$OUT"; then
        verdict=POISONED
        note="previous test left the CP dirty; result not attributable"
    fi
    printf "%s\t%s\t%s\t%s\t%s\n" "$t" "$rc" "$verdict" "${cycles:--}" "${note:--}" >> "$SUMMARY"
    printf "%-22s %-9s rc=%-4s %s\n" "$t" "$verdict" "$rc" "${cycles:+cycles=$cycles}"

    # Reset the CP before the next test if this one may have left it dirty.
    case "$verdict" in
      CP_STALL|TIMEOUT|WEDGE|POISONED)
        echo "  -> reloading AFU to reset the CP"
        bash /home/blaise/dev/v80/jtag_load_vortex.sh > "$LOGDIR/reload_$t.log" 2>&1 \
          && echo "  -> AFU reloaded" \
          || echo "  -> AFU RELOAD FAILED; later results are unreliable"
        ;;
    esac
done

echo
echo "=========== SWEEP SUMMARY ==========="
awk -F'\t' 'NR>1{c[$3]++} END{for (v in c) printf "  %-10s %d\n", v, c[v]}' "$SUMMARY"
echo "  full table: $SUMMARY"
