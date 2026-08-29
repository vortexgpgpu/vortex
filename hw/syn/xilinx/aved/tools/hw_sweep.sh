#!/bin/bash
# Run every regression test against the V80 and record a pass/fail table.
#
#   bash hw_sweep.sh [test ...]
#
# NO ROOT. The AFU must already be resident (jtag_load_vortex.sh). The SLASH
# driver autoloads from its DKMS package and vrtd is socket-activated, so no
# driver setup step is needed.
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

# Resolve the sibling tools and the tree from this script's own location. These
# used to be absolute paths into ~/dev/v80, which meant the sweep ran a stale
# private copy of run_hw_test.sh rather than the one in the tree next to it --
# so fixes to the harness silently did not apply to the sweep that uses it.
TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VORTEX_HOME="$(cd "$TOOLS_DIR/../../../../.." && pwd)"   # tools/aved/xilinx/syn/hw -> root

LOGDIR="${HW_SWEEP_LOGDIR:-$HOME/dev/v80/logs}/sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.tsv"
printf "test\trc\tverdict\tcycles\tnote\n" > "$SUMMARY"

TESTS="${*:-$(ls "$VORTEX_HOME/tests/regression" | grep -vE '^(common.mk|Makefile|run_parallel.sh.in)$')}"

# Whatever jtag_load_vortex.sh last loaded. Reloads during the sweep must put
# the SAME design back: reloading the harness default instead would swap the
# resident bitstream halfway through and attribute the remaining results to a
# design that was never under test.
RESIDENT="$(cat /tmp/v80_resident_afu.path 2>/dev/null || true)"

card_present() { [ "$(lspci -d 10ee:50c1 -nn 2>/dev/null | wc -l)" -gt 0 ]; }

for t in $TESTS; do
    [ -d "$VORTEX_HOME/build/tests/regression/$t" ] || continue
    if ! card_present; then
        printf "%s\t-\tSKIPPED\t-\tcard left the bus; sweep aborted\n" "$t" >> "$SUMMARY"
        continue
    fi
    MMIO="$LOGDIR/mmio_$t.tsv"; : > "$MMIO"
    OUT="$LOGDIR/$t.log"
    stdbuf -oL -eL env HW_TIMEOUT=300 VORTEX_CP_POLL_TIMEOUT_S=90 \
        VORTEX_AVED_NO_PROGRAM=1 VORTEX_AVED_MMIO_TRACE="$MMIO" \
        timeout 400 bash "$TOOLS_DIR/run_hw_test.sh" "$t" > "$OUT" 2>&1
    rc=$?
    cycles=$(grep -oE "cycles=[0-9]+" "$OUT" | tail -1 | cut -d= -f2)
    # Failure signatures FIRST: several apps print a summary line containing
    # "PASSED" even in runs that also report errors, and the spellings vary
    # ("PASSED!", "PASSED", "Test PASSED"). Matching success first misfiled
    # three passing tests as failures and would hide a real one just as easily.
    if grep -qE "Found [0-9]+ errors|FAILED" "$OUT"; then verdict=FAIL
    elif grep -q "CP poll timed out" "$OUT"; then verdict=CP_STALL
    elif grep -qE "\bPASSED\b" "$OUT"; then verdict=PASS
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
        # $RESIDENT, not the harness default: put back the design under test.
        bash "$TOOLS_DIR/jtag_load_vortex.sh" ${RESIDENT:+"$RESIDENT"} \
          > "$LOGDIR/reload_$t.log" 2>&1 \
          && echo "  -> AFU reloaded" \
          || echo "  -> AFU RELOAD FAILED; later results are unreliable"
        ;;
    esac
done

echo
echo "=========== SWEEP SUMMARY ==========="
awk -F'\t' 'NR>1{c[$3]++} END{for (v in c) printf "  %-10s %d\n", v, c[v]}' "$SUMMARY"
echo "  full table: $SUMMARY"
