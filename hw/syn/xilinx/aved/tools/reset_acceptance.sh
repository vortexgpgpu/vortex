#!/bin/bash
# The acceptance test for the AFU soft reset.
#
#   bash reset_acceptance.sh [test]        # default: minimal
#
# NO ROOT. The AFU must already be resident (jtag_load_vortex.sh).
#
# WHAT IS BEING ACCEPTED
# ----------------------
# Running the same test twice in one boot, with no JTAG reload in between.
#
# That is the whole point. Before the demux fix, run 1 succeeded and run 2
# killed the card: the AFU's AXI-Lite demux routed a write-data beat by a
# register that only updated at the write-address handshake, so the second
# process's CTL_AP_RESET had its AW delivered to VX_afu_ctrl and its W to the
# CP regfile. No BRESP was produced, the shell's AXI-Lite master stalled, and
# every later read died of a PCIe completion timeout -- 0xFFFFFFFF forever.
# See docs/proposals/afu_reset_architecture_proposal.md.
#
# A passing run 1 therefore proves nothing at all, which is exactly why this
# script insists on two and inspects the register trace of each rather than
# trusting an exit code.
#
# WHAT IS CHECKED, PER RUN
#   1. the test itself passed
#   2. CTL_AP_RESET was actually issued (write 0x10 to 0x0000)
#   3. the sequencer honoured it -- ap_idle set, CTL_RESET_ERROR clear
#   4. the card never returned the 0xFFFFFFFF no-completion signature
#
# Check 3 is the one that distinguishes "the reset worked" from "the reset was
# refused and we carried on anyway": VX_afu_reset_seq declines to reset a
# master that will not drain and reports it in bit 5, and a refused reset is a
# failure of this test even if the test binary passes.

set -u

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST="${1:-minimal}"

CTL_ADDR=0x00000000
CTL_AP_RESET=$((1 << 4))
CTL_AP_IDLE=$((1 << 2))
CTL_RESET_ERROR=$((1 << 5))

STAMP=/tmp/v80_resident_afu.path
if [ ! -r "$STAMP" ]; then
    echo "reset_acceptance.sh: no AFU recorded as resident ($STAMP)." >&2
    echo "  Load one first:  bash $TOOLS_DIR/jtag_load_vortex.sh <vbin>" >&2
    exit 1
fi
echo "resident AFU: $(cat "$STAMP")"
echo "test        : $TEST"
echo

LOGDIR="${RESET_ACCEPT_LOGDIR:-$HOME/dev/v80/logs}/reset_accept_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

# Inspect one run's register trace. Returns 0 only if the reset was issued AND
# honoured AND the card stayed on the bus.
check_trace() {
    local mmio="$1" run="$2" ok=0

    if [ ! -s "$mmio" ]; then
        echo "    trace  : MISSING -- cannot verify the reset"
        return 1
    fi

    # Walk the trace for the CTL_AP_RESET write and the next read of the
    # control register. Done in bash rather than awk on purpose: strtonum()
    # and and() are gawk extensions, and this machine's awk is mawk, where
    # they are absent -- the check would have quietly evaluated to nothing.
    local issued=0 verdict="no CTL_AP_RESET write in the trace"
    local t op addr val
    while IFS=$'\t' read -r t op addr val _; do
        case "$t" in \#*|"") continue ;; esac
        case "$addr" in 0x*) ;; *) continue ;; esac
        case "$val"  in 0x*) ;; *) continue ;; esac
        (( addr_n = addr, val_n = val, ctl_n = CTL_ADDR ))
        if [ "$issued" -eq 0 ]; then
            if [ "$op" = write ] && [ "$addr_n" -eq "$ctl_n" ] \
               && (( val_n & CTL_AP_RESET )); then
                issued=1; verdict="reset issued, never read back"
            fi
            continue
        fi
        if [ "$op" = read ] && [ "$addr_n" -eq "$ctl_n" ]; then
            # Follow the driver's poll loop rather than judging on its first
            # sample. The reset takes several cycles, so the first read is
            # routinely 0x00000000 -- busy, verdict not yet available. Stopping
            # there reports a refusal as merely inconclusive: in the trace that
            # motivated this, read 1 was 0x00000000 and read 2 was 0x00000024,
            # which is CTL_RESET_ERROR | ap_idle and is the actual answer.
            #
            # All-ones is checked first: it is the PCIe no-completion
            # signature, not a register value. Every status bit reads set in
            # it, CTL_RESET_ERROR included, so decoding it as a refusal would
            # report the classic wedge as an orderly refusal -- the opposite
            # of the truth.
            if [ "$val" = 0xffffffff ]; then
                verdict="CARD STOPPED ANSWERING at the reset readback (0xffffffff)"
                break
            elif (( val_n & CTL_RESET_ERROR )); then
                verdict=$(printf "REFUSED -- CTL_RESET_ERROR set (ap_ctrl=0x%08x)" "$val_n")
                break
            elif (( val_n & CTL_AP_IDLE )); then
                verdict=$(printf "honoured -- ap_idle set, no error (ap_ctrl=0x%08x)" "$val_n")
                ok=1
                break
            else
                # Busy. Keep polling with the driver.
                verdict=$(printf "never resolved; last ap_ctrl=0x%08x" "$val_n")
            fi
        fi
    done < "$mmio"

    printf "%s\t%s\n" "$([ "$ok" -eq 1 ] && echo PASS || echo FAIL)" "$verdict" \
        > "$LOGDIR/verdict_$run.txt"
    echo "    reset  : $verdict"

    # The wedge signature. A single all-ones read is the card refusing to
    # answer; it is never a legitimate register value here.
    local ones; ones=$(awk -F'\t' '$2=="read" && $4=="0xffffffff"' "$mmio" | wc -l)
    if [ "$ones" -gt 0 ]; then
        echo "    bus    : CARD STOPPED ANSWERING -- $ones all-ones reads"
        ok=0
    else
        echo "    bus    : healthy ($(grep -vc '^#' "$mmio") accesses, no all-ones reads)"
    fi

    [ "$ok" -eq 1 ]
}

overall=0
for run in 1 2; do
    OUT="$LOGDIR/run$run.log"
    MMIO="$LOGDIR/mmio_run$run.tsv"; : > "$MMIO"

    echo "--- run $run ---"
    stdbuf -oL -eL env VORTEX_AVED_MMIO_TRACE="$MMIO" HW_TIMEOUT=300 \
        bash "$TOOLS_DIR/run_hw_test.sh" "$TEST" > "$OUT" 2>&1
    rc=$?

    if grep -qE "Found [0-9]+ errors|FAILED" "$OUT"; then
        echo "    test   : FAILED (rc=$rc)"; overall=1
    elif grep -qE "\bPASSED\b" "$OUT"; then
        echo "    test   : passed (rc=$rc)"
    else
        echo "    test   : no verdict (rc=$rc) -- see $OUT"; overall=1
    fi

    check_trace "$MMIO" "$run" || overall=1
    echo
done

echo "======================================================="
if [ "$overall" -eq 0 ]; then
    echo "ACCEPTED: $TEST ran twice in one boot, both resets honoured,"
    echo "          card healthy throughout. No JTAG reload in between."
else
    echo "NOT ACCEPTED -- see $LOGDIR"
fi
echo "logs: $LOGDIR"
exit "$overall"
