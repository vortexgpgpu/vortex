#!/bin/bash
# Reconstruct exactly what the machine was doing when it died.
#
#   bash ~/dev/v80/instrument/postmortem.sh
#
# NO ROOT. Reads only. Run this as the FIRST thing after any unexpected reboot.
#
# Sources, in order of how much they survive a hard reset:
#   1. sample.tsv    -- fsync'd every second, so the last line IS the last state
#   2. breadcrumb    -- fsync'd before/after each operation
#   3. pstore/ERST   -- firmware-persistent, survives even a kernel panic
#   4. BERT          -- firmware's own record of the fatal error
#   5. journald      -- LAST, because SyncIntervalSec loses up to 5 min

set -u
LOGDIR="${LOGDIR:-/home/blaise/dev/v80/logs}"
echo "=============== V80 / orcas2 POSTMORTEM  ($(date)) ==============="
echo "logdir: $LOGDIR"

echo
echo "########## 0. MMIO TRACE -- the last register access before the wedge ##########"
# Highest-value evidence, so it leads. Every access is fsync'd, and repeated
# polls are coalesced ("xN") so a spin loop cannot push the interesting records
# out of reach. A 0xFFFFFFFF read is the PCIe completion-timeout signature.
found=0
for m in "$LOGDIR"/mmio_*.tsv; do
    [ -s "$m" ] || continue
    found=1
    n=$(grep -vc '^#' "$m"); w=$(grep -c 'CARD STOPPED ANSWERING' "$m")
    echo "  === $(basename "$m")  ($n records) ==="
    if [ "$w" -gt 0 ]; then
        echo "    *** CARD WEDGED. Context around the transition: ***"
        grep -n -B8 -A4 'CARD STOPPED ANSWERING' "$m" | sed 's/^/      /'
        echo
        echo "    -> the access on the '*** FIRST 0xFFFFFFFF' line is the first that"
        echo "       did not come back. The lines ABOVE it are what wedged the card."
    else
        echo "    no wedge recorded. Final 6 accesses:"
        grep -v '^#' "$m" | tail -6 | sed 's/^/      /'
    fi
    echo
done
[ "$found" -eq 0 ] && echo "  (no MMIO trace -- was VORTEX_AVED_MMIO_TRACE set?)"

echo
echo "########## 1. last state before death (fsync'd 1 Hz sampler) ##########"
if [ -f "$LOGDIR/sample.tsv" ]; then
    head -1 "$LOGDIR/sample.tsv" | tr '\t' '\n' | nl | sed 's/^/    /'
    echo "  --- final 8 samples ---"
    tail -8 "$LOGDIR/sample.tsv" | sed 's/^/    /'
    echo
    last=$(tail -1 "$LOGDIR/sample.tsv")
    echo "  DIED WITH breadcrumb = $(echo "$last" | cut -f4)"
    echo "  uptime at last sample = $(echo "$last" | cut -f3) s"
    echo "  V80 PF0 was           = $(echo "$last" | cut -f12)"
    echo "  link                  = $(echo "$last" | cut -f9)"
else
    echo "  (no sample.tsv -- sampler was not running)"
fi

echo
echo "########## 2. breadcrumb (operation in flight) ##########"
[ -f "$LOGDIR/breadcrumb" ] && sed 's/^/    /' "$LOGDIR/breadcrumb" || echo "  (none)"
[ -f "$LOGDIR/breadcrumb.log" ] && { echo "  --- history ---"; tail -15 "$LOGDIR/breadcrumb.log" | sed 's/^/    /'; }

echo
echo "########## 3. pstore / ERST (survives kernel panic) ##########"
if ls /sys/fs/pstore/* >/dev/null 2>&1; then
    for f in /sys/fs/pstore/*; do
        echo "  === $f ==="; head -40 "$f" 2>/dev/null | sed 's/^/    /'
    done
    echo
    echo "  NOTE: pstore records persist until deleted. After reading, clear with:"
    echo "        sudo rm /sys/fs/pstore/*"
else
    echo "  (empty -- the kernel never got far enough to dump, which is itself"
    echo "   evidence that firmware reset the box without handing control back)"
fi

echo
echo "########## 4. BERT (firmware's record of the fatal error) ##########"
if journalctl -b 0 --no-pager 2>/dev/null | grep -q '\[Hardware Error\]'; then
    journalctl -b 0 --no-pager 2>/dev/null \
      | grep -E 'fru_text|event severity|Local APIC_ID|Check Information|Error Structure|Transaction Type|Operation:|Level:|Context Corrupt' \
      | sed 's/.*\[Hardware Error\]/   /' | head -14
else
    echo "  NO error record -> the previous boot did NOT die of a machine check."
    echo "  That is a materially different failure mode; do not lump it in."
fi

echo
echo "########## 5. how the previous boot ended (journald, lossy) ##########"
echo "  --- last 6 lines ---"
journalctl -b -1 --no-pager -n 6 2>/dev/null | cut -c1-120 | sed 's/^/    /'
echo "  --- any SBR in its final 60 s? ---"
endt=$(journalctl -b -1 --no-pager -o short-iso 2>/dev/null | tail -1 | awk '{print $1}')
if [ -n "$endt" ]; then
    es=$(date -d "$endt" +%s 2>/dev/null)
    journalctl -b -1 --no-pager -o short-iso 2>/dev/null \
      | grep -E 'toggle_sbr|secondary bus reset|reset_with_ami|RESCAN|Design write' \
      | while read -r line; do
            t=$(echo "$line" | awk '{print $1}'); ts=$(date -d "$t" +%s 2>/dev/null)
            [ -n "$ts" ] && [ $((es-ts)) -le 60 ] && \
              printf "    -%02ds  %s\n" "$((es-ts))" "$(echo "$line" | cut -c30-115)"
        done
    echo "    (nothing listed = no bus reset in the final minute -> NOT the SBR failure mode)"
fi

echo
echo "########## 6. verdict ##########"
crumb=$([ -f "$LOGDIR/sample.tsv" ] && tail -1 "$LOGDIR/sample.tsv" | cut -f4 || echo unknown)
mce=$(journalctl -b 0 --no-pager 2>/dev/null | grep -c '\[Hardware Error\]')
sbr=$(journalctl -b -1 --no-pager 2>/dev/null | grep -c 'toggle_sbr')
wedge=0
for m in "$LOGDIR"/mmio_*.tsv; do
    [ -s "$m" ] || continue
    grep -q 'CARD STOPPED ANSWERING' "$m" && wedge=1
done
echo "    operation in flight : $crumb"
echo "    MCE recorded        : $([ "$mce" -gt 0 ] && echo YES || echo no)"
echo "    SBRs in that boot   : $sbr"
echo "    card wedged (MMIO)  : $([ "$wedge" -gt 0 ] && echo YES || echo no)"
# Section 0 outranks the SBR counters. A wedge recorded there means a register
# access took the card off the bus, which is a V80-coupled failure whether or
# not a bus reset was involved -- reading "no SBR" as "not our doing" once
# misfiled exactly that case.
if [ "$wedge" -gt 0 ]; then
    echo "    => V80-coupled. Section 0 names the access that wedged the card;"
    echo "       read it BEFORE trusting the SBR/MCE classification below."
elif [ "$sbr" -eq 0 ] && [ "$mce" -gt 0 ]; then
    echo "    => no wedge recorded and no SBR: host-intrinsic so far as this"
    echo "       evidence goes. Absence of a trace is not absence of a wedge --"
    echo "       confirm VORTEX_AVED_MMIO_TRACE was set for the run."
elif [ "$sbr" -gt 0 ] && [ "$mce" -gt 0 ]; then
    echo "    => check section 5: if an SBR is within 20 s, population A."
fi
