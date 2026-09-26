#!/bin/bash
# Refuses to let a run program an Alveo V80 that is not in a state to take it.
#
# Every host reset seen with this card followed one of these: a design write to
# a card that was off the bus or unhealthy, a write while another process still
# had it open, or two writes back to back. Run before anything that programs the
# card (run-firesim TARGET=hw does); it exits non-zero, saying why, instead.
#
#   VRT_DEVICE_BDF   the card to check (bus:dev, e.g. 01:00); found on the bus if unset
#   V80_COOLDOWN_S   seconds to wait after the last design write (default 60)

set -u
COOLDOWN=${V80_COOLDOWN_S:-60}

fail() { echo "v80_preflight: $*" >&2; exit 3; }

# The card: PF0 of a V80 (10ee:50b4), or the one named.
BDF=${VRT_DEVICE_BDF:-}
if [ -z "$BDF" ]; then
    BDF=$(lspci -D -d 10ee:50b4 2>/dev/null | head -1 | cut -d' ' -f1)
    [ -n "$BDF" ] || fail "no V80 (10ee:50b4) on the PCIe bus; if it dropped off, the host needs a cold power cycle"
    BDF=${BDF%.*}
fi
case "$BDF" in *:*:*) ;; *) BDF="0000:$BDF" ;; esac

# All three physical functions, and the port above them.
NFUNC=$(lspci -D -s "$BDF" 2>/dev/null | wc -l)
[ "$NFUNC" -eq 3 ] || fail "$BDF shows $NFUNC of its 3 functions on the bus"
PORT=$(basename "$(dirname "$(readlink -f /sys/bus/pci/devices/$BDF.0)")")
[ -e "/sys/bus/pci/devices/$PORT" ] || fail "the root port above $BDF ($PORT) is not present"

# The board as the shell and vrtd see it.
SHORT=${BDF#0000:}
LIST=$(timeout 20 v80-smi list 2>&1)
echo "$LIST" | grep -q "Board 0000:$SHORT OK (PF0: OK) (PF1: OK) (PF2: OK) (VRTD: OK)" \
    || fail "v80-smi does not report $BDF healthy:
$LIST"

# Nobody else holding a VRT session: a design write under a live user is what
# takes the card down.
USERS=""
for m in /proc/[0-9]*/maps; do
    pid=${m#/proc/}; pid=${pid%/maps}
    [ "$pid" = "$$" ] && continue
    grep -q 'libvrt\.so' "$m" 2>/dev/null && USERS="$USERS $pid"
done
[ -z "$USERS" ] || fail "processes still hold VRT sessions:$(ps -o pid=,etime=,args= -p ${USERS// /,} | sed 's/^/\n  /')"

# A cool-down since the last design write. The journal is the only record of
# it; a user who cannot read vrtd's entries there cannot have the wait judged,
# which is reported rather than passed off as "no recent write".
JOURNAL=$(journalctl -t vrtd --since "-$((COOLDOWN + 5))s" -o short-unix --no-pager 2>&1)
if echo "$JOURNAL" | grep -qiE "insufficient permissions|no journal files"; then
    echo "v80_preflight: cannot read the vrtd journal; not enforcing the ${COOLDOWN}s cool-down" >&2
else
    LAST=$(echo "$JOURNAL" | grep "Design write completed" | tail -1 | cut -d' ' -f1 | cut -d. -f1)
    if [ -n "$LAST" ]; then
        WAIT=$(( LAST + COOLDOWN - $(date +%s) ))
        if [ "$WAIT" -gt 0 ]; then
            echo "v80_preflight: last design write was $(( COOLDOWN - WAIT ))s ago; waiting ${WAIT}s"
            sleep "$WAIT"
        fi
    fi
fi

echo "v80_preflight: $BDF ready (port $PORT)"
