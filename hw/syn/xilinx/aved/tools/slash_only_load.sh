#!/bin/bash
# Minimal-exposure V80 bring-up: load ONLY slash.ko, never ami.ko.
#
#   sudo bash ~/dev/v80/slash_only_load.sh
#
# WHY NO ami.ko
# -------------
# Both host hard-resets whose trigger we can name involve ami:
#   * 2026-08-19 09:40  vrtd's reset_with_ami -> TOGGLE_SBR -> dead mid-sequence
#   * 2026-08-19 15:15  box idle for 2m46s with ami+slash loaded -> dead.
#                       BERT: fatal, Perr S0:T002:B05, cache error, APIC 0x2.
#                       No rescan, no SBR, no design write in flight -- the only
#                       thing still touching the card was ami heartbeating the
#                       AMC over GCQ.
#
# VRT does not need ami. vrtd discovers boards from /dev/slash_ctl* (PF2) and
# pairs /dev/slash_qdma_ctl* (PF1); PF0/ami carries sensors, identity and the
# PDI design-writer only. With the design already resident (JTAG-loaded) and
# VORTEX_AVED_NO_PROGRAM=1, none of that is used.
#
# So this drops the AMC heartbeat entirely, and pairing it with NO_PROGRAM drops
# the SBR path too -- removing both known crash mechanisms.
#
# `v80-smi list` will report PF0 NOT READY. That is expected and harmless: it is
# a readiness *report*, not a requirement. What matters is PF1, PF2 and VRTD.
#
# Writes no flash, changes no persistent config. Undone by a reboot.

set -u

SLASH_KO=/home/blaise/dev/SLASH-compute/driver/slash.ko
SMI=/opt/xilinx/slash/bin/v80-smi
LOG=/tmp/v80_slash_only.log

export LD_LIBRARY_PATH=/opt/xilinx/slash/lib:${LD_LIBRARY_PATH:-}
exec > >(tee -a "$LOG") 2>&1
echo "=========== slash-only load ($(date)) -- log: $LOG ==========="

die()  { echo; echo "FAILED: $*"; exit 1; }
step() { echo; echo "--- $* ---"; }
[ "$(id -u)" -eq 0 ] || die "must run as root"
[ -e "$SLASH_KO" ]   || die "missing $SLASH_KO"

step "1. PCIe functions (expect 50c1 + 50c2; 50b4 may be present but is unused)"
lspci -d 10ee: -nn || die "no 10ee devices -- card is not on the bus, reboot first"
lspci -d 10ee:50c1 -nn | grep -q . || die "PF1 (10ee:50c1) absent -- not the compute shell"
lspci -d 10ee:50c2 -nn | grep -q . || die "PF2 (10ee:50c2) absent"

step "2. confirm ami is NOT loaded (that is the point of this script)"
if lsmod | grep -q '^ami '; then
    echo "  ami IS loaded -- removing it to drop the AMC heartbeat"
    systemctl stop vrtd.service 2>/dev/null
    rmmod ami || die "could not remove ami"
else
    echo "  ami not loaded (good)"
fi

step "3. load slash (PF1 QDMA + PF2 control)"
if lsmod | grep -q '^slash '; then
    echo "  slash already loaded"
else
    insmod "$SLASH_KO" || die "insmod slash.ko"
fi
sleep 3

step "4. bindings and device nodes"
for f in 1 2; do
    drv=$(readlink "/sys/bus/pci/devices/0000:01:00.$f/driver" 2>/dev/null)
    printf "  0000:01:00.%s -> %s\n" "$f" "${drv:+$(basename "$drv")}${drv:-(none)}"
done
ls -l /dev/slash_* 2>/dev/null || die "no /dev/slash_* nodes -- slash did not bind"

pf_drv() { basename "$(readlink "/sys/bus/pci/devices/0000:01:00.$1/driver" 2>/dev/null)" 2>/dev/null; }
[ "$(pf_drv 1)" = "slash_qdma" ] || die "PF1 did not bind to slash_qdma.
    'already in_use (number=0)' in dmesg is the driver's leaked qdma id:
    rmmod slash && insmod $SLASH_KO"
[ "$(pf_drv 2)" = "slash_ctl" ]   || die "PF2 did not bind to slash_ctl"

step "5. (re)start vrtd -- RESTART, not start"
# vrtd enumerates /dev/slash_ctl* once at startup and never rescans, so a vrtd
# already running from before the module loaded reports 0 devices forever.
systemctl reset-failed vrtd.service 2>/dev/null
systemctl start vrtd.socket   || die "vrtd.socket"
systemctl restart vrtd.service || { journalctl -u vrtd.service -n 25 --no-pager; die "vrtd.service"; }
sleep 3
systemctl is-active --quiet vrtd.service || die "vrtd not active"

# Check THIS invocation only. Grepping the last N journal lines for
# "Discovered 0 device(s)" matches stale lines from a previous vrtd run and
# reports a false failure -- that is exactly what v80_load.sh did at 15:12.
if ! journalctl -u vrtd.service --since "-30 seconds" --no-pager 2>/dev/null \
        | grep -qE "Discovered [1-9][0-9]* device\(s\)$"; then
    journalctl -u vrtd.service --since "-30 seconds" --no-pager | tail -10
    die "vrtd did not discover any device in this invocation"
fi
echo "  vrtd discovered the board"

step "6. readiness (PF0 NOT READY is EXPECTED here -- ami is deliberately absent)"
timeout 60 "$SMI" list 2>&1 | tail -3

cat <<'EOF'

=========== ready ===========
ami is NOT loaded, so nothing is heartbeating the AMC.
Run the ladder with the design already resident and programming disabled:

    bash ~/dev/v80/hw_ladder_noprogram.sh

EOF
