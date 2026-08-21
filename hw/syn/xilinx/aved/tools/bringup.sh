#!/bin/bash
# ONE post-reboot command: load the stack, then boot the compute partition.
#
#   sudo bash ~/dev/v80/bringup.sh
#
# = step1_load.sh  +  `v80-smi reset --shell-type compute`
#
# Does NOT write flash. Both partitions already hold the SLASH compute static
# shell (written 2026-08-20 14:02 partition 1, 2026-08-21 04:38 partition 0).
# Everything here is undone by a reboot.
#
# WHY THE RESET IS NEEDED EVERY BOOT
# ----------------------------------
# `Shell:` reports which PARTITION booted, not what image is in it:
# partition 0 -> "service", partition 1 -> "compute". POST picks partition 0 by
# default, so the board comes up reporting "service". Our vbin declares
# <ShellType>compute</ShellType>, and vrtd resets the board to the matching
# partition whenever they differ -- that is the SBR we kept paying on every
# design write. Booting partition 1 once, up front, makes them match, and then
# design writes run clean (measured 2026-08-20 15:11: zero reset_with_ami,
# zero toggle_sbr).
#
# This is in-band and takes ~20 s. It is NOT a host reboot and NOT JTAG.

set -u

SMI=/opt/xilinx/slash/bin/v80-smi
LOG=/tmp/v80_bringup.log
export LD_LIBRARY_PATH=/opt/xilinx/slash/lib:${LD_LIBRARY_PATH:-}
exec > >(tee -a "$LOG") 2>&1
echo "=========== BRINGUP ($(date))  log: $LOG ==========="

die() { echo; echo "FAILED: $*"; exit 1; }
[ "$(id -u)" -eq 0 ] || die "must run as root"

echo
echo "########## part 1: load the stack ##########"
bash /home/blaise/dev/v80/step1_load.sh || die "step1_load.sh failed -- see above"

echo
echo "########## part 2: boot the compute partition ##########"
PF0=$(lspci -d 10ee:50b4 -D 2>/dev/null | awk '{print $1}')
[ -n "$PF0" ] || die "PF0 (10ee:50b4) not on the bus"
[ "$(printf '%s\n' "$PF0" | wc -l)" -eq 1 ] || die "more than one V80 found: $PF0"
BDF=${PF0%.*}

cur=$("$SMI" list 2>&1)
echo "  before: $cur"
if echo "$cur" | grep -qi 'shell.*compute'; then
    echo "  already on the compute partition -- no reset needed"
else
    echo "  resetting into partition 1 (compute)..."
    timeout 300 "$SMI" reset -d "$BDF" --shell-type compute || die "v80-smi reset failed"
fi

echo
echo "########## verify ##########"
for i in 1 2 3 4 5 6; do
    out=$("$SMI" list 2>&1); echo "  attempt $i ($(date +%T)): $out"
    if echo "$out" | grep -qi 'shell.*compute'; then
        echo
        echo "=========== READY -- Shell: compute ==========="
        echo "Design writes should now run with no secondary bus reset."
        exit 0
    fi
    [ $i -lt 6 ] && sleep 10
done

die "board did not come up reporting Shell: compute. Do NOT flash anything;
    check 'v80-smi list' and the vrtd journal first."
