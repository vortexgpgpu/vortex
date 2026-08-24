#!/bin/bash
# ONE post-reboot command: load the stack, then boot the compute partition.
#
#   sudo bash ~/dev/v80/bringup.sh
#
# = step1_load.sh, and nothing else. It performs NO device reset: see part 2.
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
echo "########## part 2: report shell (NO RESET -- see below) ##########"
# DO NOT reset into the compute partition any more.
#
# Measured 2026-08-24: 6 of 21 host crashes died in the SAME SECOND as an
# SBR / rescan / hotplug operation, including the one that killed this box at
# 13:08:28 immediately after bringup.sh's own `v80-smi reset`. The reset is a
# crash trigger, so we now avoid it entirely.
#
# We can avoid it because BOTH flash partitions hold the identical SLASH
# compute static shell (partition 1 written 2026-08-20, partition 0 written
# 2026-08-21). vrtd resets only when the vbin's <ShellType> differs from the
# booted partition's label (reset.c: shell_reset_required). POST boots
# partition 0, which vrtd labels "service". So we relabel the vbin to
# "service" instead of forcing the board to partition 1. The bitstream is
# byte-identical -- verified with cmp -- only the label differs.
#
# Use the relabelled vbin:
#   build/hw/syn/xilinx/aved/hbm1_aved_hw_svc/bin/vortex_afu.vbin
"$SMI" list 2>&1 | sed 's/^/  /'
echo
echo "=========== READY ==========="
echo "Shell above should read 'service'. That is expected and CORRECT:"
echo "partition 0 holds the compute static shell; 'service' is just the"
echo "partition-0 label. Run the ladder against the _svc vbin so vrtd"
echo "performs NO reset and NO secondary bus reset."
exit 0
