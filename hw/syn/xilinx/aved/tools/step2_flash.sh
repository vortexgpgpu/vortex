#!/bin/bash
# STEP 2 of the supported-path bring-up: write the SLASH compute static shell
# into the V80's OSPI flash. THIS IS THE STEP THAT WAS NEVER PERFORMED.
#
#   sudo bash ~/dev/v80/step2_flash.sh
#
# *** THIS WRITES NON-VOLATILE FLASH. *** It is the only irreversible action in
# the whole plan. Do not interrupt it, do not run anything else on the box
# while it runs, do not let the machine sleep.
#
# WHY THE COMMAND IS NOT THE ONE IN THE DOCS
# ------------------------------------------
# The documented form is:
#     sudo v80-smi write-static-shell --flash -d <BDF>
# which defaults to --shell-type all and resolves the PDI via
# `python3 -m slashkit static-shell-path`. On this install that fails twice:
#   * slashkit's resolver accepts only {service,compute} -- "all" is an
#     argparse error;
#   * --shell-type service resolves slashkit.resources.static_shell, which is
#     an EMPTY directory in this tree (only static_shell_compute is populated).
# So we pin both axes explicitly: --shell-type compute and an explicit --pdi.
# That writes boot partition 1 (compute) with the compute static shell, which
# is self-consistent and is the shell our vbin declares
# (<ShellType>compute</ShellType> in system_map.xml).
#
# RECOVERY IF THIS FAILS MIDWAY
# -----------------------------
# A failed flash write can leave the card unable to boot from OSPI. That is
# recoverable: the JTAG bootstrap does not read flash at all.
#     bash ~/dev/v80/jtag_load_shell.sh   (no sudo)
#     sudo reboot
# then re-run this script. The card is not permanently bricked by a failed
# write; only a failure that also loses JTAG would be, and JTAG is independent.

set -u

SMI=/opt/xilinx/slash/bin/v80-smi
PDI=/home/blaise/dev/SLASH-compute/linker/slashkit/resources/static_shell_compute/amd_v80_gen5x8_25.1.pdi
LOG=/tmp/v80_step2_flash.log

export LD_LIBRARY_PATH=/opt/xilinx/slash/lib:${LD_LIBRARY_PATH:-}
exec > >(tee -a "$LOG") 2>&1
echo "=========== STEP 2: write static shell to FLASH  ($(date))  log: $LOG ==========="

die()  { echo; echo "FAILED: $*"; exit 1; }
step() { echo; echo "--- $* ---"; }
[ "$(id -u)" -eq 0 ] || die "must run as root"

step "1. Preflight"
[ -f "$PDI" ] || die "missing PDI: $PDI"
# A flash-image PDI starts with the FPT magic 0x92F7A516 (little-endian on disk
# as 16 a5 f7 92). The no-FPT JTAG PDI does not, and --flash would reject it.
magic=$(od -A n -t x4 -N 4 "$PDI" | tr -d ' \n')
[ "$magic" = "92f7a516" ] || die "PDI does not carry the FPT magic (got $magic).
    --flash requires the flash-image PDI, not the _nofpt JTAG one."
echo "  PDI    : $PDI"
echo "  size   : $(stat -c%s "$PDI") bytes"
echo "  FPT    : magic 0x$magic OK"

lsmod | grep -q '^ami '   || die "ami.ko not loaded -- the flash write goes through the AMC behind PF0. Run step 1 first."
lsmod | grep -q '^slash ' || die "slash.ko not loaded. Run step 1 first."
systemctl is-active --quiet vrtd.service || die "vrtd not active. Run step 1 first."

step "2. Board as VRTD currently sees it"
"$SMI" list || die "v80-smi list failed -- board not resolvable, do not flash"

# Derive the BDF from PF0 in sysfs, NOT by regexing 'v80-smi list'. That output
# is free-form and a pattern like [0-9a-f]{2}:[0-9a-f]{2} also matches a clock
# time ("11:45") or a serial fragment, which would aim the write at a bogus
# target. lspci -d 10ee:50b4 -D is unambiguous: exactly the AVED management PF.
PF0=$(lspci -d 10ee:50b4 -D 2>/dev/null | awk '{print $1}')
[ -n "$PF0" ] || die "PF0 (10ee:50b4) not found on the bus -- refusing to flash blind"
[ "$(printf '%s\n' "$PF0" | wc -l)" -eq 1 ] || die "more than one V80 PF0 found:
$PF0
    This script assumes a single board. Pass the BDF by hand instead."
BDF=${PF0%.*}                       # 0000:01:00.0 -> 0000:01:00
echo
echo "  PF0 at      : $PF0"
echo "  target board: $BDF"

step "3. WRITING FLASH -- do not interrupt (driver timeout is 40 minutes)"
# WHY --shell-type all, not compute.
# `compute` writes boot partition 1 ONLY. But `v80-smi reset --shell-type`
# documents that it "defaults to service", i.e. partition 0 is the default boot
# target -- so a cold boot would still read partition 0, still find the old
# image, and Shell: would revert to unknown. That fails the one acceptance
# criterion that matters (survives a reboot). Writing `all` puts the compute
# shell in BOTH partitions, so whichever one the PMC boots reports `compute`.
#
# `all` is rejected by *slashkit's resolver*, but --pdi bypasses the resolver
# entirely, so v80-smi should accept it. If it does not, the rejection is an
# argument error raised in the first seconds, long before any flash access --
# so a fast failure is safe to retry with `compute`, and a SLOW failure is a
# real write failure that must NOT be retried blindly.
run_write() {
    local st=$1
    echo "  command: $SMI write-static-shell --flash -d $BDF --shell-type $st --pdi <pdi>"
    echo "  started: $(date)"
    echo
    local t0 rc
    t0=$(date +%s)
    timeout 3600 "$SMI" write-static-shell --flash -d "$BDF" --shell-type "$st" --pdi "$PDI"
    rc=$?
    WRITE_ELAPSED=$(( $(date +%s) - t0 ))
    return $rc
}

if run_write all; then
    echo; echo "  FLASH WRITE REPORTED SUCCESS (both partitions)  ($(date))"
elif [ "$WRITE_ELAPSED" -lt 30 ]; then
    echo
    echo "  '--shell-type all' rejected after ${WRITE_ELAPSED}s -- an argument error,"
    echo "  raised before any flash access. Falling back to partition 1 only."
    echo
    if run_write compute; then
        echo; echo "  FLASH WRITE REPORTED SUCCESS (partition 1 only)  ($(date))"
        echo "  NOTE: partition 0 still holds the OLD image. If a cold boot"
        echo "        reports Shell: unknown again, that is why."
    else
        echo; echo "  FLASH WRITE FAILED after ${WRITE_ELAPSED}s  ($(date))"
        die "see recovery notes at the top of this script"
    fi
else
    echo
    echo "  FLASH WRITE FAILED after ${WRITE_ELAPSED}s  ($(date))"
    die "This failed too slowly to be an argument error, so flash was probably
    being written. NOT retrying automatically. See the recovery notes at the
    top of this script and check 'v80-smi list' before doing anything else."
fi

step "4. Verify -- Shell: must now read 'compute'"
# The command resets the board into the programmed partition, and a full PDI
# boot takes ~13.5 s. Poll rather than sampling once: a single early read would
# report a failure that is really just the board still booting. Never `die`
# here -- the flash write already reported success, and calling a slow reboot
# a FAILURE would send us into an unnecessary and risky recovery.
for i in 1 2 3 4 5 6; do
    echo "  attempt $i ($(date +%T)):"
    out=$("$SMI" list 2>&1); echo "$out" | sed 's/^/    /'
    if echo "$out" | grep -qi 'shell.*compute'; then
        echo; echo "  Shell: compute -- CONFIRMED"; break
    fi
    [ $i -lt 6 ] && sleep 15
done

cat <<'EOF'

=========== STEP 2 COMPLETE ===========
If Shell: now reads 'compute', the omission is closed.

The real acceptance test is that it SURVIVES A REBOOT and the card enumerates
with no JTAG intervention:

    sudo reboot
    # then, with no JTAG step at all:
    ls -d /sys/bus/pci/devices/0000:00:01.1 && lspci -d 10ee: -nn
    sudo bash ~/dev/v80/step1_load.sh
EOF
