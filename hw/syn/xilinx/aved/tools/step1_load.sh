#!/bin/bash
# STEP 1 of the supported-path bring-up: load the V80 stack and report state.
#
#   sudo bash ~/dev/v80/step1_load.sh
#
# Does NOT write flash. Does NOT reset the FPGA. Does NOT program a design.
# Everything this does is undone by a reboot.
#
# `ami` IS loaded here on purpose: step 2 writes OSPI flash through the AMC
# over GCQ, which lives behind PF0, and only ami.ko speaks to it.
#
# Difference from the older ~/dev/v80_load.sh: that script's vrtd readiness
# check was `journalctl -u vrtd.service -n 20 | grep "Discovered 0 device(s)"`,
# which scans the last 20 lines of the unit journal across ALL vrtd runs in
# this boot. A stale "Discovered 0" from an earlier start makes it report
# FAILED on a perfectly healthy board (observed 2026-08-20). Here the check is
# scoped to the current invocation with --since.

set -u

AMI_KO=/home/blaise/dev/SLASH/submodules/AVED/sw/AMI/driver/ami.ko
SLASH_KO=/home/blaise/dev/SLASH-compute/driver/slash.ko
AMI_TOOL=/opt/xilinx/slash/bin/ami_tool
SMI=/opt/xilinx/slash/bin/v80-smi
LOG=/tmp/v80_step1.log

export LD_LIBRARY_PATH=/opt/xilinx/slash/lib:${LD_LIBRARY_PATH:-}
exec > >(tee -a "$LOG") 2>&1
echo "=========== STEP 1: load V80 stack  ($(date))  log: $LOG ==========="

die()  { echo; echo "FAILED: $*"; exit 1; }
step() { echo; echo "--- $* ---"; }
[ "$(id -u)" -eq 0 ] || die "must run as root"
for f in "$AMI_KO" "$SLASH_KO" "$AMI_TOOL" "$SMI"; do
    [ -e "$f" ] || die "missing: $f"
done

step "1. PCIe root port and functions"
if [ ! -e /sys/bus/pci/devices/0000:00:01.1 ]; then
    die "root port 0000:00:01.1 absent. It is created by firmware at POST and
    no rescan can conjure it. The card did not train its link in time this
    boot. Re-run the JTAG bootstrap and reboot:
        bash ~/dev/v80/jtag_load_shell.sh   (no sudo needed)
        sudo reboot"
fi
lspci -d 10ee: -nn
lspci -d 10ee:50b4 -nn | grep -q . || die "PF0 (10ee:50b4) absent -- AMC/management PF missing"
lspci -d 10ee:50c1 -nn | grep -q . || die "PF1 is not 10ee:50c1 -- board is not on the compute shell"
lspci -d 10ee:50c2 -nn | grep -q . || die "PF2 (10ee:50c2) absent"

step "2. Load ami (PF0 / AVED management -- required for the step 2 flash write)"
if lsmod | grep -q '^ami '; then echo "ami already loaded"
else insmod "$AMI_KO" || die "insmod ami.ko"; fi
sleep 2

step "3. Load slash (PF1 QDMA + PF2 control)"
if lsmod | grep -q '^slash '; then echo "slash already loaded"
else insmod "$SLASH_KO" || die "insmod slash.ko"; fi
sleep 3

step "4. Driver bindings"
# basename "" exits 0 and prints an empty line, so `pf_drv || echo none` never
# fires; test the string instead.
pf_drv() {
    local l; l=$(readlink "/sys/bus/pci/devices/0000:01:00.$1/driver" 2>/dev/null)
    [ -n "$l" ] && basename "$l" || true
}
for f in 0 1 2; do
    d=$(pf_drv "$f"); printf "  0000:01:00.%s -> %s\n" "$f" "${d:-(no driver bound)}"
done
ls -l /dev/slash_* 2>/dev/null || die "no /dev/slash_* nodes -- slash did not bind"
# PF2 alone satisfies the /dev/slash_* check, so test each PF the stack needs.
[ "$(pf_drv 0)" = "ami" ] || die "PF0 did not bind to ami -- the AVED discovery
    table is unreadable, so the fabric is not running a valid base design.
    Recover with: bash ~/dev/v80/jtag_load_shell.sh  then reboot."
[ "$(pf_drv 1)" = "slash_qdma" ] || die "PF1 did not bind to slash_qdma.
    'already in_use (number=0)' -> leaked qdma id: sudo bash ~/dev/v80/v80_reload_slash.sh
    'config bar passed is INVALID' -> fabric wedged, see PF0."

step "5. AMC state"
"$AMI_TOOL" overview 2>&1 | tail -8

step "6. (Re)start vrtd"
# RESTART, not start: vrtd enumerates /dev/slash_ctl* once at startup and never
# rescans, so a vrtd already running from before insmod stays blind to the board.
systemctl reset-failed vrtd.service 2>/dev/null
systemctl start vrtd.socket || die "vrtd.socket"
TS=$(date '+%Y-%m-%d %H:%M:%S')      # scope the readiness check to THIS restart
systemctl restart vrtd.service || { journalctl -u vrtd.service -n 25 --no-pager; die "vrtd.service"; }
sleep 3
systemctl is-active --quiet vrtd.service || die "vrtd not active"
echo "  vrtd journal for this invocation:"
journalctl -u vrtd.service --since "$TS" --no-pager | sed 's/^/    /'
if journalctl -u vrtd.service --since "$TS" --no-pager 2>/dev/null | grep -q "Discovered 0 device(s)"; then
    die "vrtd discovered 0 devices in THIS invocation -- /dev/slash_ctl* missing when it enumerated"
fi

step "7. Board readiness -- THE KEY LINE IS 'Shell:'"
"$SMI" list

cat <<'EOF'

=========== STEP 1 COMPLETE -- no flash was written ===========
Read the Shell: field above.

  Shell: unknown   -> expected. Flash holds a non-SLASH image. This is the
                      condition step 2 fixes. Proceed to step 2.
  Shell: compute   -> flash is ALREADY correct. Do NOT run step 2; the premise
                      of the whole proposal is wrong and we should re-check.
  Shell: service   -> flash holds the service shell in the booted partition.
                      Proceed to step 2 (writes the compute partition).
EOF
