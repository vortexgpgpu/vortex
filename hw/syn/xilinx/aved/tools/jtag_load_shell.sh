#!/bin/bash
# Load the SLASH compute static shell into the V80's PL over JTAG, using AMD's
# own xsdb script -- NOT `v80-smi write-static-shell --jtag`.
#
#   bash ~/dev/v80/jtag_load_shell.sh
#
# NO ROOT NEEDED (Xilinx cable udev rules are installed, FTDI node is 0666).
#
# WHY NOT v80-smi
# ---------------
# `v80-smi write-static-shell --jtag` connects to VRTD to resolve the device
# before it will do anything:
#     Connecting to VRTD... Resolving device address... Resolving VRTD device...
#     SMI execution failed: Requested resouce doesn't exist
# VRTD only knows devices that are enumerated on PCIe, which is exactly what is
# broken when you need this. The documented recovery path cannot recover an
# off-bus card. share/v80-smi/versal_flash_pdi.tcl has no such dependency.
#
# *** NO FLASH WRITE. *** The no-FPT PDI is a JTAG-bootable boot image with no
# flash-programming content. The load is volatile; a power cycle undoes it.
#
# WHY THE USB RESET FIRST
# -----------------------
# After a reboot ftdi_sio re-grabs all four FT4232H interfaces (/dev/ttyUSB0..3),
# so hw_server cannot open the JTAG channel and reports an EMPTY target list --
# which looks exactly like a dead card. A userspace USBDEVFS_RESET makes udev
# re-apply the Xilinx rules and release the JTAG interface.
#
# Check the chain with `jtag targets`, NOT `targets`: the latter needs a running
# PLM and is empty on an unconfigured device even when the chain is healthy.
# A valid idcode (xcv80 = 14d2f093) means the TAP is powered; all-ones at every
# frequency means the core rails are down and only a cold power cycle helps.

source /home/blaise/dev/xilinx_setup_aved.sh >/dev/null 2>&1
set -u

LOG=/tmp/v80_jtag_load.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

TCL=/opt/xilinx/slash/share/v80-smi/versal_flash_pdi.tcl
PDI=/home/blaise/dev/SLASH-compute/linker/slashkit/resources/static_shell_compute/amd_v80_gen5x8_25.1_nofpt.pdi

[ -f "$TCL" ] || { echo "missing $TCL"; exit 1; }
[ -f "$PDI" ] || { echo "missing $PDI"; exit 1; }

echo "=========== V80 JTAG shell load ($(date)) -- log: $LOG ==========="
echo "PDI: $PDI  (no-FPT = JTAG-bootable, no flash content)"

# Kill hw_server by exact comm. NEVER `pkill -f hw_server`: that pattern matches
# this script's own command line and takes the whole job down with it.
kill_hw_server() {
    ps -eo pid,comm | awk '$2=="hw_server"{print $1}' | xargs -r kill 2>/dev/null
    sleep 2
}

echo
echo "=== 1. release the JTAG cable from ftdi_sio ==="
ls /dev/ttyUSB* 2>/dev/null | tr '\n' ' '; echo "<- before"
python3 /home/blaise/dev/.usbreset.py || { echo "USB reset failed"; exit 1; }
sleep 4
ls /dev/ttyUSB* 2>/dev/null | tr '\n' ' '; echo "<- after"

echo
echo "=== 2. program the PDI (several minutes; do not interrupt) ==="
kill_hw_server
if PDI_PATH="$PDI" timeout 1800 xsdb "$TCL"; then
    echo
    echo "  PDI PROGRAMMED SUCCESSFULLY"
else
    rc=$?
    echo
    echo "  PDI programming FAILED (rc=$rc)"
    echo "  the script restores the pin-strapped boot mode on failure;"
    echo "  flash is untouched, so the board is no worse than before."
    exit 1
fi

cat <<'EOF'

=========== NEXT STEP: ONE REBOOT ===========
The shell is resident in the PL, but the card is still not on the PCIe bus and
no rescan can put it there: root port 0000:00:01.1 is created by firmware at
POST and only exists if the card's link is up at that instant.

    sudo reboot

Then:
    ls -d /sys/bus/pci/devices/0000:00:01.1 && lspci -d 10ee: -nn
    sudo /opt/xilinx/slash/bin/v80-smi list
EOF
