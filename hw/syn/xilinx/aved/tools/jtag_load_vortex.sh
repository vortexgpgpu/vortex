#!/bin/bash
# Load the Vortex AFU partial PDI into the V80's reconfigurable partition over
# JTAG, so the design is resident WITHOUT vrtd's design writer.
#
#   bash ~/dev/v80/jtag_load_vortex.sh [path/to/vortex_afu.vbin]
#
# NO ROOT. NO PCIe. NO REBOOT.
#
# WHY THIS EXISTS
# ---------------
# The normal route (vrt::Device programming the vbin) goes through vrtd's design
# writer, which runs reset_with_ami whenever the requested shell differs from
# the current one -- and that toggles a secondary bus reset. An SBR on root port
# 0000:00:01.1 hard-reset this host on 2026-08-19 09:40. Loading the same PDI
# over JTAG puts the AFU in the fabric with no PCIe transaction whatsoever, so
# every later run can use VORTEX_AVED_NO_PROGRAM=1 and never touch that path.
#
# Verified 2026-08-19: `device program` on the "Versal xcv80" target accepts the
# partial PDI and returns OK. Partial reconfiguration does not drop the PCIe
# link, so the card stays enumerated across the load.
#
# NOTE: a reboot loses this. The card then boots from OSPI (pin-strapped mode,
# which AMD's versal_flash_pdi.tcl deliberately restores), so re-run this after
# every reboot, before loading the drivers.

source /home/blaise/dev/xilinx_setup_aved.sh >/dev/null 2>&1
set -u

VBIN="${1:-/home/blaise/dev/vortex_gfxw_v2/build/hw/syn/xilinx/aved/hbm1_aved_hw/bin/vortex_afu.vbin}"
LOG=/tmp/v80_jtag_vortex.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
echo "=========== JTAG load of the Vortex AFU ($(date)) -- log: $LOG ==========="

[ -f "$VBIN" ] || { echo "missing vbin: $VBIN"; exit 1; }

WORK=$(mktemp -d /tmp/vortex_pdi.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
tar xzf "$VBIN" -C "$WORK" || { echo "could not extract $VBIN"; exit 1; }

PDI=$(find "$WORK" -name "*partial.pdi" | head -1)
[ -n "$PDI" ] || { echo "no partial PDI inside $VBIN"; ls -R "$WORK"; exit 1; }
echo "vbin: $VBIN"
echo "PDI : $(basename "$PDI") ($(stat -c%s "$PDI") bytes)"
grep -oE "<(BaseAddress|ShellType)>[^<]*</(BaseAddress|ShellType)>" "$WORK/system_map.xml" 2>/dev/null

# Kill hw_server by exact comm. NEVER `pkill -f hw_server` -- that pattern
# matches this script's own command line and takes the whole job down.
ps -eo pid,comm | awk '$2=="hw_server"{print $1}' | xargs -r kill 2>/dev/null
sleep 2

# ftdi_sio re-grabs all four FT4232H interfaces after a reboot, which makes
# hw_server report an EMPTY target list -- indistinguishable from a dead card.
echo
echo "=== release the JTAG cable from ftdi_sio ==="
python3 /home/blaise/dev/.usbreset.py || { echo "USB reset failed"; exit 1; }
sleep 4

cat > "$WORK/load.tcl" <<'EOF'
connect
set pdi $::env(PDI_PATH)
targets -set -filter {name =~ "Versal xcv80"}
device program $pdi
puts "PARTIAL_PDI_OK"
EOF

echo
echo "=== program the partial PDI (minutes; do not interrupt) ==="
if PDI_PATH="$PDI" timeout 1800 xsdb "$WORK/load.tcl" 2>&1 | tee /dev/stderr | grep -q PARTIAL_PDI_OK; then
    echo
    echo "  VORTEX AFU LOADED"
else
    echo
    echo "  LOAD FAILED -- the RP may be partially configured."
    echo "  Recover with: bash ~/dev/v80/jtag_load_shell.sh   (then reboot)"
    exit 1
fi

cat <<'EOF'

=========== next ===========
The AFU is resident. Load the drivers WITHOUT ami, then run the ladder:

    sudo bash ~/dev/v80/slash_only_load.sh
    bash ~/dev/v80/hw_ladder_noprogram.sh
EOF
