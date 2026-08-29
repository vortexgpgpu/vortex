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

# No default. This used to fall back to a fixed build directory, which meant
# "reload the AFU" quietly meant "load whichever build that path happens to
# hold" -- and after a rebuild under a new PREFIX, that is not the design you
# were testing. Loading the wrong bitstream is a reconfiguration plus, in the
# bad cases, a reboot; it is not worth saving one argument.
if [ $# -lt 1 ]; then
    echo "usage: jtag_load_vortex.sh <path/to/vortex_afu.vbin>" >&2
    echo >&2
    echo "available builds, newest first:" >&2
    find "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)/build/hw/syn/xilinx/aved" \
         -name vortex_afu.vbin -printf '  %TY-%Tm-%Td %TH:%TM  %p\n' 2>/dev/null | sort -r >&2
    exit 1
fi
VBIN="$1"
LOG=/tmp/v80_jtag_vortex.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
echo "=========== JTAG load of the Vortex AFU ($(date)) -- log: $LOG ==========="

[ -f "$VBIN" ] || { echo "missing vbin: $VBIN"; exit 1; }

# Refuse to reconfigure a region that is still transacting.
#
# Partial reconfiguration of an RP whose masters have AXI transactions in
# flight stalls the PLM ("PLM stalled during programming"), which leaves the
# region half-configured and costs a static-shell reprogram plus a reboot --
# the shell reload invalidates the BAR mapping firmware set up at POST.
#
# A regression binary still holding the device is the usual cause: it is
# killed, its teardown never runs, and the CP is left with an outstanding
# read. Wait for it rather than reconfiguring underneath it.
for _ in $(seq 1 20); do
    holders=$(ps -eo pid,comm,args --no-headers 2>/dev/null \
        | awk '$3 ~ /VORTEX_DRIVER=aved|\/run-aved/ || $2 ~ /^(minimal|demo|sgemm|sgemv|vecadd)$/ {print $1}')
    [ -z "$holders" ] && break
    echo "  waiting for the AFU to be released by: $holders"
    sleep 1
done
if [ -n "${holders:-}" ]; then
    echo
    echo "REFUSING to reconfigure: a process still holds the AFU ($holders)."
    echo "Let it finish, or kill it and wait for the device node to be released."
    echo "Reconfiguring an active region stalls the PLM and costs a reboot."
    exit 1
fi

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
    # Record what is actually resident, so run_hw_test.sh stops guessing.
    #
    # With VORTEX_AVED_NO_PROGRAM=1 the vbin handed to the runtime is never
    # written to the card, but it is still read: portMemoryConfig() takes the
    # connection map out of its system_map.xml and that decides whether the CP
    # memory is staged in HBM or reached through the QDMA slave bridge. Hand it
    # a vbin from a different build tree and the runtime configures itself for
    # hardware that is not in the fabric -- silently, and with symptoms that
    # look like anything but a mismatched file.
    #
    # /tmp is the correct lifetime: a reboot clears the stamp, and a reboot is
    # also exactly when the card reloads from OSPI and loses the AFU.
    realpath "$VBIN" > /tmp/v80_resident_afu.path
    echo "  recorded resident vbin: $VBIN"
else
    rm -f /tmp/v80_resident_afu.path
    echo
    echo "  LOAD FAILED -- the RP may be partially configured."
    echo "  Recover with: bash ~/dev/v80/jtag_load_shell.sh   (then reboot)"
    exit 1
fi

cat <<'EOF'

=========== next ===========
The AFU is resident. The SLASH driver autoloads and vrtd is a systemd
service, so there is nothing to load by hand -- just run a test:

    bash hw/syn/xilinx/aved/tools/run_hw_test.sh minimal
    bash hw/syn/xilinx/aved/tools/hw_sweep.sh          # full regression
EOF
