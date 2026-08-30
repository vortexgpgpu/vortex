#!/usr/bin/env python3
"""1 Hz forensic sampler for orcas2 / V80 runs.

Every sample is written AND fsync'd, so the last line on disk is the last state
the machine was in before a hard reset. That is the whole point: journald's
default SyncIntervalSec is 5 minutes, so up to five minutes of ordinary logging
is lost when this box resets, which leaves every crash with an uninformative
tail.

  python3 sampler.py <logdir>

Writes <logdir>/sample.tsv (one fsync'd line per second) and reads
<logdir>/breadcrumb (whatever the harness last wrote) so each sample is
attributed to the operation that was in flight.

Columns are fixed-width TSV so postmortem.sh can diff the last-known-good
sample against the moment of death.
"""
import os
import sys
import time

FIELDS = [
    "wall", "mono", "uptime", "breadcrumb",
    "idle_drv", "c1_usage", "c2_usage", "c2_time_us",
    "link_sta", "aer_fatal", "aer_nonfatal",
    "v80_pf0", "loadavg", "mce_count",
]

PF0 = "/sys/bus/pci/devices/0000:01:00.0"
BRIDGE = "/sys/bus/pci/devices/0000:00:01.1"
CPU0_IDLE = "/sys/devices/system/cpu/cpu0/cpuidle"


def read(path, default="-"):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return default


def cstate(idx, leaf):
    """With Power Supply Idle Control set to Typical Current Idle this box
    reports cpuidle current_driver=none, i.e. no ACPI C-states at all. Keep the
    columns so a regression back to acpi_idle/C2 is visible immediately."""
    return read(f"{CPU0_IDLE}/state{idx}/{leaf}", "-")


def idle_driver():
    return read("/sys/devices/system/cpu/cpuidle/current_driver", "none")


def v80_present():
    """01:00.0 exists even with the V80 off the bus -- it is then an AMD
    1022:1556. Presence of the path is NOT presence of the card; check IDs."""
    if read(f"{PF0}/vendor", "") != "0x10ee":
        return "GONE"
    return "up" if read(f"{PF0}/device", "") == "0x50b4" else "wrong-id"


def mce_total():
    """Machine checks delivered to Linux. Stays 0 under firmware-first mode --
    a NON-zero value here would itself be a finding worth knowing about."""
    try:
        with open("/proc/interrupts") as fh:
            for line in fh:
                if line.strip().startswith("MCE:"):
                    return str(sum(int(x) for x in line.split()[1:] if x.isdigit()))
    except OSError:
        pass
    return "-"


def aer(kind):
    return read(f"{BRIDGE}/aer_dev_{kind}", "-").replace("\n", " ")[:40]


def sample(logdir):
    return "\t".join([
        time.strftime("%Y-%m-%dT%H:%M:%S"),
        f"{time.monotonic():.1f}",
        read("/proc/uptime").split()[0] if read("/proc/uptime") != "-" else "-",
        read(os.path.join(logdir, "breadcrumb"), "none"),
        idle_driver(),
        cstate(1, "usage"),
        cstate(2, "usage"),
        cstate(2, "time"),
        read(f"{BRIDGE}/current_link_speed", "-") + "/" + read(f"{BRIDGE}/current_link_width", "-"),
        aer("fatal"),
        aer("nonfatal"),
        v80_present(),
        read("/proc/loadavg", "-").split(" ")[0],
        mce_total(),
    ])


def main():
    logdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(logdir, exist_ok=True)
    path = os.path.join(logdir, "sample.tsv")
    new = not os.path.exists(path)
    # Line-buffered, and we fsync explicitly after every record.
    with open(path, "a", buffering=1) as fh:
        if new:
            fh.write("#" + "\t".join(FIELDS) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        while True:
            fh.write(sample(logdir) + "\n")
            fh.flush()
            os.fsync(fh.fileno())      # <-- survives a hard reset
            time.sleep(1.0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
