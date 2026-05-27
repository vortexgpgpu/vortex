#!/usr/bin/env python3
"""Run DFV stall injection tests across all regression apps.

For each app, patches kernel.cpp to enable/disable stall types,
runs the RTL simulation, trims run.log to the last 15 lines,
extracts PERF and type-specific metrics, and saves logs + CSV.
"""

import csv
import os
import re
import shutil
import subprocess

VORTEX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REG_DIR = os.path.join(VORTEX_ROOT, "tests", "regression")
BUILD_DIR = os.path.join(VORTEX_ROOT, "build")
RUN_LOG = os.path.join(BUILD_DIR, "run.log")
LOG_BASE = os.path.join(VORTEX_ROOT, "logs")

# =============================================================================
# DFV TEST TYPE
#   1 = Arbitration race condition
#         Reports L1 arbiter and per-bank collision counts.
#         Use icache/dcache stall combos; collision $display lines are the signal.
#   2 = Downstream backpressure / upstream throttle
#         Reports whether the throttle counter ever reached its threshold.
#         Use fill stall; DFV_THROTTLE_REACHED $display lines are the signal.
# =============================================================================
DFV_TEST_TYPE = 2

STALL_TYPES = [
    ("icache",    "VX_CSR_DFV_ICACHE_STALL"),
    ("dcache",    "VX_CSR_DFV_DCACHE_STALL"),
    ("writeback", "VX_CSR_DFV_WRITEBACK_STALL"),
    ("fill",      "VX_CSR_DFV_FILL_STALL"),
]

# Defaults matching the normalized kernel.cpp baseline
DEFAULT_SET_THRESHOLD      = 240
DEFAULT_RELEASE_THRESHOLD  = 65504
DEFAULT_RELEASE_DELAY      = 0x1000
DEFAULT_RELEASE_FOREVER    = 1
DEFAULT_THROTTLE_THRESHOLD = 0x1800

# Each combo dict supports:
#   "stalls"             – tuple of bools, one per STALL_TYPES entry (required)
#   "label"              – short name used in log filenames and CSV (optional)
#   "set_threshold"      – VX_CSR_DFV_SET_THRESHOLD       (default 240)
#   "release_threshold"  – VX_CSR_DFV_RELEASE_THRESHOLD   (default 65504)
#   "release_delay"      – VX_CSR_DFV_RELEASE_DELAY       (default 0x1000)
#   "release_forever"    – VX_CSR_DFV_RELEASE_FOREVER     (default 1)
#   "throttle_threshold" – VX_CSR_DFV_THROTTLE_THRESHOLD  (default 0x1800)

# Type 1: watch L1 arbiter/bank collision counters
# Use faster release (threshold=240) so stalls cycle and produce many collision events
COMBOS_TYPE1 = [
    {"stalls": (False, False, False, False), "label": "none",
     "release_threshold": 240, "release_forever": 0},
    {"stalls": (False, True,  False, True),  "label": "dcache_fill",
     "release_threshold": 240, "release_forever": 0},
    {"stalls": (True,  True,  False, False), "label": "icache_dcache",
     "release_threshold": 240, "release_forever": 0},
    {"stalls": (True,  True,  False, True),  "label": "ic_dc_fill",
     "release_threshold": 240, "release_forever": 0},
]

# Type 2: watch throttle counter — did the pipeline fully freeze?
# Fill stall held until permanent release (release_forever=1); threshold 0x1800
COMBOS_TYPE2 = [
    {"stalls": (False, False, False, False), "label": "none"},
    {"stalls": (False, False, False, True),  "label": "fill_only"},
]

COMBOS = COMBOS_TYPE1 if DFV_TEST_TYPE == 1 else COMBOS_TYPE2

# Per-app arguments (size flags; -d is appended to enable DFV path)
TEST_APPS = {
    "printf":     "-n4",
    "relu":       "-n64",
    "sgemm":      "-n8",
    "sgemm2":     "-n16",
    "sgemm_tcu":  "-m8 -n8",
    "sgemv":      "-m16 -n16",
    "sort":       "-n16",
    "stencil3d":  "-n4",
    "vecadd":     "-n64",
    "basic":      "-n16",
    "conv3":      "-n8",
    "cta":        "-x2 -y2 -z2 -a2 -b2 -c2",
    "demo":       "-n16",
    "diverge":    "-n16",
    "dogfood":    "-n16",
    "dotproduct": "-n64",
    "dropout":    "-n64",
    "fence":      "-n16",
    "io_addr":    "-n16",
    "madmax":     "-n4",
    "mstress":    "-n16",
}


# =============================================================================
# Kernel patching
# =============================================================================

def kernel_path(app):
    return os.path.join(REG_DIR, app, "kernel.cpp")


def patch_kernel(app, combo_dict):
    """Rewrite CSR values in kernel.cpp according to combo_dict."""
    path = kernel_path(app)
    with open(path, "r") as f:
        src = f.read()

    stalls     = combo_dict.get("stalls", (False, False, False, False))
    set_thr    = combo_dict.get("set_threshold",      DEFAULT_SET_THRESHOLD)
    rel_thr    = combo_dict.get("release_threshold",  DEFAULT_RELEASE_THRESHOLD)
    rel_dly    = combo_dict.get("release_delay",      DEFAULT_RELEASE_DELAY)
    rel_forev  = combo_dict.get("release_forever",    DEFAULT_RELEASE_FOREVER)
    thr_thresh = combo_dict.get("throttle_threshold", DEFAULT_THROTTLE_THRESHOLD)

    for (_, csr_macro), enabled in zip(STALL_TYPES, stalls):
        pattern = rf"(csr_write\({csr_macro},\s*)\d(\))"
        src = re.sub(pattern, rf"\g<1>{int(enabled)}\2", src)

    for csr_name, value in [
        ("VX_CSR_DFV_SET_THRESHOLD",      set_thr),
        ("VX_CSR_DFV_RELEASE_THRESHOLD",  rel_thr),
        ("VX_CSR_DFV_RELEASE_DELAY",      rel_dly),
        ("VX_CSR_DFV_RELEASE_FOREVER",    rel_forev),
        ("VX_CSR_DFV_THROTTLE_THRESHOLD", thr_thresh),
    ]:
        pattern = rf"(csr_write\({csr_name},\s*)[^)]+(\))"
        src = re.sub(pattern, rf"\g<1>{value}\2", src)

    with open(path, "w") as f:
        f.write(src)


# =============================================================================
# Log parsing
# =============================================================================

def extract_collisions(lines):
    """Extract last collision count for each counter from log lines (type 1)."""
    results = {}
    for line in lines:
        m = re.search(r"DFV_COLLISION_NATURAL:.*dfv_l1_arb_ctr\s+edge_count=(\d+)", line)
        if m:
            results["natural_l1_arb"] = int(m.group(1))
        m = re.search(r"DFV_COLLISION_DFV:.*dfv_l1_arb_ctr\s+edge_count=(\d+)", line)
        if m:
            results["dfv_l1_arb"] = int(m.group(1))
        m = re.search(
            r"DFV_COLLISION_NATURAL:.*dcache.*g_banks\[(\d+)\]\.dfv_bank_ctr\s+edge_count=(\d+)", line
        )
        if m:
            results[f"natural_bank{m.group(1)}"] = int(m.group(2))
        m = re.search(
            r"DFV_COLLISION_DFV:.*dcache.*g_banks\[(\d+)\]\.dfv_bank_ctr\s+edge_count=(\d+)", line
        )
        if m:
            results[f"dfv_bank{m.group(1)}"] = int(m.group(2))
        # Appended summary lines from a previous trim
        m = re.search(r"COLLISION:\s+(\w+)=(\d+)", line)
        if m:
            results[m.group(1)] = int(m.group(2))
    return results


def extract_throttle_reached(lines):
    """Count how many times the throttle counter fired (type 2).

    Filters out the spurious init-time fire that occurs before the CSR is
    programmed: at reset, threshold_buf=0 so the counter fires immediately
    with threshold=0 in the $display output.
    """
    count = 0
    for line in lines:
        if "DFV_THROTTLE_REACHED" not in line:
            continue
        m = re.search(r"threshold=(\d+)", line)
        if m and int(m.group(1)) == 0:
            continue  # skip init-time spurious fire
        count += 1
    # Also pick up appended summary line from a previous trim
    for line in lines:
        m = re.search(r"THROTTLE_REACHED:\s*(\d+)", line)
        if m:
            count = int(m.group(1))
    return count


def extract_perf(lines):
    """Parse 'PERF: instrs=N, cycles=N, IPC=N' from log lines."""
    for line in lines:
        m = re.search(r"PERF:\s*instrs=(\d+),\s*cycles=(\d+),\s*IPC=([\d.]+)", line)
        if m:
            return int(m.group(1)), int(m.group(2)), float(m.group(3))
    return -1, -1, -1.0


def trim_log(path, keep=15):
    """Trim log to last N lines; extract metrics; append summary footer."""
    with open(path, "r", errors="replace") as f:
        all_lines = f.readlines()

    sig_mismatch = any(line.lower().startswith("error at result") for line in all_lines)

    if DFV_TEST_TYPE == 1:
        collisions       = extract_collisions(all_lines)
        throttle_reached = 0
    else:
        collisions       = {}
        throttle_reached = extract_throttle_reached(all_lines)

    tail = all_lines[-keep:]
    tail.append("\n")
    if DFV_TEST_TYPE == 1:
        for key, val in sorted(collisions.items()):
            tail.append(f"COLLISION: {key}={val}\n")
    else:
        tail.append(f"THROTTLE_REACHED: {throttle_reached}\n")
    if sig_mismatch:
        tail.append("SIGMISMATCH: true\n")

    with open(path, "w") as f:
        f.writelines(tail)
    return tail, collisions, throttle_reached, sig_mismatch


# =============================================================================
# Combo labeling
# =============================================================================

def combo_tag(combo_dict):
    if "label" in combo_dict:
        return combo_dict["label"]
    stalls = combo_dict.get("stalls", (False,) * 4)
    enabled = [name for (name, _), on in zip(STALL_TYPES, stalls) if on]
    return "_".join(enabled) if enabled else "none"


def combo_name(combo_dict):
    return f"run_{combo_tag(combo_dict)}.log"


def combo_desc(combo_dict):
    stalls = combo_dict.get("stalls", (False,) * 4)
    parts = [f"{s[0]}={'ON' if on else 'off'}" for s, on in zip(STALL_TYPES, stalls)]
    desc = ", ".join(parts)
    extras = []
    for key, default, fmt in [
        ("set_threshold",      DEFAULT_SET_THRESHOLD,      lambda v: f"set={v}"),
        ("release_threshold",  DEFAULT_RELEASE_THRESHOLD,  lambda v: f"rel={v}"),
        ("release_delay",      DEFAULT_RELEASE_DELAY,      lambda v: f"delay=0x{v:X}"),
        ("release_forever",    DEFAULT_RELEASE_FOREVER,    lambda v: f"forever={v}"),
        ("throttle_threshold", DEFAULT_THROTTLE_THRESHOLD, lambda v: f"throttle=0x{v:X}"),
    ]:
        if key in combo_dict and combo_dict[key] != default:
            extras.append(fmt(combo_dict[key]))
    if extras:
        desc += ", " + ", ".join(extras)
    return desc


# =============================================================================
# CSV helpers
# =============================================================================

def get_collision_columns(perf_rows):
    cols = set()
    for row in perf_rows:
        cols.update(row[6].keys())  # collisions dict at index 6
    arb_cols = sorted(c for c in cols if "l1_arb" in c)
    def bank_key(x):
        m = re.search(r"\d+", x.split("bank")[1])
        return (x.split("bank")[0], int(m.group()) if m else 0)
    bank_cols = sorted((c for c in cols if "bank" in c), key=bank_key)
    return arb_cols + bank_cols


# =============================================================================
# Main
# =============================================================================

def main():
    apps = list(TEST_APPS.keys())
    total = len(apps) * len(COMBOS)
    type_label = "arbitration_race" if DFV_TEST_TYPE == 1 else "fill_throttle"
    print(f"DFV_TEST_TYPE={DFV_TEST_TYPE} ({type_label}), {len(apps)} apps x {len(COMBOS)} combos = {total} runs")

    perf_rows = []  # (app, tag, instrs, cycles, ipc, status, collisions, throttle_reached)
    idx = 0

    for app in apps:
        log_dir = os.path.join(LOG_BASE, app)
        os.makedirs(log_dir, exist_ok=True)

        for combo_dict in COMBOS:
            idx += 1
            tag  = combo_tag(combo_dict)
            name = combo_name(combo_dict)
            desc = combo_desc(combo_dict)
            print(f"\n[{idx}/{total}] {app}/{name}  ({desc})")

            patch_kernel(app, combo_dict)
            args = TEST_APPS[app] + " -d"
            cmd = f"./ci/blackbox.sh --driver=rtlsim --app={app} --args='{args}' >run.log 2>&1"
            print(f"  CMD: {cmd}")
            rc = subprocess.run(cmd, shell=True, cwd=BUILD_DIR).returncode

            if not os.path.isfile(RUN_LOG):
                print(f"  WARNING: {RUN_LOG} not found")
                perf_rows.append((app, tag, -1, -1, -1.0, "NO_LOG", {}, 0))
                continue

            tail, collisions, throttle_reached, sig_mismatch = trim_log(RUN_LOG)
            instrs, cycles, ipc = extract_perf(tail)
            dest = os.path.join(log_dir, name)
            shutil.move(RUN_LOG, dest)

            if sig_mismatch:
                status = "SIGMISMATCH"
            elif rc == 0:
                status = "PASS"
            else:
                status = f"FAIL(rc={rc})"

            perf_rows.append((app, tag, instrs, cycles, ipc, status, collisions, throttle_reached))

            if DFV_TEST_TYPE == 1:
                coll_str = " ".join(f"{k}={v}" for k, v in sorted(collisions.items()))
                extra = f"[{coll_str}]" if coll_str else "[]"
            else:
                extra = f"[throttle_reached={throttle_reached}]"

            perf_str = f"instrs={instrs} cycles={cycles} IPC={ipc:.4f}" if instrs != -1 else "no PERF line"
            print(f"  {status}  {perf_str} {extra} -> {dest}")

        # Restore kernel to normalized baseline after each app
        patch_kernel(app, {"stalls": (False, False, False, True)})

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    os.makedirs(LOG_BASE, exist_ok=True)
    csv_path = os.path.join(LOG_BASE, "dfv_perf.csv")

    if DFV_TEST_TYPE == 1:
        coll_cols = get_collision_columns(perf_rows)
        header = ["app", "stall_combo", "instrs", "cycles", "IPC", "status"] + coll_cols
    else:
        coll_cols = []
        header = ["app", "stall_combo", "instrs", "cycles", "IPC", "status", "throttle_reached"]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for app, tag, instrs, cycles, ipc, status, coll, thr in perf_rows:
            row = [app, tag, instrs, cycles, ipc, status]
            if DFV_TEST_TYPE == 1:
                row += [coll.get(c, -1) for c in coll_cols]
            else:
                row.append(thr)
            w.writerow(row)
    print(f"\nPerf CSV written to {csv_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    if DFV_TEST_TYPE == 1:
        coll_hdrs = [c[:12] for c in coll_cols]
        hdr = f"{'APP':<14} {'COMBO':<16} {'INSTRS':>8} {'CYCLES':>8} {'IPC':>7}"
        for h in coll_hdrs:
            hdr += f" {h:>12}"
        hdr += " STATUS"
        width = max(len(hdr) + 10, 60)
        print(f"\n{'=' * width}\n{hdr}\n{'-' * width}")
        for app, tag, instrs, cycles, ipc, status, coll, _ in perf_rows:
            i_s = str(instrs) if instrs != -1 else "-"
            c_s = str(cycles) if cycles != -1 else "-"
            p_s = f"{ipc:.4f}" if ipc != -1.0 else "-"
            line = f"{app:<14} {tag:<16} {i_s:>8} {c_s:>8} {p_s:>7}"
            for c in coll_cols:
                v = coll.get(c, -1)
                line += f" {str(v) if v != -1 else '-':>12}"
            line += f" {status}"
            print(line)
    else:
        hdr = f"{'APP':<14} {'COMBO':<16} {'INSTRS':>8} {'CYCLES':>8} {'IPC':>7} {'THR_REACHED':>12} STATUS"
        width = max(len(hdr) + 10, 60)
        print(f"\n{'=' * width}\n{hdr}\n{'-' * width}")
        for app, tag, instrs, cycles, ipc, status, _, thr in perf_rows:
            i_s = str(instrs) if instrs != -1 else "-"
            c_s = str(cycles) if cycles != -1 else "-"
            p_s = f"{ipc:.4f}" if ipc != -1.0 else "-"
            t_s = str(thr)
            print(f"{app:<14} {tag:<16} {i_s:>8} {c_s:>8} {p_s:>7} {t_s:>12} {status}")

    print(f"{'=' * width}")
    print(f"Logs saved under {LOG_BASE}/")


if __name__ == "__main__":
    main()
