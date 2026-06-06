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
#         Use fill stall (all banks); DFV_THROTTLE_REACHED $display lines are the signal.
#   3 = Asymmetric slowdown on multi-lane resources
#         Fill stall enabled on bank 0 only; other banks proceed normally.
#         Reports IPC reduction % vs. none baseline per app.
#         release_forever disabled so stall cycles on/off via LFSR2.
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
DEFAULT_FILL_BANK_MASK     = 0xFFFF

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

# Type 3: asymmetric slowdown — stall only bank 0, let other banks proceed.
# release_forever=0: stall cycles on/off via LFSR2 to create intermittent bank 0 pressure.
COMBOS_TYPE3 = [
    {"stalls": (False, False, False, False), "label": "none",
     "release_forever": 0},
    {"stalls": (False, False, False, True),  "label": "bank0_only",
     "fill_bank_mask": 0x1, "release_forever": 0, "release_threshold": 64000},
]

COMBOS = {1: COMBOS_TYPE1, 2: COMBOS_TYPE2, 3: COMBOS_TYPE3}[DFV_TEST_TYPE]

# Per-app arguments (size flags; -d is appended to enable DFV path)
TEST_APPS = {
    #"printf":     "-n4",
    "relu":       "-n64",
    "sgemm":      "-n8",
    "sgemm2":     "-n16",
    #"sgemm_tcu":  "-m8 -n8",
    "sgemv":      "-m16 -n16",
    #"sort":       "-n16",
    #"stencil3d":  "-n4",
    "vecadd":     "-n64",
    #"basic":      "-n16",
    "conv3":      "-n8",
    "cta":        "-x2 -y2 -z2 -a2 -b2 -c2",
    "demo":       "-n16",
    #"diverge":    "-n16",
    #"dogfood":    "-n16",
    "dotproduct": "-n64",
    "dropout":    "-n64",
    "fence":      "-n16",
    "io_addr":    "-n16",
    "madmax":     "-n4",
    #"mstress":    "-n16",
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
    bank_mask  = combo_dict.get("fill_bank_mask",     DEFAULT_FILL_BANK_MASK)

    for (_, csr_macro), enabled in zip(STALL_TYPES, stalls):
        pattern = rf"(csr_write\({csr_macro},\s*)\d(\))"
        src = re.sub(pattern, rf"\g<1>{int(enabled)}\2", src)

    for csr_name, value in [
        ("VX_CSR_DFV_SET_THRESHOLD",      set_thr),
        ("VX_CSR_DFV_RELEASE_THRESHOLD",  rel_thr),
        ("VX_CSR_DFV_RELEASE_DELAY",      rel_dly),
        ("VX_CSR_DFV_RELEASE_FOREVER",    rel_forev),
        ("VX_CSR_DFV_THROTTLE_THRESHOLD", thr_thresh),
        ("VX_CSR_DFV_FILL_BANK_MASK",     bank_mask),
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


def trim_log(path, keep=200):
    """Trim log to last N lines; extract metrics; append summary footer."""
    with open(path, "r", errors="replace") as f:
        all_lines = f.readlines()

    sig_mismatch = any(line.lower().startswith("error at result") for line in all_lines)

    if DFV_TEST_TYPE == 1:
        collisions       = extract_collisions(all_lines)
        throttle_reached = 0
    elif DFV_TEST_TYPE == 2:
        collisions       = {}
        throttle_reached = extract_throttle_reached(all_lines)
    else:  # type 3: IPC drop only, no throttle counter
        collisions       = {}
        throttle_reached = 0

    tail = all_lines[-keep:]
    tail.append("\n")
    if DFV_TEST_TYPE == 1:
        for key, val in sorted(collisions.items()):
            tail.append(f"COLLISION: {key}={val}\n")
    elif DFV_TEST_TYPE == 2:
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
        ("fill_bank_mask",     DEFAULT_FILL_BANK_MASK,     lambda v: f"bank_mask=0x{v:04X}"),
    ]:
        if key in combo_dict and combo_dict[key] != default:
            extras.append(fmt(combo_dict[key]))
    if extras:
        desc += ", " + ", ".join(extras)
    return desc


# =============================================================================
# Post-processing: derived metrics
# =============================================================================

def get_collision_columns(perf_rows):
    """Return sorted list of raw collision counter names."""
    cols = set()
    for row in perf_rows:
        cols.update(row[6].keys())  # collisions dict at index 6
    arb_cols = sorted(c for c in cols if "l1_arb" in c)
    def bank_key(x):
        m = re.search(r"\d+", x.split("bank")[1])
        return (x.split("bank")[0], int(m.group()) if m else 0)
    bank_cols = sorted((c for c in cols if "bank" in c), key=bank_key)
    return arb_cols + bank_cols


def compute_collision_rates(perf_rows, coll_cols):
    """Return per-row dict of collision rates (count/cycle) for each counter.

    Rate is -1.0 when count or cycles are unavailable.
    """
    rates = []
    for _, _, _, cycles, _, _, coll, _ in perf_rows:
        row_rates = {}
        for c in coll_cols:
            count = coll.get(c, -1)
            row_rates[c] = count / cycles if (count >= 0 and cycles > 0) else -1.0
        rates.append(row_rates)
    return rates


def compute_ipc_drops(perf_rows):
    """Return per-row IPC drop % relative to the 'none' combo for the same app.

    Positive value = IPC decreased (slowdown). Returns -1.0 when unavailable.
    """
    baseline = {}  # app -> baseline IPC from 'none' combo
    for app, tag, _, _, ipc, _, _, _ in perf_rows:
        if tag == "none" and ipc > 0:
            baseline[app] = ipc

    drops = []
    for app, tag, _, _, ipc, _, _, _ in perf_rows:
        base = baseline.get(app, -1.0)
        if tag == "none" or base <= 0 or ipc < 0:
            drops.append(-1.0)
        else:
            drops.append((base - ipc) / base * 100.0)
    return drops


# =============================================================================
# Main
# =============================================================================

def main():
    apps = list(TEST_APPS.keys())
    total = len(apps) * len(COMBOS)
    type_label = {1: "arbitration_race", 2: "fill_throttle", 3: "asymmetric_slowdown"}[DFV_TEST_TYPE]
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
            elif DFV_TEST_TYPE == 3:
                extra = "[ipc_drop computed post-run]"
            else:
                extra = f"[throttle_reached={throttle_reached}]"

            perf_str = f"instrs={instrs} cycles={cycles} IPC={ipc:.4f}" if instrs != -1 else "no PERF line"
            print(f"  {status}  {perf_str} {extra} -> {dest}")

        # Restore kernel to normalized baseline after each app
        patch_kernel(app, {"stalls": (False, False, False, True)})

    # ------------------------------------------------------------------
    # Post-processing: derived metrics
    # ------------------------------------------------------------------
    os.makedirs(LOG_BASE, exist_ok=True)
    csv_path = os.path.join(LOG_BASE, "dfv_perf.csv")

    if DFV_TEST_TYPE == 1:
        coll_cols  = get_collision_columns(perf_rows)
        coll_rates = compute_collision_rates(perf_rows, coll_cols)
        rate_cols  = [f"{c}_rate" for c in coll_cols]
    elif DFV_TEST_TYPE == 3:
        ipc_drops = compute_ipc_drops(perf_rows)

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        if DFV_TEST_TYPE == 1:
            w.writerow(["app", "stall_combo", "instrs", "cycles", "IPC", "status"]
                       + coll_cols + rate_cols)
            for i, (app, tag, instrs, cycles, ipc, status, coll, _) in enumerate(perf_rows):
                row = [app, tag, instrs, cycles, ipc, status]
                row += [coll.get(c, -1) for c in coll_cols]
                row += [f"{coll_rates[i][c]:.6f}" if coll_rates[i][c] >= 0 else -1
                        for c in coll_cols]
                w.writerow(row)
        elif DFV_TEST_TYPE == 3:
            w.writerow(["app", "stall_combo", "instrs", "cycles", "IPC", "status",
                        "ipc_drop_pct"])
            for i, (app, tag, instrs, cycles, ipc, status, _, _) in enumerate(perf_rows):
                drop = ipc_drops[i]
                drop_s = f"{drop:.2f}" if drop >= 0 else -1
                w.writerow([app, tag, instrs, cycles, ipc, status, drop_s])
        else:  # type 2
            w.writerow(["app", "stall_combo", "instrs", "cycles", "IPC", "status",
                        "throttle_reached"])
            for app, tag, instrs, cycles, ipc, status, _, thr in perf_rows:
                w.writerow([app, tag, instrs, cycles, ipc, status, thr])
    print(f"\nPerf CSV written to {csv_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    if DFV_TEST_TYPE == 1:
        # Show count and rate together: "N(r)" per counter
        coll_hdrs = [c[:10] for c in coll_cols]
        hdr = f"{'APP':<14} {'COMBO':<16} {'INSTRS':>8} {'CYCLES':>8} {'IPC':>7}"
        for h in coll_hdrs:
            hdr += f" {h:>16}"
        hdr += " STATUS"
        width = max(len(hdr) + 4, 60)
        print(f"\n{'=' * width}\n{hdr}\n{'-' * width}")
        for i, (app, tag, instrs, cycles, ipc, status, coll, _) in enumerate(perf_rows):
            i_s = str(instrs) if instrs != -1 else "-"
            c_s = str(cycles) if cycles != -1 else "-"
            p_s = f"{ipc:.4f}" if ipc != -1.0 else "-"
            line = f"{app:<14} {tag:<16} {i_s:>8} {c_s:>8} {p_s:>7}"
            for c in coll_cols:
                count = coll.get(c, -1)
                rate  = coll_rates[i][c]
                if count < 0:
                    cell = "-"
                elif rate >= 0:
                    cell = f"{count}({rate:.2e})"
                else:
                    cell = str(count)
                line += f" {cell:>16}"
            line += f" {status}"
            print(line)
        print(f"{'=' * width}")
        print("  Column format: count(rate/cycle)")

    elif DFV_TEST_TYPE == 3:
        hdr = (f"{'APP':<14} {'COMBO':<16} {'INSTRS':>8} {'CYCLES':>8}"
               f" {'IPC':>7} {'IPC_BASE':>9} {'DROP%':>7} STATUS")
        width = max(len(hdr) + 4, 60)
        # Build per-app baseline lookup for display
        baseline_ipc = {}
        for app, tag, _, _, ipc, _, _, _ in perf_rows:
            if tag == "none" and ipc > 0:
                baseline_ipc[app] = ipc
        print(f"\n{'=' * width}\n{hdr}\n{'-' * width}")
        for i, (app, tag, instrs, cycles, ipc, status, _, _) in enumerate(perf_rows):
            i_s  = str(instrs) if instrs != -1 else "-"
            c_s  = str(cycles) if cycles != -1 else "-"
            p_s  = f"{ipc:.4f}" if ipc != -1.0 else "-"
            b_s  = f"{baseline_ipc[app]:.4f}" if app in baseline_ipc else "-"
            d_s  = f"{ipc_drops[i]:.2f}%" if ipc_drops[i] >= 0 else "-"
            print(f"{app:<14} {tag:<16} {i_s:>8} {c_s:>8} {p_s:>7} {b_s:>9} {d_s:>7} {status}")
        print(f"{'=' * width}")
        print("  DROP%: (IPC_none - IPC_stalled) / IPC_none * 100")

    else:  # type 2
        hdr = (f"{'APP':<14} {'COMBO':<16} {'INSTRS':>8} {'CYCLES':>8}"
               f" {'IPC':>7} {'THR_REACHED':>12} STATUS")
        width = max(len(hdr) + 4, 60)
        print(f"\n{'=' * width}\n{hdr}\n{'-' * width}")
        for app, tag, instrs, cycles, ipc, status, _, thr in perf_rows:
            i_s = str(instrs) if instrs != -1 else "-"
            c_s = str(cycles) if cycles != -1 else "-"
            p_s = f"{ipc:.4f}" if ipc != -1.0 else "-"
            print(f"{app:<14} {tag:<16} {i_s:>8} {c_s:>8} {p_s:>7} {thr:>12} {status}")
        print(f"{'=' * width}")

    print(f"Logs saved under {LOG_BASE}/")


if __name__ == "__main__":
    main()
