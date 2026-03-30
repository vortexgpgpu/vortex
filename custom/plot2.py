#!/usr/bin/env python3
import argparse
import os
import re
import bisect
import matplotlib.pyplot as plt

TICK_RE = re.compile(r'^\s*\w+\s+(\d+):|^\s*(\d+):')
ISSUE_RE = re.compile(
    r'^\s*\w+\s+(\d+):.*Issuing (?:TCU )?u-op.*sel_wis=(\d+)'
    r'(?:.*ex_type=(\d+))?(?:.*op_type=0x([0-9a-fA-F]+))?'
)
ACCU_C_ACCUM_RE = re.compile(r'\baccumulate_c=1\b')
MATRIX_A_RE = re.compile(r'matrix A:\s*(\d+)\s*x\s*(\d+)', re.IGNORECASE)
MATRIX_B_RE = re.compile(r'matrix B:\s*(\d+)\s*x\s*(\d+)', re.IGNORECASE)
ACTUAL_SPARSITY_RE = re.compile(r'Actual sparsity:\s*([0-9]+(?:\.[0-9]+)?)%?', re.IGNORECASE)
LMEM_READ_RSP_MATRIX_RE = re.compile(r'\bLMEM:\s*read\s*rsp:.*\bmatrix\s*=\s*(Bitmap|A|B|C)\b', re.IGNORECASE)
INPUT_DTYPE_RE = re.compile(r'input data type:\s*([A-Za-z0-9_]+)', re.IGNORECASE)
OUTPUT_DTYPE_RE = re.compile(r'output data type:\s*([A-Za-z0-9_]+)', re.IGNORECASE)
SPARSE_MODE_LEVEL_RE = re.compile(
    r'Sparse mode enabled \(-s\) with sparsity level:\s*([0-2])',
    re.IGNORECASE,
)
SPARSITY_TYPE_RE = re.compile(r'\bsparsity\s*=\s*([0-2])\b', re.IGNORECASE)
BLUE_MARKER_HEX = "0x12345677"
GREEN_MARKER_HEX = "0x12345676"
MARKER_HEX = "0x12345678"
PRE_TCU_MARKER_HEX = "0x12345679"
MARKER_HEXES = {
    GREEN_MARKER_HEX.lower(),
    BLUE_MARKER_HEX.lower(),
    MARKER_HEX.lower(),
    PRE_TCU_MARKER_HEX.lower(),
}
MARKER_RE = re.compile(r'\b(?:0x12345676|0x12345677|0x12345678|0x12345679)\b', re.IGNORECASE)


def find_log_file(name):
    """
    Locate the log file by searching common locations:
    - ../build/<name>
    - Provided path / ./<name>
    - Any subdirectory under the current directory
    """
    candidates = [
        os.path.join("..", "build", name),
        name,
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    for root, _, files in os.walk(".", followlinks=False):
        if name in files:
            return os.path.join(root, name)

    return None


def extract_tick(line):
    m = TICK_RE.match(line)
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def dedup_sorted(values):
    return sorted(set(values))


def parse_issue_events(lines):
    events = []
    tcu_issue_ticks = []

    for line in lines:
        if "Issuing TCU u-op" in line:
            tick = extract_tick(line)
            if tick is not None:
                tcu_issue_ticks.append(tick)
            continue
        if "Issuing u-op" in line:
            m = ISSUE_RE.match(line)
            if not m:
                continue
            tick = int(m.group(1))
            wis = int(m.group(2))
            ex_type = int(m.group(3)) if m.group(3) is not None else None
            op_type = int(m.group(4), 16) if m.group(4) is not None else None
            events.append((tick, wis, ex_type, op_type))

    return tcu_issue_ticks, events


def parse_other_issue_ticks(lines):
    ticks = []
    for line in lines:
        if "Issuing u-op" not in line:
            continue
        if "Issuing load u-op" in line:
            continue
        if "Issuing store u-op" in line:
            continue
        if "Issuing TCU" in line:
            continue
        tick = extract_tick(line)
        if tick is not None:
            ticks.append(tick)
    return ticks


def parse_pattern_ticks(lines, patterns):
    ticks = []
    for line in lines:
        if any(pat in line for pat in patterns):
            tick = extract_tick(line)
            if tick is not None:
                ticks.append(tick)
    return ticks


def parse_regex_ticks(lines, regex):
    ticks = []
    for line in lines:
        if regex.search(line):
            tick = extract_tick(line)
            if tick is not None:
                ticks.append(tick)
    return ticks


def parse_marker_events(lines, regex):
    events = []
    for line in lines:
        if "issue0-dispatcher dispatch:" not in line:
            continue
        if "ex=LSU, op=SW" not in line:
            continue
        if not regex.search(line):
            continue

        rs2_match = re.search(r'rs2_data=\{([^}]*)\}', line)
        tmask_match = re.search(r'tmask=([01]{32})', line)
        if rs2_match is None or tmask_match is None:
            continue

        rs2_vals = [v.strip() for v in rs2_match.group(1).split(",") if v.strip()]
        if not rs2_vals:
            continue

        marker_value = rs2_vals[-1].lower()
        if marker_value not in MARKER_HEXES:
            continue

        # Marker stores are emitted as single-lane scalar stores.
        if tmask_match.group(1).count("1") != 1:
            continue

        tick = extract_tick(line)
        if tick is not None:
            events.append((tick, marker_value))
    return events


def parse_run_metadata(lines):
    meta = {
        "M": None,
        "N": None,
        "K": None,
        "A_sparsity": None,
        "B_sparsity": None,
        "input_dtype": None,
        "output_dtype": None,
        "sparsity_type": None,
    }

    sparsities = []
    for line in lines:
        m_a = MATRIX_A_RE.search(line)
        if m_a:
            m_val, k_val = int(m_a.group(1)), int(m_a.group(2))
            meta["M"] = m_val
            meta["K"] = k_val
            continue

        m_b = MATRIX_B_RE.search(line)
        if m_b:
            k_val, n_val = int(m_b.group(1)), int(m_b.group(2))
            if meta["K"] is None:
                meta["K"] = k_val
            meta["N"] = n_val
            continue

        m_s = ACTUAL_SPARSITY_RE.search(line)
        if m_s:
            sparsities.append(f"{m_s.group(1)}%")
            continue

        m_in_dtype = INPUT_DTYPE_RE.search(line)
        if m_in_dtype:
            meta["input_dtype"] = m_in_dtype.group(1)
            continue

        m_out_dtype = OUTPUT_DTYPE_RE.search(line)
        if m_out_dtype:
            meta["output_dtype"] = m_out_dtype.group(1)
            continue

        m_sparse_mode = SPARSE_MODE_LEVEL_RE.search(line)
        if m_sparse_mode and meta["sparsity_type"] is None:
            meta["sparsity_type"] = int(m_sparse_mode.group(1))
            continue

        m_sparsity_type = SPARSITY_TYPE_RE.search(line)
        if m_sparsity_type and meta["sparsity_type"] is None:
            meta["sparsity_type"] = int(m_sparsity_type.group(1))

    if len(sparsities) >= 1:
        meta["A_sparsity"] = sparsities[0]
    if len(sparsities) >= 2:
        meta["B_sparsity"] = sparsities[1]

    return meta


def classify_mem_line(line):
    s = line.lower()
    if "mem:" in s:
        if "read request" in s:
            return "rd_req"
        if "write request" in s:
            return "wr_req"
        if "read rsp" in s or "read response" in s:
            return "rd_rsp"
        if "write rsp" in s or "write response" in s:
            return "wr_rsp"
    if not any(tok in s for tok in ("mem", "lmem", "cache", "dcache", "icache", "bank", "core")):
        return None
    if "rd_req" in s:
        return "rd_req"
    if "wr_req" in s:
        return "wr_req"
    if "rd_rsp" in s:
        return "rd_rsp"
    if "wr_rsp" in s:
        return "wr_rsp"
    if "rd-rsp" in s or "rsp-rd" in s:
        return "rd_rsp"
    if "wr-rsp" in s or "rsp-wr" in s:
        return "wr_rsp"
    if "rd-req" in s or "req-rd" in s:
        return "rd_req"
    if "wr-req" in s or "req-wr" in s:
        return "wr_req"
    return None


def parse_mem_events(lines, include_re=None, exclude_re=None):
    mem = {
        "rd_req_global": [],
        "wr_req_global": [],
        "rd_rsp_global": [],
        "wr_rsp_global": [],
        "rd_req_lmem": [],
        "wr_req_lmem": [],
        "rd_rsp_lmem": [],
        "wr_rsp_lmem": [],
    }
    for line in lines:
        if include_re and not include_re.search(line):
            continue
        if exclude_re and exclude_re.search(line):
            continue
        kind = classify_mem_line(line)
        if not kind:
            continue
        if "mem:" in line.lower() or "lmem" in line.lower():
            scope = "lmem"
        else:
            scope = "global"
        tick = extract_tick(line)
        if tick is None:
            continue
        mem[f"{kind}_{scope}"].append(tick)
    return mem


def parse_lmem_read_rsp_matrix_events(lines, include_re=None, exclude_re=None):
    events = []
    for line in lines:
        if include_re and not include_re.search(line):
            continue
        if exclude_re and exclude_re.search(line):
            continue
        m = LMEM_READ_RSP_MATRIX_RE.search(line)
        if not m:
            continue
        tick = extract_tick(line)
        if tick is None:
            continue
        events.append((tick, m.group(1)))
    return events


def auto_detect_tcu_ex_type(issue_events):
    ex_counts = {}
    for _, _, ex_type, _ in issue_events:
        if ex_type is None:
            continue
        ex_counts[ex_type] = ex_counts.get(ex_type, 0) + 1
    if not ex_counts:
        return None
    # Prefer the most frequent non-zero ex_type (avoids ALU-heavy streams).
    non_zero = {k: v for k, v in ex_counts.items() if k != 0}
    if non_zero:
        return max(non_zero.items(), key=lambda kv: kv[1])[0]
    return max(ex_counts.items(), key=lambda kv: kv[1])[0]


def pair_tcu_issue_commit_ticks(issue_ticks, commit_ticks):
    """
    Greedily pair each issue tick with the first commit tick at or after it.
    Preserves chronological ordering and avoids reusing commits.
    """
    issues = sorted(issue_ticks)
    commits = sorted(commit_ticks)
    pairs = []
    ci = 0
    for it in issues:
        while ci < len(commits) and commits[ci] < it:
            ci += 1
        if ci >= len(commits):
            break
        pairs.append((it, commits[ci]))
        ci += 1
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Plot TCU/FEOP/memory events from a Vortex run log."
    )
    parser.add_argument(
        "name",
        help="Log file name; searched in ../build, current dir, and subdirectories."
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional output image filename (e.g. ticks.png). "
             "If omitted, saves to <logname>.png in the log directory."
    )
    parser.add_argument(
        "--tcu-ex-type",
        type=int,
        help="ex_type value to treat as TCU issue when 'Issuing TCU u-op' is absent."
    )
    parser.add_argument(
        "--tcu-op-type",
        type=lambda s: int(s, 0),
        default=0x3,
        help="Filter TCU issue events by op_type (hex or decimal). Default: 0x3 (TCU MMA_OP)."
    )
    parser.add_argument(
        "--feop-pattern",
        action="append",
        default=["FEOP-enq", "FEOP execution fired"],
        help="Substring to identify FEOP assignment lines. Repeatable."
    )
    parser.add_argument(
        "--mem-include",
        default=r"MEM:|LMEM",
        help="Regex to include only matching memory lines (default: MEM:|LMEM)."
    )
    parser.add_argument(
        "--mem-exclude",
        help="Regex to exclude matching memory lines."
    )
    parser.add_argument(
        "--no-dedup-mem",
        action="store_true",
        help="Do not de-duplicate memory events per tick."
    )
    parser.add_argument(
        "--no-dedup-feop",
        action="store_true",
        help="Do not de-duplicate FEOP events per tick."
    )
    args = parser.parse_args()

    log_path = find_log_file(args.name)
    if not log_path:
        print("Error: log file not found under ../build, current directory, or any subdirectory.")
        return

    with open(log_path, "r") as f:
        lines = f.readlines()
    error_count = sum(len(re.findall(r"\berror\b", line, flags=re.IGNORECASE)) for line in lines)
    has_errors = error_count > 0
    run_meta = parse_run_metadata(lines)

    # Issue events (TCU + all scoreboard issues)
    tcu_issue_ticks = parse_pattern_ticks(lines, ["Issuing TCU MMA_OP"])
    tcu_commit_ticks = parse_regex_ticks(lines, re.compile(r'commit:.*\bex=TCU\b'))
    # Stage markers: only consider explicit MARKER writes (0x12345678).
    marker_events = parse_marker_events(lines, MARKER_RE)
    _, issue_events = parse_issue_events(lines)
    load_issue_ticks = parse_pattern_ticks(lines, ["Issuing load u-op"])
    store_issue_ticks = parse_pattern_ticks(lines, ["Issuing store u-op"])
    other_issue_ticks = parse_other_issue_ticks(lines)
    c_accum_ticks = parse_regex_ticks(lines, ACCU_C_ACCUM_RE)
    xbar_stall_ticks = parse_pattern_ticks(lines, [
        "[feop_accu]: ERROR: xbar queues are full - must stall",
        "[feop_accu]: xbar queues are full - must stall",
    ])

    tcu_ex_type = args.tcu_ex_type
    if tcu_ex_type is None and not tcu_issue_ticks:
        tcu_ex_type = auto_detect_tcu_ex_type(issue_events)
        if tcu_ex_type is not None:
            print(f"Auto-detected TCU ex_type={tcu_ex_type} (most frequent non-zero ex_type)")

    for tick, _, ex_type, op_type in issue_events:
        if args.tcu_op_type is not None and op_type != args.tcu_op_type:
            continue
        if tcu_ex_type is not None and ex_type != tcu_ex_type:
            continue
        tcu_issue_ticks.append(tick)
    if tcu_issue_ticks:
        tcu_issue_ticks = dedup_sorted(tcu_issue_ticks)
    if tcu_commit_ticks:
        tcu_commit_ticks = dedup_sorted(tcu_commit_ticks)
    marker_ticks = [t for t, value in marker_events if value == MARKER_HEX.lower()]
    if marker_ticks:
        marker_ticks = dedup_sorted(marker_ticks)
    green_marker_ticks = [t for t, value in marker_events if value == GREEN_MARKER_HEX.lower()]
    if green_marker_ticks:
        green_marker_ticks = dedup_sorted(green_marker_ticks)
    blue_marker_ticks = [t for t, value in marker_events if value == BLUE_MARKER_HEX.lower()]
    if blue_marker_ticks:
        blue_marker_ticks = dedup_sorted(blue_marker_ticks)
    pre_tcu_marker_ticks = [t for t, value in marker_events if value == PRE_TCU_MARKER_HEX.lower()]
    if pre_tcu_marker_ticks:
        pre_tcu_marker_ticks = dedup_sorted(pre_tcu_marker_ticks)

    first_tcu_tick = min(tcu_issue_ticks) if tcu_issue_ticks else None
    last_tcu_commit = max(tcu_commit_ticks) if tcu_commit_ticks else None

    # FEOP assignment (or fallback to FEOP accu activity)
    feop_ticks = parse_pattern_ticks(lines, args.feop_pattern)
    if not feop_ticks:
        feop_ticks = parse_pattern_ticks(lines, ["[feop_accu]"])
        if feop_ticks:
            print("FEOP patterns not found; using [feop_accu] activity as FEOP work markers.")

    # Memory events
    include_re = re.compile(args.mem_include) if args.mem_include else None
    exclude_re = re.compile(args.mem_exclude) if args.mem_exclude else None
    mem = parse_mem_events(lines, include_re=include_re, exclude_re=exclude_re)
    lmem_read_rsp_matrix_events = parse_lmem_read_rsp_matrix_events(
        lines, include_re=include_re, exclude_re=exclude_re
    )
    mem_right = {k: list(v) for k, v in mem.items()}
    lmem_read_rsp_matrix_events_right = list(lmem_read_rsp_matrix_events)
    if first_tcu_tick is not None:
        mem_right["wr_req_global"] = [t for t in mem_right["wr_req_global"] if t >= first_tcu_tick]
        mem_right["wr_req_lmem"] = [t for t in mem_right["wr_req_lmem"] if t >= first_tcu_tick]
        lmem_read_rsp_matrix_events_right = [
            (t, m) for (t, m) in lmem_read_rsp_matrix_events_right if t >= first_tcu_tick
        ]

    if not args.no_dedup_mem:
        mem = {k: dedup_sorted(v) for k, v in mem.items()}
    feop_ticks = [t for t in feop_ticks if t >= 4]
    if not args.no_dedup_feop:
        feop_ticks = dedup_sorted(feop_ticks)
    if xbar_stall_ticks:
        xbar_stall_ticks = dedup_sorted(xbar_stall_ticks)
    if other_issue_ticks:
        other_issue_ticks = dedup_sorted(other_issue_ticks)

    # FEOP window metrics: from first FEOP-assigned tick to last FEOP-assigned tick.
    feop_first_tick = min(feop_ticks) if feop_ticks else None
    feop_last_tick = max(feop_ticks) if feop_ticks else None
    feop_cycles = 0
    feop_assigned = len(feop_ticks)
    xbar_stall_cycles = 0
    stall_pct = 0.0
    if feop_first_tick is not None and feop_last_tick is not None:
        feop_cycles = (feop_last_tick - feop_first_tick + 1)
        xbar_stalls_in_feop = [
            t for t in xbar_stall_ticks
            if feop_first_tick <= t <= feop_last_tick
        ]
        xbar_stall_cycles = len(xbar_stalls_in_feop)
        denom = xbar_stall_cycles + feop_assigned
        if denom > 0:
            stall_pct = (xbar_stall_cycles / denom) * 100.0

    # TCU issue->commit stall metrics (per paired TCU op).
    tcu_pairs = pair_tcu_issue_commit_ticks(tcu_issue_ticks, tcu_commit_ticks)
    tcu_total_cycles = 0
    tcu_active_cycles = 0
    tcu_stall_cycles = 0
    tcu_stall_pct = None
    active_ticks_tcu = dedup_sorted(
        mem_right["rd_req_lmem"]
        + mem_right["wr_req_global"]
        + mem_right["wr_req_lmem"]
        + c_accum_ticks
        + feop_ticks
    )
    for issue_t, commit_t in tcu_pairs:
        if commit_t < issue_t:
            continue
        total_cycles_this = ((commit_t - issue_t) // 2) + 1
        if total_cycles_this <= 0:
            continue
        active_cycle_slots = set()
        for t in active_ticks_tcu:
            if t < issue_t or t > commit_t:
                continue
            active_cycle_slots.add((t - issue_t) // 2)
        active_cycles_this = min(len(active_cycle_slots), total_cycles_this)
        stall_cycles_this = total_cycles_this - active_cycles_this

        tcu_total_cycles += total_cycles_this
        tcu_active_cycles += active_cycles_this
        tcu_stall_cycles += stall_cycles_this
    if tcu_total_cycles > 0:
        tcu_stall_pct = (tcu_stall_cycles / tcu_total_cycles) * 100.0

    if not any([
        tcu_issue_ticks,
        feop_ticks,
        mem["rd_req_global"],
        mem["wr_req_global"],
        mem["rd_rsp_global"],
        mem["wr_rsp_global"],
        mem["rd_req_lmem"],
        mem["wr_req_lmem"],
        mem["rd_rsp_lmem"],
        mem["wr_rsp_lmem"],
        tcu_commit_ticks,
        marker_ticks,
        green_marker_ticks,
        blue_marker_ticks,
        xbar_stall_ticks,
        other_issue_ticks,
        load_issue_ticks,
        store_issue_ticks,
    ]):
        print("No matching events found (TCU/FEOP/memory).")
        return

    if marker_events:
        marker_events_sorted = sorted(marker_events, key=lambda x: x[0])
        unique_markers = []
        seen = set()
        for tick, value in marker_events_sorted:
            key = (tick, value)
            if key in seen:
                continue
            seen.add(key)
            unique_markers.append((tick, value))
        print("Marker ticks:")
        for tick, value in unique_markers:
            print(f"  {value} @ tick {tick}")

    marker_to_tcu_delta = None
    pre_tcu_marker_tick = None
    pre_tcu_issue_tick = None
    if pre_tcu_marker_ticks and tcu_issue_ticks:
        pre_tcu_marker_tick = pre_tcu_marker_ticks[0]
        pre_tcu_issue_tick = next((t for t in tcu_issue_ticks if t >= pre_tcu_marker_tick), None)
        if pre_tcu_issue_tick is not None:
            marker_to_tcu_delta = pre_tcu_issue_tick - pre_tcu_marker_tick

    # -------- Build unified x-axis from ALL issue events (left plot) --------
    issue_events = []
    for t in tcu_issue_ticks:
        issue_events.append(("tcu", t))
    for t in load_issue_ticks:
        issue_events.append(("load", t))
    for t in store_issue_ticks:
        issue_events.append(("store", t))
    for t in other_issue_ticks:
        issue_events.append(("other", t))

    issue_events.sort(key=lambda x: x[1])

    x_pos = {"tcu": [], "load": [], "store": [], "other": [], "feop": [], "rd_req": [], "wr_req": [], "rd_rsp": [], "wr_rsp": []}
    y_pos = {"tcu": [], "load": [], "store": [], "other": [], "feop": [], "rd_req": [], "wr_req": [], "rd_rsp": [], "wr_rsp": []}
    issue_ticks_sorted = []

    for idx, (kind, tick) in enumerate(issue_events):
        issue_ticks_sorted.append(tick)
        x_pos[kind].append(idx)
        y_pos[kind].append(tick)

    # Map FEOP/memory events onto issue index for the left plot
    def map_to_issue_index(ticks):
        mapped = []
        if not issue_ticks_sorted:
            return mapped
        for t in ticks:
            pos = bisect.bisect_left(issue_ticks_sorted, t)
            candidates = []
            if pos > 0:
                candidates.append((abs(t - issue_ticks_sorted[pos - 1]), pos - 1))
            if pos < len(issue_ticks_sorted):
                candidates.append((abs(t - issue_ticks_sorted[pos]), pos))
            best_idx = min(candidates, key=lambda c: c[0])[1]
            mapped.append(best_idx)
        return mapped

    feop_x_left = map_to_issue_index(feop_ticks)
    marker_x_left = map_to_issue_index(marker_ticks)
    green_marker_x_left = map_to_issue_index(green_marker_ticks)
    blue_marker_x_left = map_to_issue_index(blue_marker_ticks)
    rd_req_x_left_g = map_to_issue_index(mem["rd_req_global"])
    wr_req_x_left_g = map_to_issue_index(mem["wr_req_global"])
    rd_rsp_x_left_g = map_to_issue_index(mem["rd_rsp_global"])
    wr_rsp_x_left_g = map_to_issue_index(mem["wr_rsp_global"])
    rd_req_x_left_l = map_to_issue_index(mem["rd_req_lmem"])
    wr_req_x_left_l = map_to_issue_index(mem["wr_req_lmem"])
    rd_rsp_x_left_l = map_to_issue_index(mem["rd_rsp_lmem"])
    wr_rsp_x_left_l = map_to_issue_index(mem["wr_rsp_lmem"])

    # For the right plot: unified TCU timeline, one index per unique tick
    tcu_ticks = dedup_sorted(
        tcu_issue_ticks
        + feop_ticks
        + mem_right["rd_req_global"]
        + mem_right["wr_req_global"]
        + mem_right["rd_rsp_global"]
        + mem_right["wr_rsp_global"]
        + mem_right["rd_req_lmem"]
        + mem_right["wr_req_lmem"]
        + mem_right["rd_rsp_lmem"]
        + mem_right["wr_rsp_lmem"]
        + tcu_commit_ticks
        + marker_ticks
        + green_marker_ticks
        + c_accum_ticks
        + xbar_stall_ticks
    )
    tcu_ticks = [t for t in tcu_ticks if t >= 4]
    if first_tcu_tick is not None:
        tcu_ticks = [t for t in tcu_ticks if t >= first_tcu_tick]
    tcu_tick_to_idx = {t: i for i, t in enumerate(tcu_ticks)}
    right_tick0 = first_tcu_tick if first_tcu_tick is not None else (min(tcu_ticks) if tcu_ticks else 0)

    def rel_tick(t):
        return t - right_tick0

    # Detect bubble jumps on the right plot: consecutive unique-tick deltas > 2.
    # Exclude jumps that include xbar queue stall ticks.
    non_xbar_jump_annotations = []
    if len(tcu_ticks) >= 2:
        xbar_sorted = xbar_stall_ticks if xbar_stall_ticks else []
        for i in range(1, len(tcu_ticks)):
            prev_t = tcu_ticks[i - 1]
            curr_t = tcu_ticks[i]
            jump_ticks = curr_t - prev_t
            if jump_ticks <= 2:
                continue
            lo = bisect.bisect_right(xbar_sorted, prev_t)
            hi = bisect.bisect_right(xbar_sorted, curr_t)
            has_xbar_stall_in_jump = hi > lo
            if has_xbar_stall_in_jump:
                continue
            non_xbar_jump_annotations.append((i, rel_tick(curr_t), jump_ticks))

    tcu_x_pos = {
        "tcu": [],
        "tcu_commit": [],
        "feop": [],
        "rd_req_g": [],
        "wr_req_g": [],
        "rd_rsp_g": [],
        "wr_rsp_g": [],
        "rd_req_l": [],
        "wr_req_l": [],
        "rd_rsp_l": [],
        "wr_rsp_l": [],
    }
    tcu_y_pos = {
        "tcu": [],
        "tcu_commit": [],
        "feop": [],
        "rd_req_g": [],
        "wr_req_g": [],
        "rd_rsp_g": [],
        "wr_rsp_g": [],
        "rd_req_l": [],
        "wr_req_l": [],
        "rd_rsp_l": [],
        "wr_rsp_l": [],
    }
    for t in tcu_issue_ticks:
        if t in tcu_tick_to_idx:
            tcu_x_pos["tcu"].append(tcu_tick_to_idx[t])
            tcu_y_pos["tcu"].append(rel_tick(t))
    for t in tcu_commit_ticks:
        if t in tcu_tick_to_idx:
            tcu_x_pos["tcu_commit"].append(tcu_tick_to_idx[t])
            tcu_y_pos["tcu_commit"].append(rel_tick(t))
    for t in feop_ticks:
        if t in tcu_tick_to_idx:
            tcu_x_pos["feop"].append(tcu_tick_to_idx[t])
            tcu_y_pos["feop"].append(rel_tick(t))
    for t in mem_right["rd_req_global"]:
        if t in tcu_tick_to_idx:
            tcu_x_pos["rd_req_g"].append(tcu_tick_to_idx[t])
            tcu_y_pos["rd_req_g"].append(rel_tick(t))
    for t in mem_right["wr_req_global"]:
        if t in tcu_tick_to_idx:
            tcu_x_pos["wr_req_g"].append(tcu_tick_to_idx[t])
            tcu_y_pos["wr_req_g"].append(rel_tick(t))
    for t in mem_right["rd_rsp_global"]:
        if t in tcu_tick_to_idx:
            tcu_x_pos["rd_rsp_g"].append(tcu_tick_to_idx[t])
            tcu_y_pos["rd_rsp_g"].append(rel_tick(t))
    for t in mem_right["wr_rsp_global"]:
        if t in tcu_tick_to_idx:
            tcu_x_pos["wr_rsp_g"].append(tcu_tick_to_idx[t])
            tcu_y_pos["wr_rsp_g"].append(rel_tick(t))
    for t in mem_right["rd_req_lmem"]:
        if t in tcu_tick_to_idx:
            tcu_x_pos["rd_req_l"].append(tcu_tick_to_idx[t])
            tcu_y_pos["rd_req_l"].append(rel_tick(t))
    for t in mem_right["wr_req_lmem"]:
        if t in tcu_tick_to_idx:
            tcu_x_pos["wr_req_l"].append(tcu_tick_to_idx[t])
            tcu_y_pos["wr_req_l"].append(rel_tick(t))
    for t in mem_right["rd_rsp_lmem"]:
        if t in tcu_tick_to_idx:
            tcu_x_pos["rd_rsp_l"].append(tcu_tick_to_idx[t])
            tcu_y_pos["rd_rsp_l"].append(rel_tick(t))
    for t in mem_right["wr_rsp_lmem"]:
        if t in tcu_tick_to_idx:
            tcu_x_pos["wr_rsp_l"].append(tcu_tick_to_idx[t])
            tcu_y_pos["wr_rsp_l"].append(rel_tick(t))
    c_accum_x_right = [tcu_tick_to_idx[t] for t in c_accum_ticks if t in tcu_tick_to_idx]
    xbar_stall_points_right = [(tcu_tick_to_idx[t], rel_tick(t)) for t in xbar_stall_ticks if t in tcu_tick_to_idx]
    lmem_rsp_matrix_vlines_right = []
    for t, matrix_name in lmem_read_rsp_matrix_events_right:
        if t not in tcu_tick_to_idx:
            continue
        lmem_rsp_matrix_vlines_right.append((tcu_tick_to_idx[t], matrix_name))

    # ---------------------------
    #      CREATE 2 SUBPLOTS
    # ---------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # LEFT PLOT = unified timeline
    if mem["wr_req_global"]:
        ax1.plot(wr_req_x_left_g, mem["wr_req_global"], marker="v", linestyle="none",
                 markersize=2, color="tab:blue", label="Mem write req (global)", zorder=8)
    if mem["wr_rsp_global"]:
        ax1.plot(wr_rsp_x_left_g, mem["wr_rsp_global"], marker="D", linestyle="none",
                 markersize=2, color="tab:purple", label="Mem write rsp (global)", zorder=6)
    if mem["wr_req_lmem"]:
        ax1.plot(wr_req_x_left_l, mem["wr_req_lmem"], marker="v", linestyle="none",
                 markersize=2, color="blue", label="Mem write req (lmem)", zorder=8)
    if mem["wr_rsp_lmem"]:
        ax1.plot(wr_rsp_x_left_l, mem["wr_rsp_lmem"], marker="D", linestyle="none",
                 markersize=2, color="indigo", label="Mem write rsp (lmem)", zorder=6)
    if feop_ticks:
        ax1.plot(feop_x_left, feop_ticks, marker="o", linestyle="none",
                 markersize=3, color="orange", label="FEOP work assigned", zorder=4)
    if y_pos["other"]:
        ax1.plot(x_pos["other"], y_pos["other"], marker=".", linestyle="none",
                 markersize=1, color="gray", label="Issuing u-op", zorder=3)
    if y_pos["load"]:
        ax1.plot(x_pos["load"], y_pos["load"], marker=".", linestyle="none",
                 markersize=2, color="red", label="Issuing load u-op", zorder=3.2)
    if y_pos["store"]:
        ax1.plot(x_pos["store"], y_pos["store"], marker=".", linestyle="none",
                 markersize=2, color="black", label="Issuing store u-op", zorder=3.3)
    if y_pos["tcu"]:
        ax1.plot(x_pos["tcu"], y_pos["tcu"], marker="x", linestyle="none",
                 markersize=4, color="g", label="TCU op issued", zorder=3.5)
    if marker_x_left:
        for x in marker_x_left:
            ax1.axvline(x, color="black", linestyle="--", alpha=0.7, linewidth=0.8, zorder=1.2)
    if green_marker_x_left:
        for x in green_marker_x_left:
            ax1.axvline(x, color="green", linestyle="--", alpha=0.7, linewidth=0.8, zorder=1.2)
    if blue_marker_x_left:
        for x in blue_marker_x_left:
            ax1.axvline(x, color="blue", linestyle="--", alpha=0.7, linewidth=0.8, zorder=1.2)
    if marker_to_tcu_delta is not None and pre_tcu_marker_tick is not None and pre_tcu_issue_tick is not None:
        marker_x = map_to_issue_index([pre_tcu_marker_tick])[0]
        tcu_x = map_to_issue_index([pre_tcu_issue_tick])[0]
        ax1.axhline(
            y=pre_tcu_marker_tick,
            color="black",
            linestyle="--",
            alpha=0.28,
            linewidth=0.8,
            zorder=1.1,
        )
        ax1.axhline(
            y=pre_tcu_issue_tick,
            color="black",
            linestyle="--",
            alpha=0.28,
            linewidth=0.8,
            zorder=1.1,
        )
        mid_x = (marker_x + tcu_x) / 2
        mid_y = (pre_tcu_marker_tick + pre_tcu_issue_tick) / 2
        ax1.text(
            mid_x,
            mid_y,
            f"{marker_to_tcu_delta}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.45, pad=0.1),
            zorder=9,
        )

    ax1.set_title("All scoreboard issues (with FEOP + TCU MEM)")
    ax1.set_xlabel("Issue event index")
    ax1.set_ylabel("Tick")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()

    # RIGHT PLOT = TCU-only timeline
    if tcu_y_pos["rd_req_g"]:
        ax2.plot(tcu_x_pos["rd_req_g"], tcu_y_pos["rd_req_g"], marker="^", linestyle="none",
                 markersize=2, color="tab:blue", label="Mem read req (global)", zorder=6)
    if tcu_y_pos["wr_req_g"]:
        ax2.plot(tcu_x_pos["wr_req_g"], tcu_y_pos["wr_req_g"], marker="v", linestyle="none",
                 markersize=2, color="tab:red", label="Mem write req (global)", zorder=6)
    if tcu_y_pos["rd_rsp_g"]:
        ax2.plot(tcu_x_pos["rd_rsp_g"], tcu_y_pos["rd_rsp_g"], marker="s", linestyle="none",
                 markersize=2, color="tab:cyan", label="Mem read rsp (global)", zorder=6)
    if tcu_y_pos["wr_rsp_g"]:
        ax2.plot(tcu_x_pos["wr_rsp_g"], tcu_y_pos["wr_rsp_g"], marker="D", linestyle="none",
                 markersize=2, color="tab:purple", label="Mem write rsp (global)", zorder=6)
    if tcu_y_pos["rd_req_l"]:
        ax2.plot(tcu_x_pos["rd_req_l"], tcu_y_pos["rd_req_l"], marker="^", linestyle="none",
                 markersize=2, color="dodgerblue", label="Mem read req (lmem)", zorder=6)
    if tcu_y_pos["wr_req_l"]:
        ax2.plot(tcu_x_pos["wr_req_l"], tcu_y_pos["wr_req_l"], marker="v", linestyle="none",
                 markersize=2, color="darkred", label="Mem write req (lmem)", zorder=6)
    if tcu_y_pos["rd_rsp_l"]:
        ax2.plot(tcu_x_pos["rd_rsp_l"], tcu_y_pos["rd_rsp_l"], marker="o", linestyle="none",
                 markersize=1.5, color="black", label="Mem read rsp (lmem)", zorder=6)
    if tcu_y_pos["wr_rsp_l"]:
        ax2.plot(tcu_x_pos["wr_rsp_l"], tcu_y_pos["wr_rsp_l"], marker="D", linestyle="none",
                 markersize=2, color="indigo", label="Mem write rsp (lmem)", zorder=6)
    if tcu_y_pos["feop"]:
        ax2.plot(tcu_x_pos["feop"], tcu_y_pos["feop"], marker="o", linestyle="none",
                 markersize=3, color="orange", label="FEOP work assigned", zorder=4)
    if tcu_y_pos["tcu"]:
        ax2.plot(tcu_x_pos["tcu"], tcu_y_pos["tcu"], marker="x", linestyle="none",
                 markersize=4, color="g", label="TCU op issued", zorder=7)
    if tcu_y_pos["tcu_commit"]:
        ax2.plot(tcu_x_pos["tcu_commit"], tcu_y_pos["tcu_commit"], marker="x", linestyle="none",
                 markersize=4, color="black", label="TCU op commit", zorder=7.5)
        first = True
        for y in dedup_sorted(tcu_y_pos["tcu_commit"]):
            ax2.axhline(
                y=y,
                color="black",
                linestyle="--",
                alpha=0.35,
                linewidth=0.8,
                label="TCU commit tick" if first else None,
                zorder=1.5,
            )
            ax2.text(
                0.5,
                y,
                f"{y}",
                transform=ax2.get_yaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2),
                zorder=9,
            )
            first = False
    if c_accum_x_right:
        first = True
        for x in c_accum_x_right:
            ax2.axvline(x, color="gray", linestyle="--", alpha=0.6, linewidth=0.9,
                        label="C block accumulated" if first else None, zorder=2)
            first = False
    if lmem_rsp_matrix_vlines_right:
        matrix_line_style = {
            "A": ("red", "LMEM rsp: A"),
            "B": ("blue", "LMEM rsp: B"),
            "Bitmap": ("green", "LMEM rsp: Bitmap"),
        }
        shown_labels = set()
        for x, matrix_name in lmem_rsp_matrix_vlines_right:
            raw = matrix_name.strip().lower()
            key = "Bitmap" if raw == "bitmap" else raw.upper()
            if key not in matrix_line_style:
                continue
            color, label = matrix_line_style[key]
            draw_label = label if label not in shown_labels else None
            ax2.axvline(
                x,
                color=color,
                linestyle=":",
                alpha=0.8,
                linewidth=1.0,
                label=draw_label,
                zorder=2.2,
            )
            if draw_label is not None:
                shown_labels.add(label)
    if xbar_stall_points_right:
        first = True
        for x, y in xbar_stall_points_right:
            ax2.hlines(y=y, xmin=x - 0.35, xmax=x + 0.35, colors="orange",
                       linestyles=":", linewidth=1.0,
                       label="xbar queue stall" if first else None, zorder=8)
            first = False

    title_parts = ["TCU-only timeline (TCU issue + FEOP + MEM)"]
    details = []
    if run_meta["A_sparsity"] is not None or run_meta["B_sparsity"] is not None:
        a_sp = run_meta["A_sparsity"] if run_meta["A_sparsity"] is not None else "?"
        b_sp = run_meta["B_sparsity"] if run_meta["B_sparsity"] is not None else "?"
        details.append(f"A sparsity={a_sp}, B sparsity={b_sp}")
    if (run_meta["M"] is not None
            and run_meta["N"] is not None
            and run_meta["K"] is not None):
        details.append(f"MxNxK={run_meta['M']}x{run_meta['N']}x{run_meta['K']}")
    if run_meta["input_dtype"] is not None:
        details.append(f"It={run_meta['input_dtype']}")
    if run_meta["output_dtype"] is not None:
        details.append(f"Ot={run_meta['output_dtype']}")
    if run_meta["sparsity_type"] is not None:
        details.append(f"Sparsity Mode:{run_meta['sparsity_type']}")
    if details:
        title_parts.append(" | ".join(details))
    ax2.set_title("\n".join(title_parts))
    ax2.set_xlabel("Event index (unique TCU ticks)")
    ax2.set_ylabel("Tick since first TCU issue (cycles)")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, linestyle="--", alpha=0.3)

    # Label non-xbar vertical jumps with the jump size in ticks.
    for x, y, jump_ticks in non_xbar_jump_annotations:
        ax2.annotate(
            f"{jump_ticks}",
            xy=(x, y),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.2),
            zorder=9,
        )

    ax2.legend()

    # Add FEOP/xbar summary at the bottom of the figure.
    if feop_assigned > 0:
        summary = (
            f"xbar_queue_stall cycles={xbar_stall_cycles} | "
            f"FEOP assigned={feop_assigned} | "
            f"XBAR STALLS % = {stall_pct:.2f}%"
        )
    else:
        summary = "xbar_queue_stall cycles=0 | FEOP assigned=0 | XBAR STALLS % = N/A"
    if tcu_stall_pct is not None:
        tcu_summary = (
            f"TCU total cycles={tcu_total_cycles} | "
            f"active cycles={tcu_active_cycles} | "
            f"stall cycles={tcu_stall_cycles} | "
            f"TCU STALL % = {tcu_stall_pct:.2f}%"
        )
    else:
        tcu_summary = "TCU total cycles=0 | active cycles=0 | stall cycles=0 | TCU STALL % = N/A"
    summary_y = 0.01
    if has_errors:
        fig.text(0.5, summary_y, f"{error_count} ERRORS FOUND", ha="center", va="bottom",
                 fontsize=10, color="red")
        summary_y = 0.04
    fig.text(0.5, summary_y + 0.03, summary, ha="center", va="bottom", fontsize=10)
    fig.text(0.5, summary_y, tcu_summary, ha="center", va="bottom", fontsize=10)

    # ---------------------------
    # Save
    # ---------------------------
    output_dir = os.path.join(os.path.dirname(__file__), "run_OP_MEM")
    os.makedirs(output_dir, exist_ok=True)
    if args.output:
        output_name = os.path.basename(args.output)
        output_name = os.path.join(output_dir, output_name)
    else:
        base = os.path.splitext(os.path.basename(log_path))[0]
        output_name = os.path.join(output_dir, f"{base}_plot2.png")

    bottom_margin = 0.13 if has_errors else 0.09
    plt.tight_layout(rect=[0, bottom_margin, 1, 1])
    plt.savefig(output_name, bbox_inches="tight")
    print(f"Saved plot to {output_name}")


if __name__ == "__main__":
    main()
