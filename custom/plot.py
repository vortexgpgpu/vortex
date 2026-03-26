#!/usr/bin/env python3
import argparse
import os
import re
import bisect
import matplotlib.pyplot as plt

def find_log_file(name):
    """
    Locate the log file by searching common locations:
    - ../build/<name>
    - Provided path / ./<name>
    - Any subdirectory under the current directory
    """
    candidates = [
        os.path.join("..", "build", name),
        name,  # as provided (may already include a relative path)
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    for root, _, files in os.walk(".", followlinks=False):
        if name in files:
            return os.path.join(root, name)

    return None

def parse_ticks(log_path, pattern):
    """
    Parse ticks from lines that contain `pattern`.
    Assumes tick is the integer at the beginning of the line before ':'.
    """
    ticks = []
    # Accept lines like "TRACE     123: ..." or "   123: ..."
    tick_regex = re.compile(r'^\s*\w+\s+(\d+):|^\s*(\d+):')

    with open(log_path, 'r') as f:
        for line in f:
            if pattern in line:
                m = tick_regex.match(line)
                if m:
                    # m.group(1) is populated for "TRACE  123:", group(2) for " 123:"
                    tick = m.group(1) or m.group(2)
                    ticks.append(int(tick))
    return ticks

def parse_issue_events(log_path, pattern):
    """
    Parse issue events for a given pattern.
    Returns list of (tick, wis, wid). If wid is missing in the log line, wid defaults to 0.
    """
    events = []
    tick_wis_regex = re.compile(
        r'^\s*\w+\s+(\d+):.*sel_wis=(\d+)(?:.*wid=(\d+))?|^\s*(\d+):.*sel_wis=(\d+)(?:.*wid=(\d+))?'
    )
    with open(log_path, 'r') as f:
        for line in f:
            if pattern in line:
                m = tick_wis_regex.match(line)
                if m:
                    # Capture groups 1/2/3 for "TRACE  123:" format, 4/5/6 for " 123:"
                    tick = int(m.group(1) or m.group(4))
                    wis  = int(m.group(2) or m.group(5))
                    wid_str = m.group(3) or m.group(6)
                    wid  = int(wid_str) if wid_str is not None else 0
                    events.append((tick, wis, wid))
    return events

def find_blocks(ticks, min_gap=67):
    """
    Group ticks into blocks. A new block starts whenever the gap between
    consecutive ticks is >= min_gap.
    Returns a list of (start_index, end_index) pairs.
    """
    if not ticks:
        return []

    blocks = []
    start = 0
    for i in range(1, len(ticks)):
        if ticks[i] - ticks[i-1] >= min_gap:
            blocks.append((start, i - 1))
            start = i
    blocks.append((start, len(ticks) - 1))
    return blocks

def main():
    parser = argparse.ArgumentParser(
        description="Plot FEOP vs TCU issue ticks from a Vortex run log."
    )
    parser.add_argument(
        "name",
        help="Log file name; searched in ../build, current dir, and subdirectories."
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional output image filename (e.g. ticks.png). "
             "If omitted, saves to <logname>.png in the current directory."
    )
    parser.add_argument(
        "--trace-warp",
        type=int,
        help="Highlight a specific warp id: draw vertical lines where this warp issues u-ops."
    )
    args = parser.parse_args()

    log_path = find_log_file(args.name)
    if not log_path:
        print(
            "Error: log file not found under ../build, current directory, or any subdirectory."
        )
        return

    # Patterns to search for
    pattern_feop         = "FEOP execution fired"
    pattern_issue_tcu    = "Issuing TCU u-op"
    pattern_issue_other  = "Issuing u-op"
    pattern_issue_load   = "Issuing load u-op"
    pattern_issue_store  = "Issuing store u-op"

    # Extract ticks
    raw_feop_ticks    = parse_ticks(log_path, pattern_feop)
    tcu_events        = parse_issue_events(log_path, pattern_issue_tcu)
    load_events       = parse_issue_events(log_path, pattern_issue_load)
    store_events      = parse_issue_events(log_path, pattern_issue_store)
    other_events      = parse_issue_events(log_path, pattern_issue_other)

    tcu_issue_ticks   = [t for t, _, _ in tcu_events]
    load_issue_ticks  = [t for t, _, _ in load_events]
    store_issue_ticks = [t for t, _, _ in store_events]
    other_issue_ticks = [t for t, _, _ in other_events]

    # Filter out FEOP events before tick 10
    feop_ticks = [t for t in raw_feop_ticks if t >= 10]

    if not feop_ticks and not tcu_issue_ticks and not other_issue_ticks \
       and not load_issue_ticks and not store_issue_ticks:
        print("No matching lines found for any pattern (after filtering FEOP).")
        return

    # -------- Build unified x-axis from ALL issue events (for left plot) --------
    all_issue_events = []
    for t, w, wid in tcu_events:
        all_issue_events.append(("tcu", t, w, wid))
    for t, w, wid in load_events:
        all_issue_events.append(("load", t, w, wid))
    for t, w, wid in store_events:
        all_issue_events.append(("store", t, w, wid))
    for t, w, wid in other_events:
        all_issue_events.append(("other", t, w, wid))

    all_issue_events.sort(key=lambda x: x[1])  # sort by tick

    x_tcu = []
    x_other = []
    x_load = []
    x_store = []
    wis_by_x = []
    wid_by_x = []
    all_issue_ticks_sorted = []

    for idx, (kind, tick, wis, wid) in enumerate(all_issue_events):
        all_issue_ticks_sorted.append(tick)
        wis_by_x.append(wis)
        wid_by_x.append(wid)
        if kind == "tcu":
            x_tcu.append(idx)
        elif kind == "load":
            x_load.append(idx)
        elif kind == "store":
            x_store.append(idx)
        else:
            x_other.append(idx)

    # Map FEOP events onto the same x-axis (nearest issue tick)
    feop_x_unified = []
    if feop_ticks and all_issue_ticks_sorted:
        for t in feop_ticks:
            pos = bisect.bisect_left(all_issue_ticks_sorted, t)
            candidates = []
            if pos > 0:
                candidates.append((abs(t - all_issue_ticks_sorted[pos - 1]), pos - 1))
            if pos < len(all_issue_ticks_sorted):
                candidates.append((abs(t - all_issue_ticks_sorted[pos]), pos))
            best_idx = min(candidates, key=lambda c: c[0])[1]
            feop_x_unified.append(best_idx)

    # For the right plot: simple per-group indices
    feop_x_seq = list(range(len(feop_ticks)))
    tcu_x_seq  = list(range(len(tcu_issue_ticks)))
    feop_gaps  = [(i, feop_ticks[i] - feop_ticks[i-1]) for i in range(1, len(feop_ticks))]
    feop_blocks = find_blocks(feop_ticks) if feop_ticks else []
    # Track context switches (when the selected warp changes)
    def switch_indices(seq):
        return [i for i in range(1, len(seq)) if seq[i] != seq[i-1]]
    wis_switches_all = switch_indices(wis_by_x) if wis_by_x else []
    tcu_wis_seq = [w for _, w, _ in tcu_events]
    wis_switches_tcu = switch_indices(tcu_wis_seq) if tcu_wis_seq else []
    warp_trace = args.trace_warp
    warp_issue_indices = [idx for idx, wid in enumerate(wid_by_x) if wid == warp_trace] if warp_trace is not None else []
    warp_tcu_indices   = [idx for idx, (_, _, wid) in enumerate(tcu_events) if wid == warp_trace] if warp_trace is not None else []

    # ---------------------------
    #      CREATE 2 SUBPLOTS
    # ---------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # ===========================
    # LEFT PLOT = FULL VIEW
    # ===========================
    if other_issue_ticks:
        ax1.plot(
            x_other,
            other_issue_ticks,
            marker='.',
            linestyle='none',
            markersize=1,
            label='Issuing u-op',
            zorder=4
        )

    if load_issue_ticks:
        ax1.plot(
            x_load,
            load_issue_ticks,
            marker='.',
            linestyle='none',
            markersize=2,
            color='red',
            label='Issuing load u-op',
            zorder=4.1
        )

    if store_issue_ticks:
        ax1.plot(
            x_store,
            store_issue_ticks,
            marker='.',
            linestyle='none',
            markersize=2,
            color='black',
            label='Issuing store u-op',
            zorder=4.2
        )

    feop_x_left = feop_x_unified if len(feop_x_unified) == len(feop_ticks) else list(range(len(feop_ticks)))

    if feop_ticks:
        ax1.plot(
            feop_x_left,
            feop_ticks,
            marker='o',
            linestyle='none',
            color='orange',
            label='FEOP execution fired',
            zorder=2
        )

    if tcu_issue_ticks:
        ax1.plot(
            x_tcu,
            tcu_issue_ticks,
            marker='x',
            linestyle='none',
            markersize=4,
            color='g',
            label='Issuing TCU u-op',
            zorder=3
        )

    # Shade regions by wis ownership on the unified x-axis
    if wis_by_x:
        palette = ['#f0f8ff', '#e8f5e9', '#fff3e0', '#f3e5f5', '#e3f2fd', '#fbe9e7', '#ede7f6', '#e8eaf6']
        start = 0
        current = wis_by_x[0]
        for idx in range(1, len(wis_by_x)):
            if wis_by_x[idx] != current:
                ax1.axvspan(start - 0.5, idx - 0.5, color=palette[current % len(palette)], alpha=0.1, zorder=0)
                start = idx
                current = wis_by_x[idx]
        ax1.axvspan(start - 0.5, len(wis_by_x) - 0.5, color=palette[current % len(palette)], alpha=0.1, zorder=0)
        # Vertical dashed lines at context switches (suppressed if tracing a warp)
        if warp_trace is None:
            for sw in wis_switches_all:
                ax1.axvline(sw - 0.5, color='gray', linestyle='--', alpha=0.6, linewidth=0.8, zorder=1.5)

    ax1.set_title("All issues (u-op + TCU + FEOP)")
    ax1.set_xlabel("Issue event index")
    ax1.set_ylabel("Tick")
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend()

    # ===========================
    # RIGHT PLOT = TCU + FEOP ONLY
    # ===========================
    # Plot FEOP (orange) first, then TCU (green) with higher zorder
    if feop_ticks:
        ax2.plot(
            feop_x_seq,
            feop_ticks,
            marker='o',
            linestyle='-',
            color='orange',     # FEOP always orange
            label='FEOP execution fired',
            zorder=1
        )

    if tcu_issue_ticks:
        ax2.plot(
            tcu_x_seq,
            tcu_issue_ticks,
            marker='x',
            linestyle='-',
            color='g',          # TCU always green
            label='Issuing TCU u-op',
            zorder=2            # draw above orange
        )
        # Vertical dashed lines at context switches in TCU stream
        if warp_trace is None:
            for sw in wis_switches_tcu:
                ax2.axvline(sw - 0.5, color='gray', linestyle='--', alpha=0.6, linewidth=0.8, zorder=1.5)

    # Annotate FEOP block vertical extents
    if feop_blocks:
        print("FEOP block extents (ticks):", [(b[0], b[1], feop_ticks[b[0]], feop_ticks[b[1]], feop_ticks[b[1]] - feop_ticks[b[0]]) for b in feop_blocks])
        for start_idx, end_idx in feop_blocks:
            y0 = feop_ticks[start_idx]
            y1 = feop_ticks[end_idx]
            height = y1 - y0
            x_mid = 0.5 * (start_idx + end_idx)
            ax2.vlines(x_mid, y0, y1, colors='orange', linestyles=':', linewidth=1, zorder=1.4)
            ax2.text(
                x_mid,
                y1,
                f"Δ={height}",
                fontsize=9,
                color='black',
                ha='center',
                va='bottom'
            )

    # Highlight warp-specific issue points with vertical lines
    if warp_trace is not None and warp_issue_indices:
        for idx in warp_issue_indices:
            ax1.axvline(idx, color='blue', linestyle='--', alpha=0.8, linewidth=1.0, zorder=2)
    if warp_trace is not None and warp_tcu_indices:
        for idx in warp_tcu_indices:
            ax2.axvline(idx, color='blue', linestyle='--', alpha=0.8, linewidth=1.0, zorder=2)

    # Annotate the largest FEOP gaps (vertical jumps) on the right plot
    if feop_gaps:
        top_gaps = sorted(feop_gaps, key=lambda x: x[1], reverse=True)[:3]
        print("Top FEOP gaps (ticks):", top_gaps)
        for idx, gap in top_gaps:
            x_mid = idx - 0.5  # halfway between the two points on x-axis
            y_mid = 0.5 * (feop_ticks[idx] + feop_ticks[idx-1])
            ax2.text(
                x_mid,
                y_mid,
                f"Δ={gap}",
                fontsize=10,
                color='black',
                ha='center',
                va='bottom'
            )

    ax2.set_title("FEOP vs TCU Issue Ticks (TCU-only view)")
    ax2.set_xlabel("Event index (per group)")
    ax2.set_ylabel("Tick")
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend()

    # ---------------------------
    # Save
    # ---------------------------
    if args.output:
        output_name = args.output
        if not os.path.isabs(output_name):
            output_name = os.path.join(os.path.dirname(log_path), output_name)
    else:
        base = os.path.splitext(os.path.basename(log_path))[0]
        output_name = os.path.join(os.path.dirname(log_path), f"{base}.png")

    plt.tight_layout()
    plt.savefig(output_name, bbox_inches='tight')
    print(f"Saved plot to {output_name}")

if __name__ == "__main__":
    main()
