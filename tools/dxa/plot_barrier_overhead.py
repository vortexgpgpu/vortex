#!/usr/bin/env python3
"""Plot barrier overhead sweeps using release-minus-event completion."""

import argparse
import csv
from pathlib import Path


def load_rows(path):
    with Path(path).open(newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("status") == "PASS"]


def as_float(row, key):
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    return float(value)


def plot(rows, out_dir):
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    rows = sorted(rows, key=lambda r: (int(r["payload_bytes"]), r.get("mode", "")))
    payloads = sorted({int(r["payload_bytes"]) for r in rows})
    labels = [f"{p // 1024}KB" for p in payloads]
    modes = sorted({r.get("mode", "") for r in rows})
    mode_names = {
        "hard_barrier": "hard barrier",
        "soft_smem_atomic": "soft smem+atomic",
        "soft_software_completion": "soft smem+atomic",
    }
    colors = {
        "hard_barrier": "#4C78A8",
        "soft_smem_atomic": "#F58518",
        "soft_software_completion": "#F58518",
    }
    by_key = {(int(r["payload_bytes"]), r.get("mode", "")): r for r in rows}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    x = list(range(len(labels)))
    width = min(0.36, 0.8 / max(1, len(modes)))

    handles = []
    labels_for_legend = []
    for idx, mode in enumerate(modes):
        offset = (idx - (len(modes) - 1) / 2) * width
        overhead = [
            as_float(by_key.get((payload, mode), {}), "barrier_overhead_per_iter")
            for payload in payloads
        ]
        release = [
            as_float(by_key.get((payload, mode), {}), "release_cycles_per_iter")
            for payload in payloads
        ]
        ratio = [
            100.0 * as_float(by_key.get((payload, mode), {}), "overhead_ratio")
            for payload in payloads
        ]
        label = mode_names.get(mode, mode)
        bars = axes[0].bar([i + offset for i in x], release, width,
                           color=colors.get(mode), label=label)
        axes[1].bar([i + offset for i in x], overhead, width,
                    color=colors.get(mode), label=label)
        axes[2].plot(labels, ratio, marker="o",
                     color=colors.get(mode), label=label)
        handles.append(bars[0])
        labels_for_legend.append(label)

    axes[0].set_title("Event Register to Barrier Release")
    axes[0].set_ylabel("cycles / iteration")
    axes[0].set_xlabel("workload payload")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].set_title("True Barrier Overhead")
    axes[1].set_ylabel("release - event completion (cycles / iteration)")
    axes[1].set_xlabel("workload payload")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].set_title("Overhead Share Shrinks With Workload")
    axes[2].set_ylabel("overhead / event duration (%)")
    axes[2].set_xlabel("workload payload")
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("Barrier Cost: Release Latency Context + True Runtime Overhead")
    fig.legend(handles, labels_for_legend, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 0.94))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("hard_soft_barrier_true_overhead", "soft_smem_barrier_overhead"):
        for suffix in (".png", ".svg", ".pdf"):
            fig.savefig(out_dir / f"{name}{suffix}", dpi=200, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/results/dxa_barrier_overhead"))
    args = parser.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit("no PASS rows to plot")
    plot(rows, args.out_dir)


if __name__ == "__main__":
    main()
