#!/usr/bin/env python3
import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


KERNELS = ["sgemm_tcu", "sgemm_tcu_sp", "sgemm_tcu_op"]
KERNEL_LABELS = {
    "sgemm_tcu": "SGEMM",
    "sgemm_tcu_sp": "SGEMM_SP",
    "sgemm_tcu_op": "SGEMM_OP",
}
TARGET_SPARSITIES = [(0, 0), (20, 20), (60, 60), (90, 90)]
PLOT_BARS = [
    ("SGEMM", "sgemm_tcu", 0, 0),
    ("SGEMM_SP", "sgemm_tcu_sp", 0, 0),
    ("SGEMM_OP-0-0", "sgemm_tcu_op", 0, 0),
    ("SGEMM_OP-20-20", "sgemm_tcu_op", 20, 20),
    ("SGEMM_OP-60-60", "sgemm_tcu_op", 60, 60),
    ("SGEMM_OP-90-90", "sgemm_tcu_op", 90, 90),
]

META_RE = re.compile(
    r"Sparsity Mode:(?P<mode>\d+)\s*\|\s*"
    r"MxNxK=(?P<shape>\d+x\d+x\d+)\s*\|\s*"
    r"A sparsity=(?P<a>[0-9.]+)%\s*\|\s*"
    r"B sparsity=(?P<b>[0-9.]+)%\s*\|\s*"
    r"It=(?P<dtype>\S+)"
)
TESTBENCH_RE = re.compile(r"^Testbench:\s*(?P<testbench>\S+)")
BODY_CYCLES_RE = re.compile(r"^Kernel body cycles:\s*(?P<value>\d+)")
BODY_INSTRUCTIONS_RE = re.compile(r"^Kernel body instructions:\s*(?P<value>\d+)")


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Plot kernel body cycles and instructions for SGEMM kernels at "
            "selected A/B sparsities."
        )
    )
    parser.add_argument(
        "--stats-dirs",
        nargs="+",
        default=[root / "stats", root / "run_all"],
        type=Path,
        help="Directories to scan for .stat files. Default: custom/stats custom/run_all.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=root / "stats" / "sgemm_tcu_sparsity_cycles_instructions.png",
        type=Path,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        type=Path,
        help="Output CSV path. Default: same basename as --output with .csv.",
    )
    return parser.parse_args()


def rounded_sparsity(value):
    return int(round(value / 10.0) * 10)


def parse_stat(path):
    row = {
        "file": str(path),
        "testbench": "",
        "shape": "",
        "dtype": "",
        "a_sparsity": None,
        "b_sparsity": None,
        "a_label": None,
        "b_label": None,
        "body_cycles": None,
        "body_instructions": None,
    }

    with path.open() as f:
        for line in f:
            line = line.strip()

            match = TESTBENCH_RE.match(line)
            if match:
                row["testbench"] = match.group("testbench")
                continue

            match = META_RE.search(line)
            if match:
                row["shape"] = match.group("shape")
                row["dtype"] = match.group("dtype")
                row["a_sparsity"] = float(match.group("a"))
                row["b_sparsity"] = float(match.group("b"))
                row["a_label"] = rounded_sparsity(row["a_sparsity"])
                row["b_label"] = rounded_sparsity(row["b_sparsity"])
                continue

            match = BODY_CYCLES_RE.match(line)
            if match:
                row["body_cycles"] = int(match.group("value"))
                continue

            match = BODY_INSTRUCTIONS_RE.match(line)
            if match:
                row["body_instructions"] = int(match.group("value"))

    if row["testbench"] not in KERNELS:
        return None
    if row["body_cycles"] is None or row["body_instructions"] is None:
        return None

    # The baseline IP stats used by oscar_bars.py do not include sparsity
    # metadata; treat them as dense runs.
    if row["a_label"] is None and row["testbench"] in ("sgemm_tcu", "sgemm_tcu_sp"):
        row["a_sparsity"] = 0.0
        row["b_sparsity"] = 0.0
        row["a_label"] = 0
        row["b_label"] = 0

    if row["a_label"] is None or row["b_label"] is None:
        return None
    if (row["a_label"], row["b_label"]) not in TARGET_SPARSITIES:
        return None
    return row


def stat_paths(stats_dirs):
    paths = []
    for stats_dir in stats_dirs:
        paths.extend(sorted(stats_dir.glob("*.stat")))
    return paths


def select_rows(rows):
    selected = {}
    for row in rows:
        key = (row["testbench"], row["a_label"], row["b_label"])
        current = selected.get(key)
        if current is None or score_row(row) > score_row(current):
            selected[key] = row
    return selected


def score_row(row):
    score = 0
    path = Path(row["file"])
    if "run_all" in path.parts:
        score += 100
    if row["dtype"] == "fp8":
        score += 10
    if row["shape"]:
        score += 1
    return score


def write_csv(selected, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kernel",
        "a_sparsity",
        "b_sparsity",
        "body_cycles",
        "body_instructions",
        "shape",
        "dtype",
        "source",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for kernel in KERNELS:
            for a_label, b_label in TARGET_SPARSITIES:
                row = selected.get((kernel, a_label, b_label))
                writer.writerow(
                    {
                        "kernel": kernel,
                        "a_sparsity": a_label,
                        "b_sparsity": b_label,
                        "body_cycles": "" if row is None else row["body_cycles"],
                        "body_instructions": "" if row is None else row["body_instructions"],
                        "shape": "" if row is None else row["shape"],
                        "dtype": "" if row is None else row["dtype"],
                        "source": "" if row is None else row["file"],
                    }
                )


def plot(selected, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = [label for label, _, _, _ in PLOT_BARS]
    x = np.arange(len(PLOT_BARS))
    width = 0.66
    sgemm_cycles = selected.get(("sgemm_tcu", 0, 0), {}).get("body_cycles")
    colors = {
        "sgemm_tcu": "#4C78A8",
        "sgemm_tcu_sp": "#F58518",
        "sgemm_tcu_op": "#54A24B",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), constrained_layout=True)
    for ax, metric, ylabel in (
        (axes[0], "body_cycles", "Kernel body cycles"),
        (axes[1], "body_instructions", "Kernel body instructions"),
    ):
        values = [
            selected.get((kernel, a_label, b_label), {}).get(metric, np.nan)
            for _, kernel, a_label, b_label in PLOT_BARS
        ]
        bar_colors = [colors[kernel] for _, kernel, _, _ in PLOT_BARS]
        bars = ax.bar(x, values, width, color=bar_colors)

        if metric == "body_cycles" and sgemm_cycles:
            for bar, value in zip(bars, values):
                if np.isnan(value) or value == 0:
                    continue
                speedup = sgemm_cycles / value
                ax.annotate(
                    f"{speedup:.2f}x",
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                )
            ax.set_ylim(top=max(values) * 1.16)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=24, ha="right")
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(axis="y", alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_missing(selected):
    missing = []
    for label, kernel, a_label, b_label in PLOT_BARS:
        if (kernel, a_label, b_label) not in selected:
            missing.append(label)
    if missing:
        print("missing data:")
        for item in missing:
            print(f"  {item}")


def main():
    args = parse_args()
    rows = [row for path in stat_paths(args.stats_dirs) if (row := parse_stat(path)) is not None]
    selected = select_rows(rows)
    csv_output = args.csv or args.output.with_suffix(".csv")
    write_csv(selected, csv_output)
    plot(selected, args.output)
    print_missing(selected)
    print(f"wrote {args.output}")
    print(f"wrote {csv_output}")


if __name__ == "__main__":
    main()
