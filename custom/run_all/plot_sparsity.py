#!/usr/bin/env python3
import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


META_RE = re.compile(
    r"Sparsity Mode:(?P<mode>\d+)\s*\|\s*"
    r"MxNxK=(?P<shape>\d+x\d+x\d+)\s*\|\s*"
    r"A sparsity=(?P<a>[0-9.]+)%\s*\|\s*"
    r"B sparsity=(?P<b>[0-9.]+)%\s*\|\s*"
    r"It=(?P<dtype>\S+)"
)
TESTBENCH_RE = re.compile(r"^Testbench:\s*(?P<testbench>\S+)")
TOTAL_CYCLES_RE = re.compile(r"^Total kernel cycles:\s*(?P<value>\d+)")
BODY_CYCLES_RE = re.compile(r"^Kernel body cycles:\s*(?P<value>\d+)")
TCU_UTIL_RE = re.compile(r"^TCU utilization:\s*(?P<value>[0-9.]+)%")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot run_all sparsity sweep stats with linear axes."
    )
    parser.add_argument(
        "stats_dir",
        nargs="?",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Directory containing run*.stat files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=Path,
        help="Output PNG path. Defaults to <stats_dir>/run_all_sparsity_linear.png.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        type=Path,
        help="Output CSV path. Defaults to <stats_dir>/run_all_sparsity_linear.csv.",
    )
    parser.add_argument(
        "--metric",
        choices=("total", "body"),
        default="body",
        help="Cycle metric to plot.",
    )
    return parser.parse_args()


def rounded_sparsity(value):
    return round(value / 10.0) * 10


def parse_stat(path):
    row = {
        "file": path.name,
        "run": int(re.search(r"run(\d+)", path.name).group(1)),
        "testbench": "",
        "mode": None,
        "shape": "",
        "a_sparsity": None,
        "b_sparsity": None,
        "dtype": "",
        "total_cycles": None,
        "body_cycles": None,
        "tcu_utilization": None,
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
                row["mode"] = int(match.group("mode"))
                row["shape"] = match.group("shape")
                row["a_sparsity"] = float(match.group("a"))
                row["b_sparsity"] = float(match.group("b"))
                row["dtype"] = match.group("dtype")
                continue

            match = TOTAL_CYCLES_RE.match(line)
            if match:
                row["total_cycles"] = int(match.group("value"))
                continue

            match = BODY_CYCLES_RE.match(line)
            if match:
                row["body_cycles"] = int(match.group("value"))
                continue

            match = TCU_UTIL_RE.match(line)
            if match:
                row["tcu_utilization"] = float(match.group("value"))

    missing = [
        key
        for key in ("mode", "a_sparsity", "b_sparsity", "total_cycles", "body_cycles")
        if row[key] is None
    ]
    if missing:
        raise ValueError(f"{path}: missing required fields: {', '.join(missing)}")

    row["a_label"] = rounded_sparsity(row["a_sparsity"])
    row["b_label"] = rounded_sparsity(row["b_sparsity"])
    return row


def load_rows(stats_dir):
    rows = [parse_stat(path) for path in sorted(stats_dir.glob("run*.stat"))]
    if not rows:
        raise FileNotFoundError(f"no run*.stat files found in {stats_dir}")
    return sorted(rows, key=lambda row: row["run"])


def write_csv(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "run",
        "testbench",
        "shape",
        "dtype",
        "a_sparsity",
        "b_sparsity",
        "a_label",
        "b_label",
        "total_cycles",
        "body_cycles",
        "tcu_utilization",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def grouped_series(rows):
    dense_rows = [row for row in rows if row["a_label"] == 0 and row["b_label"] == 0]
    dense = dense_rows[0] if dense_rows else None
    b_labels = sorted({row["b_label"] for row in rows if row["b_label"] != 0})
    series = {}

    for b_label in b_labels:
        points = []
        if dense is not None:
            points.append(dense)
        points.extend(
            sorted(
                (
                    row
                    for row in rows
                    if row["b_label"] == b_label and row["a_label"] != 0
                ),
                key=lambda row: row["a_sparsity"],
            )
        )
        series[b_label] = points

    return series


def plot(rows, output, metric):
    series = grouped_series(rows)
    metric_key = "total_cycles" if metric == "total" else "body_cycles"
    metric_label = "Total Kernel Cycles"
    shape = next((row["shape"] for row in rows if row["shape"]), "")
    dtype = next((row["dtype"] for row in rows if row["dtype"]), "")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    colors = {
        20: "#1f77b4",
        60: "#d62728",
        90: "#2ca02c",
    }
    markers = {
        20: "o",
        60: "s",
        90: "^",
    }

    for b_label, points in series.items():
        x_values = [point["a_sparsity"] for point in points]
        y_values = [point[metric_key] for point in points]
        ax.plot(
            x_values,
            y_values,
            marker=markers.get(b_label, "o"),
            linewidth=2.0,
            markersize=6,
            label=f"Matrix B {b_label}% sparse",
            color=colors.get(b_label),
        )

    ax.set_title(
        f"SGEMM TCU OP Sparsity Sweep - Matrix operation: {shape}, Input type: {dtype}"
    )
    ax.set_xlabel("Matrix A Sparsity (%)")
    ax.set_ylabel(metric_label)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(row[metric_key] for row in rows) * 1.12)
    ax.set_xticks(range(0, 101, 10))
    ax.grid(True, which="major", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        title="Matrix B Sparsity",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=max(1, len(series)),
        frameon=True,
        fancybox=False,
        edgecolor="#333333",
    )

    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    stats_dir = args.stats_dir.resolve()
    output = args.output or stats_dir / "run_all_sparsity_linear.png"
    csv_output = args.csv or stats_dir / "run_all_sparsity_linear.csv"
    rows = load_rows(stats_dir)
    write_csv(rows, csv_output)
    plot(rows, output, args.metric)
    print(f"wrote {output}")
    print(f"wrote {csv_output}")


if __name__ == "__main__":
    main()
