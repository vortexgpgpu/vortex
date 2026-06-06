#!/usr/bin/env python3
import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


RUN_RE = re.compile(r"(?:run_s_|run)(?P<run>\d+)")
META_RE = re.compile(
    r"(?:Sparsity Mode:(?P<mode>\d+)\s*\|\s*)?"
    r"MxNxK=(?P<shape>\d+x\d+x\d+)\s*\|\s*"
    r"A sparsity=(?P<a>[0-9.]+)%\s*\|\s*"
    r"B sparsity=(?P<b>[0-9.]+)%\s*\|\s*"
    r"It=(?P<dtype>\S+)"
)
STATUS_RE = re.compile(r"^(PASS|FAIL)$")
TOTAL_CYCLES_RE = re.compile(r"^Total kernel cycles:\s*(?P<value>\d+)")
BODY_CYCLES_RE = re.compile(r"^Kernel body cycles:\s*(?P<value>\d+)")
TCU_UTIL_RE = re.compile(r"^TCU utilization:\s*(?P<value>[0-9.]+)%")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Matrix A/B sparsity sweep for run_s_*.stat files."
    )
    parser.add_argument(
        "stats_dir",
        nargs="?",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Directory containing run_s_*.stat files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=Path,
        help="Output image path. Defaults to <stats_dir>/sparsity_sweep.png. A PDF copy is also written.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        type=Path,
        help="Output CSV path. Defaults to <stats_dir>/sparsity_sweep.csv.",
    )
    parser.add_argument(
        "--metric",
        choices=("total", "body"),
        default="body",
        help="Cycle metric to plot. Default: body, labeled as total kernel cycles on the plot.",
    )
    parser.add_argument(
        "--b-sparsities",
        nargs="+",
        default=None,
        type=int,
        help="Optional Matrix B sparsity labels to include, e.g. --b-sparsities 20 60 90.",
    )
    parser.add_argument(
        "--all-b",
        action="store_true",
        help="Plot every Matrix B sparsity found in the stats directory. This is the default.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include stat files whose parsed status is not PASS.",
    )
    return parser.parse_args()


def rounded_percent(value):
    return int(round(value))


def parse_stat(path):
    run_match = RUN_RE.search(path.name)
    if not run_match:
        raise ValueError(f"{path}: cannot parse run number")

    row = {
        "file": path.name,
        "run": int(run_match.group("run")),
        "status": "",
        "sparsity_mode": None,
        "shape": "",
        "dtype": "",
        "a_sparsity": None,
        "b_sparsity": None,
        "a_label": None,
        "b_label": None,
        "total_cycles": None,
        "body_cycles": None,
        "tcu_utilization": None,
    }

    with path.open() as f:
        for line in f:
            line = line.strip()

            match = STATUS_RE.match(line)
            if match:
                row["status"] = match.group(1)
                continue

            match = META_RE.search(line)
            if match:
                row["sparsity_mode"] = (
                    int(match.group("mode")) if match.group("mode") is not None else None
                )
                row["shape"] = match.group("shape")
                row["a_sparsity"] = float(match.group("a"))
                row["b_sparsity"] = float(match.group("b"))
                row["a_label"] = rounded_percent(row["a_sparsity"])
                row["b_label"] = rounded_percent(row["b_sparsity"])
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
        for key in (
            "status",
            "shape",
            "dtype",
            "a_sparsity",
            "b_sparsity",
            "total_cycles",
            "body_cycles",
        )
        if row[key] in ("", None)
    ]
    if missing:
        raise ValueError(f"{path}: missing required fields: {', '.join(missing)}")

    return row


def load_rows(stats_dir, include_failed):
    rows = []
    skipped = []
    for path in sorted(stats_dir.glob("run_s_*.stat")):
        row = parse_stat(path)
        if include_failed or row["status"] == "PASS":
            rows.append(row)
        else:
            skipped.append(path.name)

    if skipped:
        print(f"warning: skipped non-passing stats: {', '.join(skipped)}")
    if not rows:
        raise FileNotFoundError(f"no usable run_s_*.stat files found in {stats_dir}")

    return sorted(rows, key=lambda row: (row["b_label"], row["a_label"], row["run"]))


def write_csv(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "run",
        "status",
        "sparsity_mode",
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


def grouped_series(rows, selected_b_labels):
    b_labels = sorted({row["b_label"] for row in rows if row["b_label"] != 0})
    if selected_b_labels is not None:
        selected = set(selected_b_labels)
        b_labels = [label for label in b_labels if label in selected]

    series = {}
    for b_label in b_labels:
        points = sorted(
            (
                row
                for row in rows
                if row["sparsity_mode"] == 1
                and row["a_label"] == 0
                and row["b_label"] == b_label
            ),
            key=lambda row: row["run"],
        )[:1]
        points.extend(
            sorted(
                (
                    row
                    for row in rows
                    if row["sparsity_mode"] == 2
                    and row["b_label"] == b_label
                    and row["a_label"] != 0
                ),
                key=lambda row: (row["a_label"], row["run"]),
            )
        )
        if points:
            series[b_label] = points

    return series


def save_figure(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")

    pdf_output = output.with_suffix(".pdf")
    if pdf_output != output:
        fig.savefig(pdf_output, bbox_inches="tight")
    return pdf_output


def plot(rows, output, metric, b_sparsities):
    series = grouped_series(rows, b_sparsities)
    if not series:
        raise ValueError("no series to plot after applying Matrix B sparsity filter")

    metric_key = "total_cycles" if metric == "total" else "body_cycles"
    metric_label = "Total Kernel Cycles"

    colors = [
        "#2f6f9f",
        "#c7533b",
        "#558b2f",
        "#7a5ea8",
        "#b8792d",
        "#4b8f8c",
    ]
    markers = ["o", "s", "^", "D", "v", "P"]

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for index, (b_label, points) in enumerate(series.items()):
        x_values = [point["a_label"] for point in points]
        y_values = [point[metric_key] for point in points]
        ax.plot(
            x_values,
            y_values,
            marker=markers[index % len(markers)],
            linewidth=2.0,
            markersize=5.5,
            label=f"Matrix B {b_label}% sparse",
            color=colors[index % len(colors)],
        )

    max_value = max(
        point[metric_key]
        for points in series.values()
        for point in points
    )
    ax.set_xlabel("Matrix A Sparsity (%)")
    ax.set_ylabel(metric_label)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max_value * 1.12)
    ax.set_xticks(range(0, 101, 10))
    ax.grid(True, which="major", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        title="Matrix B Sparsity",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=min(3, max(1, len(series))),
        frameon=True,
        fancybox=False,
        edgecolor="#333333",
    )

    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    pdf_output = save_figure(fig, output)
    plt.close(fig)
    return pdf_output


def main():
    args = parse_args()
    stats_dir = args.stats_dir.resolve()
    output = args.output or stats_dir / "sparsity_sweep.png"
    csv_output = args.csv or stats_dir / "sparsity_sweep.csv"

    rows = load_rows(stats_dir, args.include_failed)
    write_csv(rows, csv_output)
    b_sparsities = None if args.all_b or args.b_sparsities is None else args.b_sparsities
    pdf_output = plot(rows, output, args.metric, b_sparsities)
    print(f"wrote {output}")
    print(f"wrote {pdf_output}")
    print(f"wrote {csv_output}")


if __name__ == "__main__":
    main()
