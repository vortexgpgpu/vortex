#!/usr/bin/env python3
import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


RUN_RE = re.compile(r"(?:run_[sa]_|run)(?P<run>\d+)")
META_RE = re.compile(
    r"Sparsity Mode:(?P<mode>\d+)\s*\|\s*"
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
        description=(
            "Plot Matrix A/B sparsity sweep comparing run_s compressed-A stats "
            "with run_a uncompressed-A stats."
        )
    )
    parser.add_argument(
        "stats_dir",
        nargs="?",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Directory containing run_a_*.stat files.",
    )
    parser.add_argument(
        "--run-s-dir",
        default=None,
        type=Path,
        help="Directory containing previous run_s_*.stat files. Defaults to ../run_s.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=Path,
        help=(
            "Output image path. Defaults to "
            "<stats_dir>/sparsity_sweep_comparison.png. A PDF copy is also written."
        ),
    )
    parser.add_argument(
        "--csv",
        default=None,
        type=Path,
        help="Output CSV path. Defaults to <stats_dir>/sparsity_sweep_comparison.csv.",
    )
    parser.add_argument(
        "--metric",
        choices=("total", "body"),
        default="body",
        help="Cycle metric to plot. Default: body.",
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
        help="Plot every Matrix B sparsity found in the stats directories. This is the default.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include stat files whose parsed status is not PASS.",
    )
    return parser.parse_args()


def rounded_percent(value):
    return int(round(value))


def parse_stat(path, dataset):
    run_match = RUN_RE.search(path.name)
    if not run_match:
        raise ValueError(f"{path}: cannot parse run number")

    row = {
        "dataset": dataset,
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
                row["sparsity_mode"] = int(match.group("mode"))
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
            "sparsity_mode",
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


def load_rows(stats_dir, pattern, dataset, include_failed):
    rows = []
    skipped = []
    for path in sorted(stats_dir.glob(pattern)):
        row = parse_stat(path, dataset)
        if include_failed or row["status"] == "PASS":
            rows.append(row)
        else:
            skipped.append(path.name)

    if skipped:
        print(f"warning: skipped non-passing {dataset} stats: {', '.join(skipped)}")
    if not rows:
        raise FileNotFoundError(f"no usable {pattern} files found in {stats_dir}")

    return sorted(rows, key=lambda row: (row["b_label"], row["a_label"], row["run"]))


def write_csv(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
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


def select_b_labels(rows, selected_b_labels):
    b_labels = sorted({row["b_label"] for row in rows if row["b_label"] != 0})
    if selected_b_labels is not None:
        selected = set(selected_b_labels)
        b_labels = [label for label in b_labels if label in selected]
    return b_labels


def build_series(run_s_rows, run_a_rows, selected_b_labels):
    b_labels = select_b_labels(run_s_rows + run_a_rows, selected_b_labels)

    series = []
    for b_label in b_labels:
        compressed_points = sorted(
            (
                row
                for row in run_s_rows
                if row["sparsity_mode"] == 1
                and row["a_label"] == 0
                and row["b_label"] == b_label
            ),
            key=lambda row: row["run"],
        )[:1]
        compressed_points.extend(
            sorted(
                (
                    row
                    for row in run_s_rows
                    if row["sparsity_mode"] == 2
                    and row["b_label"] == b_label
                    and row["a_label"] != 0
                ),
                key=lambda row: (row["a_label"], row["run"]),
            )
        )
        if compressed_points:
            series.append(("A+B compressed", b_label, compressed_points))

        uncompressed_points = sorted(
            (
                row
                for row in run_s_rows
                if row["sparsity_mode"] == 1
                and row["a_label"] == 0
                and row["b_label"] == b_label
            ),
            key=lambda row: row["run"],
        )[:1]
        uncompressed_points.extend(
            sorted(
                (
                    row
                    for row in run_a_rows
                    if row["sparsity_mode"] == 1
                    and row["b_label"] == b_label
                    and row["a_label"] != 0
                ),
                key=lambda row: (row["a_label"], row["run"]),
            )
        )
        if uncompressed_points:
            series.append(("A uncompressed", b_label, uncompressed_points))

    return series


def save_figure(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")

    pdf_output = output.with_suffix(".pdf")
    if pdf_output != output:
        fig.savefig(pdf_output, bbox_inches="tight")
    return pdf_output


def plot(run_s_rows, run_a_rows, output, metric, b_sparsities):
    series = build_series(run_s_rows, run_a_rows, b_sparsities)
    if not series:
        raise ValueError("no series to plot after applying Matrix B sparsity filter")

    metric_key = "total_cycles" if metric == "total" else "body_cycles"
    metric_label = "Total Kernel Cycles"

    colors = {
        20: "#2f6f9f",
        40: "#c7533b",
        60: "#558b2f",
        80: "#7a5ea8",
        90: "#b8792d",
        99: "#4b8f8c",
    }
    fallback_colors = ["#2f6f9f", "#c7533b", "#558b2f", "#7a5ea8", "#b8792d", "#4b8f8c"]
    styles = {
        "A+B compressed": {"linestyle": "-", "marker": "o", "alpha": 0.95},
        "A uncompressed": {"linestyle": "--", "marker": "s", "alpha": 0.85},
    }

    fig, ax = plt.subplots(figsize=(10.6, 6.1))
    for index, (group_label, b_label, points) in enumerate(series):
        x_values = [point["a_label"] for point in points]
        y_values = [point[metric_key] for point in points]
        color = colors.get(b_label, fallback_colors[index % len(fallback_colors)])
        style = styles[group_label]
        ax.plot(
            x_values,
            y_values,
            marker=style["marker"],
            linewidth=2.0,
            markersize=5.2,
            linestyle=style["linestyle"],
            label=f"B {b_label}% sparse, {group_label}",
            color=color,
            alpha=style["alpha"],
        )

    max_value = max(
        point[metric_key]
        for _group_label, _b_label, points in series
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
        title="Matrix B Sparsity / Matrix A Storage",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
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
    run_s_dir = (args.run_s_dir or (stats_dir.parent / "run_s")).resolve()
    output = args.output or stats_dir / "sparsity_sweep_comparison.png"
    csv_output = args.csv or stats_dir / "sparsity_sweep_comparison.csv"

    run_s_rows = load_rows(run_s_dir, "run_s_*.stat", "run_s", args.include_failed)
    run_a_rows = load_rows(stats_dir, "run_a_*.stat", "run_a", args.include_failed)
    write_csv(run_s_rows + run_a_rows, csv_output)
    b_sparsities = None if args.all_b or args.b_sparsities is None else args.b_sparsities
    pdf_output = plot(run_s_rows, run_a_rows, output, args.metric, b_sparsities)
    print(f"wrote {output}")
    print(f"wrote {pdf_output}")
    print(f"wrote {csv_output}")


if __name__ == "__main__":
    main()
