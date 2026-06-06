#!/usr/bin/env python3
import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


RUN_RE = re.compile(r"(?:run_k_|run)(?P<run>\d+)")
META_RE = re.compile(
    r"MxNxK=(?P<m>\d+)x(?P<n>\d+)x(?P<k>\d+)\s*\|\s*"
    r"A sparsity=(?P<a>[0-9.]+)%\s*\|\s*"
    r"B sparsity=(?P<b>[0-9.]+)%\s*\|\s*"
    r"It=(?P<dtype>\S+)"
)
FEOP_RE = re.compile(
    r"BLOCK_M=(?P<block_m>\d+),\s*"
    r"BLOCK_N=(?P<block_n>\d+),\s*"
    r"XBAR_QUEUE_DEPTH=(?P<queue_depth>\d+)"
)
STATUS_RE = re.compile(r"^(PASS|FAIL)$")
TCU_UTIL_RE = re.compile(r"^TCU utilization:\s*(?P<value>[0-9.]+)%")
TCU_CYCLES_RE = re.compile(
    r"^TCU cycles \(from first TCU issue to last TCU commit\):\s*(?P<value>\d+)"
)
MUL_ACTIVE_RE = re.compile(r"^MULs active:\s*(?P<value>\d+)")
KERNEL_BODY_RE = re.compile(r"^Kernel body cycles:\s*(?P<value>\d+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot K-size effect on TCU utilization for run_k_1.stat through run_k_8.stat."
    )
    parser.add_argument(
        "stats_dir",
        nargs="?",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Directory containing run_k_1.stat through run_k_8.stat.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=Path,
        help="Output image path. Defaults to <stats_dir>/k_tcu_utilization.png. A PDF copy is also written.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        type=Path,
        help="Output CSV path. Defaults to <stats_dir>/k_tcu_utilization.csv.",
    )
    parser.add_argument(
        "--start-run",
        default=1,
        type=int,
        help="First K-analysis stat number to include. Default: 1.",
    )
    parser.add_argument(
        "--end-run",
        default=8,
        type=int,
        help="Last K-analysis stat number to include. Default: 8.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include stat files whose parsed status or run_all.csv status is not PASS/0.",
    )
    parser.add_argument(
        "--status-csv",
        default=None,
        type=Path,
        help="Manifest CSV used to skip stale/nonzero runs. Defaults to <stats_dir>/run_all.csv or <stats_dir>/../run_all.csv when present.",
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
        "m": None,
        "n": None,
        "k": None,
        "dtype": "",
        "a_sparsity": None,
        "b_sparsity": None,
        "block_m": None,
        "block_n": None,
        "queue_depth": None,
        "kernel_body_cycles": None,
        "tcu_cycles": None,
        "mul_active_cycles": None,
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
                row["m"] = int(match.group("m"))
                row["n"] = int(match.group("n"))
                row["k"] = int(match.group("k"))
                row["a_sparsity"] = float(match.group("a"))
                row["b_sparsity"] = float(match.group("b"))
                row["dtype"] = match.group("dtype")
                continue

            match = FEOP_RE.search(line)
            if match:
                row["block_m"] = int(match.group("block_m"))
                row["block_n"] = int(match.group("block_n"))
                row["queue_depth"] = int(match.group("queue_depth"))
                continue

            match = KERNEL_BODY_RE.match(line)
            if match:
                row["kernel_body_cycles"] = int(match.group("value"))
                continue

            match = TCU_CYCLES_RE.match(line)
            if match:
                row["tcu_cycles"] = int(match.group("value"))
                continue

            match = MUL_ACTIVE_RE.match(line)
            if match:
                row["mul_active_cycles"] = int(match.group("value"))
                continue

            match = TCU_UTIL_RE.match(line)
            if match:
                row["tcu_utilization"] = float(match.group("value"))

    missing = [
        key
        for key in (
            "status",
            "m",
            "n",
            "k",
            "dtype",
            "a_sparsity",
            "b_sparsity",
            "block_m",
            "block_n",
            "queue_depth",
            "kernel_body_cycles",
            "tcu_cycles",
            "mul_active_cycles",
            "tcu_utilization",
        )
        if row[key] in ("", None)
    ]
    if missing:
        raise ValueError(f"{path}: missing required fields: {', '.join(missing)}")

    row["shape"] = f"{row['m']}x{row['n']}x{row['k']}"
    return row


def load_status_csv(path):
    statuses = {}
    if path is None or not path.exists():
        return statuses

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row.get("file", "")
            status = row.get("status", "")
            if not file_name or status == "":
                continue
            stat_name = f"{Path(file_name).stem}.stat"
            if stat_name not in statuses or status == "0":
                statuses[stat_name] = status
    return statuses


def load_rows(stats_dir, start_run, end_run, include_failed, status_csv):
    manifest_statuses = load_status_csv(status_csv)
    rows = []
    missing = []
    skipped = []
    for run in range(start_run, end_run + 1):
        path = stats_dir / f"run_k_{run}.stat"
        if not path.exists():
            legacy_path = stats_dir / f"run{run}.stat"
            if legacy_path.exists():
                path = legacy_path
        if not path.exists():
            missing.append(f"run_k_{run}.stat")
            continue

        row = parse_stat(path)
        row["manifest_status"] = manifest_statuses.get(path.name, "")
        stat_passed = row["status"] == "PASS"
        manifest_passed = row["manifest_status"] in ("", "0")
        if include_failed or (stat_passed and manifest_passed):
            rows.append(row)
        else:
            skipped.append(path.name)

    if missing:
        print(f"warning: missing stats: {', '.join(missing)}")
    if skipped:
        print(f"warning: skipped non-passing or stale stats: {', '.join(skipped)}")
    if not rows:
        raise FileNotFoundError(
            f"no usable run_k_{start_run}.stat through run_k_{end_run}.stat files found in {stats_dir}"
        )
    return sorted(rows, key=lambda row: (row["k"], row["run"]))


def write_csv(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "run",
        "status",
        "manifest_status",
        "shape",
        "dtype",
        "a_sparsity",
        "b_sparsity",
        "block_m",
        "block_n",
        "queue_depth",
        "kernel_body_cycles",
        "tcu_cycles",
        "mul_active_cycles",
        "tcu_utilization",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def save_figure(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")

    pdf_output = output.with_suffix(".pdf")
    if pdf_output != output:
        fig.savefig(pdf_output, bbox_inches="tight")
    return pdf_output


def plot(rows, output):
    k_values = sorted({row["k"] for row in rows})
    step_sizes = sorted(
        {(row["block_m"], row["block_n"]) for row in rows},
        key=lambda item: (item[0] * item[1], item[0], item[1]),
    )
    by_key = {
        (row["k"], row["block_m"], row["block_n"]): row
        for row in rows
    }
    y_values = [row["tcu_utilization"] for row in rows]

    fig_width = max(8.8, 1.35 * len(k_values) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, 5.7))
    x_positions = list(range(len(k_values)))
    colors = ["#2f6f9f", "#c7533b", "#558b2f", "#7a5ea8"]
    group_width = min(0.76, 0.28 * len(step_sizes) + 0.20)
    bar_width = group_width / max(1, len(step_sizes))

    for step_index, (block_m, block_n) in enumerate(step_sizes):
        offsets = [
            x - group_width / 2 + bar_width * (step_index + 0.5)
            for x in x_positions
        ]
        values = []
        present_offsets = []
        for offset, k_value in zip(offsets, k_values):
            row = by_key.get((k_value, block_m, block_n))
            if row is None:
                continue
            present_offsets.append(offset)
            values.append(row["tcu_utilization"])

        bars = ax.bar(
            present_offsets,
            values,
            width=bar_width * 0.88,
            label=f"Step {block_m}x{block_n}",
            color=colors[step_index % len(colors)],
            edgecolor="#222222",
            linewidth=0.8,
        )

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.0,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax.set_xlabel("K Dimension")
    ax.set_ylabel("TCU Utilization (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(k_value) for k_value in k_values])
    ax.set_ylim(0, min(100, max(y_values) * 1.18))
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        title="FEOP step size",
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        loc="lower right",
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
    output = args.output or stats_dir / "k_tcu_utilization.png"
    csv_output = args.csv or stats_dir / "k_tcu_utilization.csv"
    status_csv = args.status_csv
    if status_csv is None:
        default_status_csv = stats_dir / "run_all.csv"
        parent_status_csv = stats_dir.parent / "run_all.csv"
        if default_status_csv.exists():
            status_csv = default_status_csv
        elif parent_status_csv.exists():
            status_csv = parent_status_csv
        else:
            status_csv = None

    rows = load_rows(
        stats_dir,
        args.start_run,
        args.end_run,
        args.include_failed,
        status_csv,
    )
    write_csv(rows, csv_output)
    pdf_output = plot(rows, output)
    print(f"wrote {output}")
    print(f"wrote {pdf_output}")
    print(f"wrote {csv_output}")


if __name__ == "__main__":
    main()
