#!/usr/bin/env python3
import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


RUN_RE = re.compile(r"(?:run_q_|run)(?P<run>\d+)")
META_RE = re.compile(
    r"MxNxK=(?P<shape>\d+x\d+x\d+)\s*\|\s*"
    r"A sparsity=(?P<a>[0-9.]+)%\s*\|\s*"
    r"B sparsity=(?P<b>[0-9.]+)%\s*\|\s*"
    r"It=(?P<dtype>\S+)"
)
FEOP_RE = re.compile(
    r"BLOCK_M=(?P<block_m>\d+),\s*"
    r"BLOCK_N=(?P<block_n>\d+),\s*"
    r"XBAR_QUEUE_DEPTH=(?P<queue_depth>\d+)"
)
XBAR_RE = re.compile(
    r"XBAR stalls:\s*(?P<percent>[0-9.]+)%\s*\((?P<cycles>\d+)\s+cycles\)"
)
STATUS_RE = re.compile(r"^(PASS|FAIL)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot queue-depth effect on crossbar stalls for run_q_1.stat through run_q_16.stat."
    )
    parser.add_argument(
        "stats_dir",
        nargs="?",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Directory containing run_q_1.stat through run_q_16.stat.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=Path,
        help="Output image path. Defaults to <stats_dir>/queue_depth_results.png. A PDF copy is also written.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        type=Path,
        help="Output CSV path. Defaults to <stats_dir>/queue_depth_crossbar_stalls.csv.",
    )
    parser.add_argument(
        "--start-run",
        default=1,
        type=int,
        help="First queue-analysis stat number to include. Default: 1.",
    )
    parser.add_argument(
        "--end-run",
        default=16,
        type=int,
        help="Last queue-analysis stat number to include. Default: 16.",
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
        "shape": "",
        "dtype": "",
        "a_sparsity": None,
        "b_sparsity": None,
        "block_m": None,
        "block_n": None,
        "queue_depth": None,
        "crossbar_stall_percent": None,
        "crossbar_stall_cycles": None,
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
                row["shape"] = match.group("shape")
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

            match = XBAR_RE.search(line)
            if match:
                row["crossbar_stall_percent"] = float(match.group("percent"))
                row["crossbar_stall_cycles"] = int(match.group("cycles"))

    missing = [
        key
        for key in (
            "status",
            "shape",
            "dtype",
            "a_sparsity",
            "b_sparsity",
            "block_m",
            "block_n",
            "queue_depth",
            "crossbar_stall_percent",
            "crossbar_stall_cycles",
        )
        if row[key] in ("", None)
    ]
    if missing:
        raise ValueError(f"{path}: missing required fields: {', '.join(missing)}")

    row["config_label"] = (
        f"A/B {rounded_percent(row['a_sparsity'])}%/{rounded_percent(row['b_sparsity'])}%\n"
        f"STEP {row['block_m']}x{row['block_n']}"
    )
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
            statuses[stat_name] = status
    return statuses


def load_rows(stats_dir, start_run, end_run, include_failed, status_csv):
    manifest_statuses = load_status_csv(status_csv)
    rows = []
    missing = []
    skipped = []
    for run in range(start_run, end_run + 1):
        path = stats_dir / f"run_q_{run}.stat"
        if not path.exists():
            legacy_path = stats_dir / f"run{run}.stat"
            if legacy_path.exists():
                path = legacy_path
        if not path.exists():
            missing.append(f"run_q_{run}.stat")
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
            f"no usable run{start_run}.stat through run{end_run}.stat files found in {stats_dir}"
        )
    return sorted(rows, key=lambda row: (row["queue_depth"], row["config_label"]))


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
        "crossbar_stall_percent",
        "crossbar_stall_cycles",
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
    queue_depths = sorted({row["queue_depth"] for row in rows})
    config_labels = sorted({row["config_label"] for row in rows})
    values = {
        (row["queue_depth"], row["config_label"]): row["crossbar_stall_percent"]
        for row in rows
    }

    width = 0.78 / max(1, len(config_labels))
    x_positions = list(range(len(queue_depths)))

    fig_width = max(9.5, 1.65 * len(queue_depths) + 1.15 * len(config_labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    colors = ["#2f6f9f", "#c7533b", "#3f8f5f", "#8a6bb8", "#b07d2f"]

    for label_index, label in enumerate(config_labels):
        offset = (label_index - (len(config_labels) - 1) / 2.0) * width
        heights = [values.get((queue_depth, label), 0.0) for queue_depth in queue_depths]
        bars = ax.bar(
            [x + offset for x in x_positions],
            heights,
            width=width,
            label=label,
            color=colors[label_index % len(colors)],
            edgecolor="#222222",
            linewidth=0.7,
        )
        for bar, height in zip(bars, heights):
            if height <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.8,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_xlabel("Crossbar Queue Depth")
    ax.set_ylabel("Crossbar Stall Cycles (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(queue_depth) for queue_depth in queue_depths])
    ax.set_ylim(0, max(row["crossbar_stall_percent"] for row in rows) * 1.22)
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        title="Sparsity and Step Dimensions",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=min(len(config_labels), 4),
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
    output = args.output or stats_dir / "queue_depth_results.png"
    csv_output = args.csv or stats_dir / "queue_depth_crossbar_stalls.csv"
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
