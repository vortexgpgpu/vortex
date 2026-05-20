#!/usr/bin/env python3
import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


RUN_RE = re.compile(r"(?:run_b_|run)(?P<run>\d+)")
TESTBENCH_RE = re.compile(r"^Testbench:\s*(?P<testbench>\S+)")
META_RE = re.compile(
    r"MxNxK=(?P<shape>\d+x\d+x\d+).*?\|\s*It=(?P<dtype>\S+)"
)
SPARSITY_RE = re.compile(
    r"A sparsity=(?P<a_sparsity>[0-9.]+)%\s*\|\s*B sparsity=(?P<b_sparsity>[0-9.]+)%"
)
STATUS_RE = re.compile(r"^(PASS|FAIL)$")
TOTAL_INSTRUCTIONS_RE = re.compile(r"^Total kernel instructions:\s*(?P<value>\d+)")
KERNEL_BODY_RE = re.compile(r"^Kernel body cycles:\s*(?P<value>\d+)")
TCU_UTIL_RE = re.compile(r"^TCU utilization:\s*(?P<value>[0-9.]+)%")
MEMORY_REQS_RE = re.compile(
    r"^(?:PERF2:\s*)?memory:\s*reqs=(?P<total>\d+)\s*"
    r"\(r=(?P<reads>\d+),\s*w=(?P<writes>\d+)\)"
)

KERNELS = ["sgemm_tcu", "sgemm_tcu_sp", "sgemm_tcu_op"]
KERNEL_LABELS = {
    "sgemm_tcu": "SGEMM TCU",
    "sgemm_tcu_sp": "SGEMM TCU SP",
    "sgemm_tcu_op": "SGEMM TCU OP",
}
DTYPE_ORDER = ["fp32", "fp16", "fp8"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot run_b stats, including dtype comparisons and the sgemm_tcu_op sparsity sweep."
    )
    parser.add_argument(
        "stats_dir",
        nargs="?",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Directory containing run_b_*.stat files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=Path,
        help="Output image path. Defaults to <stats_dir>/input_dtype_kernel_body_cycles.png. A PDF copy is also written.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        type=Path,
        help="Output CSV path. Defaults to <stats_dir>/input_dtype_kernel_body_cycles.csv.",
    )
    parser.add_argument(
        "--instructions-output",
        default=None,
        type=Path,
        help="Output image path for fp16 total instructions. Defaults to <stats_dir>/fp16_total_instructions.png. A PDF copy is also written.",
    )
    parser.add_argument(
        "--instructions-csv",
        default=None,
        type=Path,
        help="Output CSV path for fp16 total instructions. Defaults to <stats_dir>/fp16_total_instructions.csv.",
    )
    parser.add_argument(
        "--utilization-output",
        default=None,
        type=Path,
        help="Output image path for TCU utilization. Defaults to <stats_dir>/input_dtype_tcu_utilization.png. A PDF copy is also written.",
    )
    parser.add_argument(
        "--utilization-csv",
        default=None,
        type=Path,
        help="Output CSV path for TCU utilization. Defaults to <stats_dir>/input_dtype_tcu_utilization.csv.",
    )
    parser.add_argument(
        "--memory-output",
        default=None,
        type=Path,
        help="Output image path for memory requests. Defaults to <stats_dir>/memory_requests_read_write.png. A PDF copy is also written.",
    )
    parser.add_argument(
        "--memory-csv",
        default=None,
        type=Path,
        help="Output CSV path for memory requests. Defaults to <stats_dir>/memory_requests_read_write.csv.",
    )
    parser.add_argument(
        "--sparsity-output",
        default=None,
        type=Path,
        help="Output image path for sgemm_tcu_op sparsity cycles. Defaults to <stats_dir>/sparsity_kernel_body_cycles.png. A PDF copy is also written.",
    )
    parser.add_argument(
        "--sparsity-csv",
        default=None,
        type=Path,
        help="Output CSV path for sgemm_tcu_op sparsity metrics. Defaults to <stats_dir>/sparsity_sgemm_tcu_op.csv.",
    )
    parser.add_argument(
        "--sparsity-utilization-output",
        default=None,
        type=Path,
        help="Output image path for sgemm_tcu_op sparsity utilization. Defaults to <stats_dir>/sparsity_tcu_utilization.png. A PDF copy is also written.",
    )
    parser.add_argument(
        "--sparsity-memory-output",
        default=None,
        type=Path,
        help="Output image path for sgemm_tcu_op sparsity memory requests. Defaults to <stats_dir>/sparsity_memory_requests.png. A PDF copy is also written.",
    )
    parser.add_argument(
        "--sparsity-instructions-output",
        default=None,
        type=Path,
        help="Output image path for sgemm_tcu_op sparsity instructions. Defaults to <stats_dir>/sparsity_total_instructions.png. A PDF copy is also written.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        default=True,
        help="Include failed stats when cycle metrics are present. Default: enabled.",
    )
    return parser.parse_args()


def parse_stat(path):
    run_match = RUN_RE.search(path.name)
    if not run_match:
        raise ValueError(f"{path}: cannot parse run number")

    row = {
        "file": path.name,
        "run": int(run_match.group("run")),
        "testbench": "",
        "status": "",
        "shape": "",
        "dtype": "",
        "a_sparsity": None,
        "b_sparsity": None,
        "total_instructions": None,
        "kernel_body_cycles": None,
        "tcu_utilization": None,
        "memory_reqs": None,
        "memory_reads": None,
        "memory_writes": None,
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
                sparsity_match = SPARSITY_RE.search(line)
                if sparsity_match:
                    row["a_sparsity"] = float(sparsity_match.group("a_sparsity"))
                    row["b_sparsity"] = float(sparsity_match.group("b_sparsity"))
                continue

            match = STATUS_RE.match(line)
            if match:
                row["status"] = match.group(1)
                continue

            match = TOTAL_INSTRUCTIONS_RE.match(line)
            if match:
                row["total_instructions"] = int(match.group("value"))
                continue

            match = KERNEL_BODY_RE.match(line)
            if match:
                row["kernel_body_cycles"] = int(match.group("value"))
                continue

            match = TCU_UTIL_RE.match(line)
            if match:
                row["tcu_utilization"] = float(match.group("value"))
                continue

            match = MEMORY_REQS_RE.match(line)
            if match and row["memory_reqs"] is None:
                row["memory_reqs"] = int(match.group("total"))
                row["memory_reads"] = int(match.group("reads"))
                row["memory_writes"] = int(match.group("writes"))

    missing = [
        key
        for key in (
            "testbench",
            "status",
            "shape",
            "dtype",
            "total_instructions",
            "kernel_body_cycles",
            "tcu_utilization",
            "memory_reqs",
            "memory_reads",
            "memory_writes",
        )
        if row[key] in ("", None)
    ]
    if missing:
        raise ValueError(f"{path}: missing required fields: {', '.join(missing)}")

    return row


def load_rows(stats_dir, include_failed, required_runs):
    rows = []
    skipped = []
    for run in required_runs:
        path = stats_dir / f"run_b_{run}.stat"
        if not path.exists():
            raise FileNotFoundError(f"missing required stat file: {path}")
        try:
            row = parse_stat(path)
        except ValueError as err:
            print(f"warning: skipped incomplete stat: {err}")
            continue
        if row["testbench"] not in KERNELS:
            continue
        if row["dtype"] not in DTYPE_ORDER:
            continue
        if include_failed or row["status"] == "PASS":
            rows.append(row)
        else:
            skipped.append(path.name)

    if skipped:
        print(f"warning: skipped non-passing stats: {', '.join(skipped)}")
    if not rows:
        run_list = ", ".join(f"run_b_{run}.stat" for run in required_runs)
        raise FileNotFoundError(f"no usable {run_list} files found in {stats_dir}")
    return sorted(rows, key=lambda row: (DTYPE_ORDER.index(row["dtype"]), KERNELS.index(row["testbench"])))


def load_dtype_rows(stats_dir, include_failed):
    return load_rows(stats_dir, include_failed, range(1, 10))


def load_all_rows(stats_dir, include_failed):
    return load_rows(stats_dir, include_failed, range(1, 13))


def write_csv(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "run",
        "testbench",
        "status",
        "shape",
        "dtype",
        "a_sparsity",
        "b_sparsity",
        "total_instructions",
        "kernel_body_cycles",
        "tcu_utilization",
        "memory_reqs",
        "memory_reads",
        "memory_writes",
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


def write_fp16_instructions_csv(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "run",
        "testbench",
        "status",
        "shape",
        "dtype",
        "total_instructions",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in fp16_rows(rows):
            writer.writerow({key: row[key] for key in fieldnames})


def fp16_rows(rows):
    by_kernel = {
        row["testbench"]: row
        for row in rows
        if row["dtype"] == "fp16" and row["testbench"] in KERNELS
    }
    missing = [kernel for kernel in KERNELS if kernel not in by_kernel]
    if missing:
        raise ValueError(f"missing fp16 stats for kernels: {', '.join(missing)}")
    return [by_kernel[kernel] for kernel in KERNELS]


def plot_fp16_instructions(rows, output):
    selected = fp16_rows(rows)
    values = [row["total_instructions"] for row in selected]
    shape = selected[0]["shape"]
    colors = ["#2f6f9f", "#c7533b", "#558b2f", "#7a5ea8"]
    x_positions = list(range(len(KERNELS)))

    fig, ax = plt.subplots(figsize=(8.8, 5.7))
    ax.bar(
        x_positions,
        values,
        width=0.58,
        color=[colors[index % len(colors)] for index in x_positions],
        edgecolor="#222222",
        linewidth=0.8,
    )

    ax.set_title(f"FP16 Total Kernel Instructions by Kernel\nMxNxK={shape}")
    ax.set_xlabel("Kernel")
    ax.set_ylabel("Total Kernel Instructions")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([KERNEL_LABELS[kernel] for kernel in KERNELS])
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    pdf_output = save_figure(fig, output)
    plt.close(fig)
    return pdf_output


def write_utilization_csv(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "run",
        "testbench",
        "status",
        "shape",
        "dtype",
        "tcu_utilization",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def plot_utilization(rows, output):
    by_key = {(row["dtype"], row["testbench"]): row for row in rows}
    values = [row["tcu_utilization"] for row in rows]
    shape = next((row["shape"] for row in rows if row["shape"]), "")

    fig, ax = plt.subplots(figsize=(8.8, 5.7))
    x_positions = list(range(len(DTYPE_ORDER)))
    colors = ["#2f6f9f", "#c7533b", "#558b2f", "#7a5ea8"]
    group_width = min(0.76, 0.28 * len(KERNELS) + 0.20)
    bar_width = group_width / len(KERNELS)

    for kernel_index, kernel in enumerate(KERNELS):
        offsets = [
            x - group_width / 2 + bar_width * (kernel_index + 0.5)
            for x in x_positions
        ]
        present_offsets = []
        bar_values = []
        for offset, dtype in zip(offsets, DTYPE_ORDER):
            row = by_key.get((dtype, kernel))
            if row is None:
                continue
            present_offsets.append(offset)
            bar_values.append(row["tcu_utilization"])

        ax.bar(
            present_offsets,
            bar_values,
            width=bar_width * 0.88,
            label=KERNEL_LABELS[kernel],
            color=colors[kernel_index % len(colors)],
            edgecolor="#222222",
            linewidth=0.8,
        )

    ax.set_title(f"TCU Utilization by Input Data Type\nMxNxK={shape}")
    ax.set_xlabel("Input Data Type")
    ax.set_ylabel("TCU Utilization (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(DTYPE_ORDER)
    ax.set_ylim(0, min(100, max(values) * 1.18))
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        title="Kernel",
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(KERNELS),
    )

    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    pdf_output = save_figure(fig, output)
    plt.close(fig)
    return pdf_output


def write_memory_csv(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "run",
        "testbench",
        "status",
        "shape",
        "dtype",
        "memory_reqs",
        "memory_reads",
        "memory_writes",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def plot_memory_requests(rows, output):
    ordered_rows = sorted(
        rows,
        key=lambda row: (DTYPE_ORDER.index(row["dtype"]), KERNELS.index(row["testbench"])),
    )
    x_positions = list(range(len(ordered_rows)))
    writes = [row["memory_writes"] for row in ordered_rows]
    reads = [row["memory_reads"] for row in ordered_rows]
    totals = [row["memory_reqs"] for row in ordered_rows]
    shape = next((row["shape"] for row in rows if row["shape"]), "")

    fig_width = max(10.8, 0.86 * len(ordered_rows) + 3.2)
    fig, ax = plt.subplots(figsize=(fig_width, 5.7))
    ax.bar(
        x_positions,
        writes,
        width=0.62,
        label="Writes",
        color="#c7533b",
        edgecolor="#222222",
        linewidth=0.8,
    )
    ax.bar(
        x_positions,
        reads,
        width=0.62,
        bottom=writes,
        label="Reads",
        color="#2f6f9f",
        edgecolor="#222222",
        linewidth=0.8,
    )

    labels = [
        f"{row['dtype']}\n{KERNEL_LABELS[row['testbench']].replace('SGEMM ', '')}"
        for row in ordered_rows
    ]
    ax.set_title(f"Memory Requests by Input Data Type and Kernel\nMxNxK={shape}")
    ax.set_xlabel("Input Data Type / Kernel")
    ax.set_ylabel("Memory Requests")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(totals) * 1.15)
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        title="Request Type",
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=2,
    )

    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    pdf_output = save_figure(fig, output)
    plt.close(fig)
    return pdf_output


def sparsity_rows(rows):
    selected = [
        row
        for row in rows
        if row["testbench"] == "sgemm_tcu_op"
        and row["dtype"] == "fp8"
        and row["a_sparsity"] is not None
        and row["b_sparsity"] is not None
    ]
    if not selected:
        raise ValueError("missing sgemm_tcu_op fp8 sparsity stats")
    return sorted(selected, key=lambda row: (row["a_sparsity"] + row["b_sparsity"]) / 2.0)


def sparsity_label(row):
    avg_sparsity = (row["a_sparsity"] + row["b_sparsity"]) / 2.0
    return f"{avg_sparsity:.0f}%"


def write_sparsity_csv(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "run",
        "testbench",
        "status",
        "shape",
        "dtype",
        "a_sparsity",
        "b_sparsity",
        "total_instructions",
        "kernel_body_cycles",
        "tcu_utilization",
        "memory_reqs",
        "memory_reads",
        "memory_writes",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sparsity_rows(rows):
            writer.writerow({key: row[key] for key in fieldnames})


def plot_sparsity_metric(rows, output, metric, title, ylabel, color):
    selected = sparsity_rows(rows)
    values = [row[metric] for row in selected]
    x_positions = list(range(len(selected)))
    labels = [sparsity_label(row) for row in selected]
    shape = selected[0]["shape"]

    fig, ax = plt.subplots(figsize=(8.8, 5.7))
    ax.plot(
        x_positions,
        values,
        marker="o",
        markersize=7,
        linewidth=2.2,
        color=color,
    )
    ax.scatter(
        x_positions,
        values,
        s=58,
        color=color,
        edgecolor="#222222",
        linewidth=0.8,
        zorder=3,
    )

    ax.set_title(f"{title}\nMxNxK={shape}, Kernel=SGEMM TCU OP, It=fp8")
    ax.set_xlabel("Average A/B Sparsity")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    pdf_output = save_figure(fig, output)
    plt.close(fig)
    return pdf_output


def plot_sparsity_memory(rows, output):
    selected = sparsity_rows(rows)
    x_positions = list(range(len(selected)))
    writes = [row["memory_writes"] for row in selected]
    reads = [row["memory_reads"] for row in selected]
    totals = [row["memory_reqs"] for row in selected]
    labels = [sparsity_label(row) for row in selected]
    shape = selected[0]["shape"]

    fig, ax = plt.subplots(figsize=(8.8, 5.7))
    ax.bar(
        x_positions,
        writes,
        width=0.58,
        label="Writes",
        color="#c7533b",
        edgecolor="#222222",
        linewidth=0.8,
    )
    ax.bar(
        x_positions,
        reads,
        width=0.58,
        bottom=writes,
        label="Reads",
        color="#2f6f9f",
        edgecolor="#222222",
        linewidth=0.8,
    )

    ax.set_title(f"Memory Requests by Sparsity\nMxNxK={shape}, Kernel=SGEMM TCU OP, It=fp8")
    ax.set_xlabel("Average A/B Sparsity")
    ax.set_ylabel("Memory Requests")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(totals) * 1.15)
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        title="Request Type",
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
    )

    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.0)

    fig.tight_layout()
    pdf_output = save_figure(fig, output)
    plt.close(fig)
    return pdf_output


def plot(rows, output):
    by_key = {(row["dtype"], row["testbench"]): row for row in rows}
    values = [row["kernel_body_cycles"] for row in rows]
    shape = next((row["shape"] for row in rows if row["shape"]), "")

    fig, ax = plt.subplots(figsize=(8.8, 5.7))
    x_positions = list(range(len(DTYPE_ORDER)))
    colors = ["#2f6f9f", "#c7533b", "#558b2f", "#7a5ea8"]
    group_width = min(0.76, 0.28 * len(KERNELS) + 0.20)
    bar_width = group_width / len(KERNELS)

    for kernel_index, kernel in enumerate(KERNELS):
        offsets = [
            x - group_width / 2 + bar_width * (kernel_index + 0.5)
            for x in x_positions
        ]
        present_offsets = []
        bar_values = []
        for offset, dtype in zip(offsets, DTYPE_ORDER):
            row = by_key.get((dtype, kernel))
            if row is None:
                continue
            present_offsets.append(offset)
            bar_values.append(row["kernel_body_cycles"])

        bars = ax.bar(
            present_offsets,
            bar_values,
            width=bar_width * 0.88,
            label=KERNEL_LABELS[kernel],
            color=colors[kernel_index % len(colors)],
            edgecolor="#222222",
            linewidth=0.8,
        )

    ax.set_title(f"Total Kernel Cycles by Input Data Type\nMxNxK={shape}")
    ax.set_xlabel("Input Data Type")
    ax.set_ylabel("Total Kernel Cycles")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(DTYPE_ORDER)
    ax.set_ylim(0, max(values) * 1.22)
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        title="Kernel",
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(KERNELS),
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
    output = args.output or stats_dir / "input_dtype_kernel_body_cycles.png"
    csv_output = args.csv or stats_dir / "input_dtype_kernel_body_cycles.csv"
    instructions_output = args.instructions_output or stats_dir / "fp16_total_instructions.png"
    instructions_csv = args.instructions_csv or stats_dir / "fp16_total_instructions.csv"
    utilization_output = args.utilization_output or stats_dir / "input_dtype_tcu_utilization.png"
    utilization_csv = args.utilization_csv or stats_dir / "input_dtype_tcu_utilization.csv"
    memory_output = args.memory_output or stats_dir / "memory_requests_read_write.png"
    memory_csv = args.memory_csv or stats_dir / "memory_requests_read_write.csv"
    sparsity_output = args.sparsity_output or stats_dir / "sparsity_kernel_body_cycles.png"
    sparsity_csv = args.sparsity_csv or stats_dir / "sparsity_sgemm_tcu_op.csv"
    sparsity_utilization_output = (
        args.sparsity_utilization_output
        or stats_dir / "sparsity_tcu_utilization.png"
    )
    sparsity_memory_output = args.sparsity_memory_output or stats_dir / "sparsity_memory_requests.png"
    sparsity_instructions_output = (
        args.sparsity_instructions_output
        or stats_dir / "sparsity_total_instructions.png"
    )

    dtype_rows = load_dtype_rows(stats_dir, args.include_failed)
    all_rows = load_all_rows(stats_dir, args.include_failed)
    write_csv(dtype_rows, csv_output)
    pdf_output = plot(dtype_rows, output)
    write_fp16_instructions_csv(dtype_rows, instructions_csv)
    instructions_pdf_output = plot_fp16_instructions(dtype_rows, instructions_output)
    write_utilization_csv(dtype_rows, utilization_csv)
    utilization_pdf_output = plot_utilization(dtype_rows, utilization_output)
    write_memory_csv(dtype_rows, memory_csv)
    memory_pdf_output = plot_memory_requests(dtype_rows, memory_output)
    write_sparsity_csv(all_rows, sparsity_csv)
    sparsity_pdf_output = plot_sparsity_metric(
        all_rows,
        sparsity_output,
        "kernel_body_cycles",
        "Kernel Body Cycles by Sparsity",
        "Kernel Body Cycles",
        "#558b2f",
    )
    sparsity_utilization_pdf_output = plot_sparsity_metric(
        all_rows,
        sparsity_utilization_output,
        "tcu_utilization",
        "TCU Utilization by Sparsity",
        "TCU Utilization (%)",
        "#7a5ea8",
    )
    sparsity_memory_pdf_output = plot_sparsity_memory(all_rows, sparsity_memory_output)
    sparsity_instructions_pdf_output = plot_sparsity_metric(
        all_rows,
        sparsity_instructions_output,
        "total_instructions",
        "Total Kernel Instructions by Sparsity",
        "Total Kernel Instructions",
        "#2f6f9f",
    )
    print(f"wrote {output}")
    print(f"wrote {pdf_output}")
    print(f"wrote {csv_output}")
    print(f"wrote {instructions_output}")
    print(f"wrote {instructions_pdf_output}")
    print(f"wrote {instructions_csv}")
    print(f"wrote {utilization_output}")
    print(f"wrote {utilization_pdf_output}")
    print(f"wrote {utilization_csv}")
    print(f"wrote {memory_output}")
    print(f"wrote {memory_pdf_output}")
    print(f"wrote {memory_csv}")
    print(f"wrote {sparsity_output}")
    print(f"wrote {sparsity_pdf_output}")
    print(f"wrote {sparsity_csv}")
    print(f"wrote {sparsity_utilization_output}")
    print(f"wrote {sparsity_utilization_pdf_output}")
    print(f"wrote {sparsity_memory_output}")
    print(f"wrote {sparsity_memory_pdf_output}")
    print(f"wrote {sparsity_instructions_output}")
    print(f"wrote {sparsity_instructions_pdf_output}")


if __name__ == "__main__":
    main()
