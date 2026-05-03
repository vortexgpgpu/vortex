#!/usr/bin/env python3
import argparse
import os
import re

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_STATS = [
    "run1_ip.stat",
    "run1_ip_sp.stat",
    "run1.stat",
    "run2_ip.stat",
    "run2_ip_sp.stat",
    "run2.stat",
]

KERNEL_ORDER = ["sgemm_tcu", "sgemm_tcu_sp", "sgemm_tcu_op"]
DTYPE_ORDER = ["fp8", "fp16"]
DTYPE_LABELS = {
    "fp8": "Input type fp8",
    "fp16": "input type fp16",
}
KERNEL_LABELS = {
    "sgemm_tcu": "SGEMM",
    "sgemm_tcu_sp": "SGEMM_SP",
    "sgemm_tcu_op": "SGEMM_OP",
}

TESTBENCH_RE = re.compile(r"^Testbench:\s*(\S+)")
DTYPE_RE = re.compile(r"\bIt=([A-Za-z0-9_]+)")
INT_METRIC_RE = re.compile(r"^(Kernel body cycles|Kernel body instructions):\s*(\d+)")
UTIL_RE = re.compile(r"^TCU utilization:\s*([0-9.]+)%")
MEM_RE = re.compile(r"^PERF2:\s*memory:\s*reqs=(\d+)\s*\(r=(\d+),\s*w=(\d+)\)")


def parse_stat(path):
    row = {
        "path": path,
        "kernel": None,
        "dtype": None,
        "kernel_body_cycles": None,
        "kernel_body_instructions": None,
        "tcu_utilization": None,
        "memory_reqs": None,
        "memory_reads": None,
        "memory_writes": None,
    }

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            m_testbench = TESTBENCH_RE.match(line)
            if m_testbench:
                row["kernel"] = m_testbench.group(1)
                continue

            m_dtype = DTYPE_RE.search(line)
            if m_dtype:
                row["dtype"] = m_dtype.group(1)

            m_int = INT_METRIC_RE.match(line)
            if m_int:
                key = m_int.group(1).lower().replace(" ", "_")
                row[key] = int(m_int.group(2))
                continue

            m_util = UTIL_RE.match(line)
            if m_util:
                row["tcu_utilization"] = float(m_util.group(1))
                continue

            m_mem = MEM_RE.match(line)
            if m_mem:
                row["memory_reqs"] = int(m_mem.group(1))
                row["memory_reads"] = int(m_mem.group(2))
                row["memory_writes"] = int(m_mem.group(3))
                continue

    missing = [key for key, value in row.items()
               if key not in ("path", "memory_reads", "memory_writes") and value is None]
    if missing:
        raise ValueError(f"{path}: missing required fields: {', '.join(missing)}")
    return row


def load_rows(stats_dir, names):
    rows = []
    for name in names:
        path = name if os.path.isabs(name) else os.path.join(stats_dir, name)
        rows.append(parse_stat(path))
    return rows


def metric_lookup(rows):
    return {(row["kernel"], row["dtype"]): row for row in rows}


def ordered_rows(rows):
    by_key = metric_lookup(rows)
    ordered = []
    for kernel in KERNEL_ORDER:
        for dtype in DTYPE_ORDER:
            row = by_key.get((kernel, dtype))
            if row is not None:
                ordered.append(row)
    return ordered


def plot_execution(rows, output):
    by_key = metric_lookup(rows)
    metrics = [
        ("kernel_body_cycles", "Total execution cycles"),
        ("kernel_body_instructions", "Total execution instructions"),
        ("tcu_utilization", "Tensor core utilization"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
    x = np.arange(len(KERNEL_ORDER))
    width = 0.36
    colors = {"fp8": "#4C78A8", "fp16": "#F58518"}

    for ax, (metric, title) in zip(axes, metrics):
        for idx, dtype in enumerate(DTYPE_ORDER):
            values = [
                by_key.get((kernel, dtype), {}).get(metric, np.nan)
                for kernel in KERNEL_ORDER
            ]
            offset = (idx - 0.5) * width
            ax.bar(x + offset, values, width, label=DTYPE_LABELS[dtype], color=colors[dtype])

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([KERNEL_LABELS[k] for k in KERNEL_ORDER], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=len(DTYPE_ORDER),
               bbox_to_anchor=(0.5, -0.08))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_memory(rows, output):
    by_key = metric_lookup(rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    x = np.arange(len(KERNEL_ORDER))
    width = 0.36
    colors = {
        "fp8_read": "#4C78A8",
        "fp8_write": "#9ECae9",
        "fp16_read": "#F58518",
        "fp16_write": "#FFBF79",
    }

    for idx, dtype in enumerate(DTYPE_ORDER):
        reads = [
            by_key.get((kernel, dtype), {}).get("memory_reads", np.nan)
            for kernel in KERNEL_ORDER
        ]
        writes = [
            by_key.get((kernel, dtype), {}).get("memory_writes", np.nan)
            for kernel in KERNEL_ORDER
        ]
        offset = (idx - 0.5) * width
        ax.bar(x + offset, reads, width, label=f"{DTYPE_LABELS[dtype]} reads",
               color=colors[f"{dtype}_read"])
        ax.bar(x + offset, writes, width, bottom=reads, label=f"{DTYPE_LABELS[dtype]} writes",
               color=colors[f"{dtype}_write"])

    ax.set_title("Memory Requests")
    ax.set_ylabel("Memory requests")
    ax.set_xticks(x)
    ax.set_xticklabels([KERNEL_LABELS[k] for k in KERNEL_ORDER])
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=1, fontsize=8)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_summary(rows):
    print("kernel,dtype,body_cycles,body_instr,tcu_util_pct,memory_reqs")
    for row in ordered_rows(rows):
        print(
            f"{row['kernel']},{row['dtype']},"
            f"{row['kernel_body_cycles']},{row['kernel_body_instructions']},"
            f"{row['tcu_utilization']},{row['memory_reqs']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Create compact comparison bar charts for sgemm_tcu stat files."
    )
    parser.add_argument(
        "--stats-dir",
        default=os.path.join(os.path.dirname(__file__), "stats"),
        help="Directory containing .stat files. Default: custom/stats.",
    )
    parser.add_argument(
        "--stats",
        nargs="+",
        default=DEFAULT_STATS,
        help="Stat files to parse. Default: run1/run2 with _ip and _ip_sp.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "stats"),
        help="Directory for output PNGs. Default: custom/stats.",
    )
    parser.add_argument(
        "--prefix",
        default="oscar",
        help="Output filename prefix. Default: oscar.",
    )
    args = parser.parse_args()

    rows = load_rows(args.stats_dir, args.stats)
    os.makedirs(args.output_dir, exist_ok=True)

    execution_path = os.path.join(args.output_dir, f"{args.prefix}_execution.png")
    memory_path = os.path.join(args.output_dir, f"{args.prefix}_memory.png")

    plot_execution(rows, execution_path)
    plot_memory(rows, memory_path)
    print_summary(rows)
    print(f"Saved {execution_path}")
    print(f"Saved {memory_path}")


if __name__ == "__main__":
    main()
