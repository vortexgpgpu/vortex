#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

 
MATRIX_SHAPE = "128x128x512"
DTYPE = "fp8 -> fp32"

BENCHMARKS = [
    ("sgemm_tcu", "dense", 2.36),
    ("sgemm_tcu_sp", "0%, 50% sparse", 1.52),
    ("sgemm_tcu_op", "dense", 0.55),
    ("sgemm_tcu_op", "0%, 50% sparse", 0.504),
]

MEMORY_REQS = [
    ("run2_ip.stat", "sgemm_tcu", "baseline IP", 1693, 1953),
    ("run2.stat", "sgemm_tcu_op", "dense", 472, 463),
    ("run3.stat", "sgemm_tcu_op", "50%, 50% sparse", 523, 564),
]

DCACHE_REQS = [
    ("run2_ip.stat", "sgemm_tcu", "baseline IP", 6063, 1928),
    ("run2.stat", "sgemm_tcu_op", "dense", 777, 325),
    ("run3.stat", "sgemm_tcu_op", "50%, 50% sparse", 973, 424),
]

KERNEL_BODY_INSTRUCTIONS = [
    ("run2_ip.stat", "sgemm_tcu", "baseline IP", 593),
    ("run2.stat", "sgemm_tcu_op", "dense", 60),
    ("run3.stat", "sgemm_tcu_op", "50%, 50% sparse", 77),
]

GMEM_READS = [
    ("run8.stat", "sgemm_tcu_op", "dense", 128),
    ("run6.stat", "sgemm_tcu_op", "0%, 50% sparse", 106),
    ("run7.stat", "sgemm_tcu_op", "50%, 50% sparse", 82),
]

COLORS = {
    "dense": "#3b6fb6",
    "50%, 50% sparse": "#2f9e66",
    "0%, 50% sparse": "#d18f26",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot kernel runtime bars for SGEMM TCU variants."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="custom/stats",
        help=(
            "Output directory for --plot all, or output image path for a single plot. "
            "Use an empty string with --show to skip saving."
        ),
    )
    parser.add_argument(
        "--plot",
        choices=("all", "runtime", "memory", "dcache", "body-instr", "gmem"),
        default="all",
        help="Which bar plot to generate.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window.",
    )
    return parser.parse_args()


def make_labels():
    labels = []
    for name, variant, _ in BENCHMARKS:
        if variant == "dense" and name != "sgemm_tcu_op":
            labels.append(name)
        else:
            labels.append(f"{name}\n{variant}")
    return labels


def save_or_show(fig, output, show):
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        print(f"saved {path}")

    if show:
        plt.show()

    plt.close(fig)


def output_path(output, plot_name):
    if not output:
        return ""

    path = Path(output)
    if path.suffix:
        return path

    return path / f"{plot_name}.png"


def plot_runtime_bars(output, show):
    labels = make_labels()
    values = [ticks for _, _, ticks in BENCHMARKS]
    colors = [COLORS.get(variant, "#707070") for _, variant, _ in BENCHMARKS]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars = ax.bar(labels, values, color=colors, width=0.62)

    ax.set_title(f"SGEMM TCU Kernel Runtime ({MATRIX_SHAPE}, {DTYPE})")
    ax.set_ylabel("Ticks (M)")
    ax.set_ylim(0, max(values) * 1.22)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.025,
            f"{value:.3g}M",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    save_or_show(fig, output, show)


def plot_memory_bars(output, show):
    labels = [f"{bench}\n{variant}\n{stat}" for stat, bench, variant, _, _ in MEMORY_REQS]
    reads = [read_reqs for _, _, _, read_reqs, _ in MEMORY_REQS]
    writes = [write_reqs for _, _, _, _, write_reqs in MEMORY_REQS]
    totals = [read_reqs + write_reqs for read_reqs, write_reqs in zip(reads, writes)]
    x_positions = range(len(MEMORY_REQS))

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    read_bars = ax.bar(
        x_positions,
        reads,
        width=0.58,
        label="reads",
        color="#3b6fb6",
    )
    write_bars = ax.bar(
        x_positions,
        writes,
        bottom=reads,
        width=0.58,
        label="writes",
        color="#d18f26",
    )

    ax.set_title(f"PERF2 DRAM-Side Memory Requests ({MATRIX_SHAPE}, {DTYPE})")
    ax.set_ylabel("Requests")
    ax.set_xticks(list(x_positions), labels)
    ax.set_ylim(0, max(totals) * 1.18)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    for index, total in enumerate(totals):
        ax.text(
            index,
            total + max(totals) * 0.025,
            f"{total}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    for bars, values in ((read_bars, reads), (write_bars, writes)):
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{value}",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
            )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    save_or_show(fig, output, show)


def plot_dcache_bars(output, show):
    labels = [f"{bench}\n{variant}\n{stat}" for stat, bench, variant, _, _ in DCACHE_REQS]
    reqs = [total_reqs for _, _, _, total_reqs, _ in DCACHE_REQS]
    miss_r = [read_misses for _, _, _, _, read_misses in DCACHE_REQS]
    x_positions = list(range(len(DCACHE_REQS)))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    req_bars = ax.bar(
        [x - width / 2 for x in x_positions],
        reqs,
        width=width,
        label="dcache reqs",
        color="#3b6fb6",
    )
    miss_bars = ax.bar(
        [x + width / 2 for x in x_positions],
        miss_r,
        width=width,
        label="read misses",
        color="#c44e52",
    )

    ax.set_title(f"PERF2 D-Cache Requests and Read Misses ({MATRIX_SHAPE}, {DTYPE})")
    ax.set_ylabel("Count")
    ax.set_xticks(x_positions, labels)
    ax.set_ylim(0, max(reqs) * 1.18)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    for bars in (req_bars, miss_bars):
        for bar in bars:
            value = int(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(reqs) * 0.02,
                f"{value}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    save_or_show(fig, output, show)


def plot_gmem_bars(output, show):
    labels = [f"{bench}\n{variant}\n{stat}" for stat, bench, variant, _ in GMEM_READS]
    values = [gmem_reads for _, _, _, gmem_reads in GMEM_READS]
    colors = [COLORS.get(variant, "#707070") for _, _, variant, _ in GMEM_READS]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar(labels, values, color=colors, width=0.58)

    ax.set_title(f"PERF6 DXA Global Memory Reads ({MATRIX_SHAPE}, {DTYPE})")
    ax.set_ylabel("gmem_reads")
    ax.set_ylim(0, max(values) * 1.22)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.025,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    save_or_show(fig, output, show)


def plot_body_instr_bars(output, show):
    labels = [
        f"{bench}\n{variant}\n{stat}"
        for stat, bench, variant, _ in KERNEL_BODY_INSTRUCTIONS
    ]
    values = [instrs for _, _, _, instrs in KERNEL_BODY_INSTRUCTIONS]
    colors = [
        COLORS.get(variant, "#707070")
        for _, _, variant, _ in KERNEL_BODY_INSTRUCTIONS
    ]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar(labels, values, color=colors, width=0.58)

    ax.set_title(f"Kernel Body Instructions ({MATRIX_SHAPE}, {DTYPE})")
    ax.set_ylabel("Instructions")
    ax.set_ylim(0, max(values) * 1.22)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.025,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    save_or_show(fig, output, show)


def main():
    args = parse_args()
    if args.plot in ("all", "runtime"):
        plot_runtime_bars(output_path(args.output, "sgemm_tcu_runtime_bars"), args.show)
    if args.plot in ("all", "memory"):
        plot_memory_bars(output_path(args.output, "sgemm_tcu_memory_reqs"), args.show)
    if args.plot in ("all", "dcache"):
        plot_dcache_bars(output_path(args.output, "sgemm_tcu_dcache_reqs"), args.show)
    if args.plot in ("all", "body-instr"):
        plot_body_instr_bars(
            output_path(args.output, "sgemm_tcu_kernel_body_instructions"),
            args.show,
        )
    if args.plot in ("all", "gmem"):
        plot_gmem_bars(output_path(args.output, "sgemm_tcu_gmem_reads"), args.show)


if __name__ == "__main__":
    main()
