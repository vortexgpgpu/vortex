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
        default="custom/stats/sgemm_tcu_bars.png",
        help="Output image path. Use an empty string with --show to skip saving.",
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


def plot_bars(output, show):
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

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        print(f"saved {path}")

    if show:
        plt.show()

    plt.close(fig)


def main():
    args = parse_args()
    plot_bars(args.output, args.show)


if __name__ == "__main__":
    main()
