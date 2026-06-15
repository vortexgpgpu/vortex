#!/usr/bin/env python3
import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_CUSTOM_DIR = SCRIPT_DIR.parent
DEFAULT_STATS_DIR = REPO_CUSTOM_DIR / "run_all" / "run_b"
RUN_B_PLOT_SCRIPT = DEFAULT_STATS_DIR / "plot_input_dtype_cycles.py"


def load_run_b_module():
    spec = importlib.util.spec_from_file_location("run_b_plot_input_dtype_cycles", RUN_B_PLOT_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {RUN_B_PLOT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the combined presentation plot for run_b instructions and TCU multiplier utilization."
    )
    parser.add_argument(
        "stats_dir",
        nargs="?",
        default=DEFAULT_STATS_DIR,
        type=Path,
        help="Directory containing run_b_1.stat through run_b_18.stat.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=SCRIPT_DIR / "input_dtype_instructions_tcu_presentation.png",
        type=Path,
        help="Output image path. A PDF copy is also written next to it.",
    )
    parser.add_argument(
        "--csv",
        default=SCRIPT_DIR / "input_dtype_instructions_tcu_presentation.csv",
        type=Path,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        default=True,
        help="Include failed stats when metrics are present. Default: enabled.",
    )
    return parser.parse_args()


def save_figure(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=260, bbox_inches="tight")

    pdf_output = output.with_suffix(".pdf")
    if pdf_output != output:
        fig.savefig(pdf_output, bbox_inches="tight")
    return pdf_output


def style_axis(ax):
    ax.tick_params(axis="both", which="major", length=7, width=1.2)
    ax.grid(True, axis="y", linewidth=1.0, alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.2)


def annotate_speedups(ax, bars, dtypes, values, baselines, run_b, fontsize=23):
    for bar, dtype, value in zip(bars, dtypes, values):
        baseline = baselines.get(dtype)
        label = run_b.speedup_label(baseline, value) if baseline else ""
        if not label:
            continue
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=90,
            clip_on=False,
        )


def plot_instructions(ax, rows, run_b):
    by_key = {(row["dtype"], row["variant"]): row for row in rows}
    centers, positions, bar_width = run_b.grouped_positions()
    values = [row["total_instructions"] for row in rows]
    baselines = {
        dtype: by_key[(dtype, run_b.VARIANT_ORDER[0])]["total_instructions"]
        for dtype in run_b.DTYPE_ORDER
        if (dtype, run_b.VARIANT_ORDER[0]) in by_key
    }

    for variant in run_b.VARIANT_ORDER:
        x_values = []
        y_values = []
        dtypes = []
        for dtype in run_b.DTYPE_ORDER:
            row = by_key.get((dtype, variant))
            if row is None:
                continue
            x_values.append(positions[(dtype, variant)])
            y_values.append(row["total_instructions"])
            dtypes.append(dtype)

        bars = ax.bar(
            x_values,
            y_values,
            width=bar_width * 0.88,
            color=run_b.VARIANT_COLORS[variant],
            edgecolor="#222222",
            linewidth=1.0,
        )
        annotate_speedups(ax, bars, dtypes, y_values, baselines, run_b)

    ax.set_ylabel("Total Kernel Instructions", labelpad=14)
    ax.set_xticks(centers)
    ax.set_xticklabels([])
    ax.set_ylim(0, max(values) * 1.42)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    style_axis(ax)


def plot_tcu_utilization(ax, rows, run_b):
    by_key = {(row["dtype"], row["variant"]): row for row in rows}
    centers, positions, bar_width = run_b.grouped_positions()
    values = [row["tcu_utilization"] for row in rows]

    for variant in run_b.VARIANT_ORDER:
        x_values = []
        y_values = []
        for dtype in run_b.DTYPE_ORDER:
            row = by_key.get((dtype, variant))
            if row is None:
                continue
            x_values.append(positions[(dtype, variant)])
            y_values.append(row["tcu_utilization"])

        bars = ax.bar(
            x_values,
            y_values,
            width=bar_width * 0.88,
            color=run_b.VARIANT_COLORS[variant],
            edgecolor="#222222",
            linewidth=1.0,
        )
        for bar, value in zip(bars, y_values):
            ax.annotate(
                f"{value:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=22,
                rotation=90,
                clip_on=False,
            )

    ax.set_xlabel("Input Data Type", labelpad=14)
    ax.set_ylabel("TCU Multiplier Utilization (%)", labelpad=14)
    ax.set_xticks(centers)
    ax.set_xticklabels(run_b.DTYPE_ORDER)
    ax.set_ylim(0, min(112, max(values) * 1.18))
    style_axis(ax)


def plot_presentation(rows, output, run_b):
    plt.rcParams.update(
        {
            "font.size": 25,
            "axes.labelsize": 30,
            "xtick.labelsize": 27,
            "ytick.labelsize": 23,
            "legend.fontsize": 23,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, (instructions_ax, tcu_ax) = plt.subplots(
        2,
        1,
        figsize=(16.8, 15.8),
        gridspec_kw={"height_ratios": [1, 1]},
    )

    plot_instructions(instructions_ax, rows, run_b)
    plot_tcu_utilization(tcu_ax, rows, run_b)

    variant_handles = [
        Patch(
            facecolor=run_b.VARIANT_COLORS[variant],
            edgecolor="#222222",
            label=run_b.VARIANT_LABELS[variant],
        )
        for variant in run_b.VARIANT_ORDER
    ]

    fig.legend(
        handles=variant_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="#333333",
        borderpad=0.7,
        handlelength=1.5,
        columnspacing=1.35,
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.985, bottom=0.225, hspace=0.24)
    pdf_output = save_figure(fig, output)
    plt.close(fig)
    return pdf_output


def main():
    args = parse_args()
    run_b = load_run_b_module()
    rows = run_b.load_rows(args.stats_dir.resolve(), args.include_failed)
    run_b.write_csv(rows, args.csv)
    pdf_output = plot_presentation(rows, args.output, run_b)
    print(f"wrote {args.output}")
    print(f"wrote {pdf_output}")
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
