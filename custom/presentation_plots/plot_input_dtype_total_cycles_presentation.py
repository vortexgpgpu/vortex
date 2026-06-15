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
        description="Create the presentation version of the run_b total kernel cycles plot with baselines."
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
        default=SCRIPT_DIR / "input_dtype_total_kernel_cycles_presentation.png",
        type=Path,
        help="Output image path. A PDF copy is also written next to it.",
    )
    parser.add_argument(
        "--csv",
        default=SCRIPT_DIR / "input_dtype_kernel_body_cycles_presentation.csv",
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


def plot_presentation(rows, output, run_b):
    by_key = {(row["dtype"], row["variant"]): row for row in rows}
    centers, positions, bar_width = run_b.grouped_positions()
    metric = "kernel_body_cycles"
    values = [row[metric] for row in rows]
    speedup_baselines = {
        dtype: by_key[(dtype, run_b.VARIANT_ORDER[0])][metric]
        for dtype in run_b.DTYPE_ORDER
        if (dtype, run_b.VARIANT_ORDER[0]) in by_key
    }

    plt.rcParams.update(
        {
            "font.size": 26,
            "axes.labelsize": 31,
            "xtick.labelsize": 28,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(16.8, 9.4))
    max_value = max(values)
    for variant in run_b.VARIANT_ORDER:
        x_values = []
        y_values = []
        dtypes = []
        for dtype in run_b.DTYPE_ORDER:
            row = by_key.get((dtype, variant))
            if row is None:
                continue
            x_values.append(positions[(dtype, variant)])
            y_values.append(row[metric])
            dtypes.append(dtype)

        bars = ax.bar(
            x_values,
            y_values,
            width=bar_width * 0.88,
            label=run_b.VARIANT_LABELS[variant],
            color=run_b.VARIANT_COLORS[variant],
            edgecolor="#222222",
            linewidth=1.0,
        )

        for bar, dtype, value in zip(bars, dtypes, y_values):
            baseline = speedup_baselines.get(dtype)
            label = run_b.speedup_label(baseline, value) if baseline else ""
            if not label:
                continue
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=25,
                rotation=90,
                clip_on=False,
            )

    ax.set_xlabel("Input Data Type", labelpad=14)
    ax.set_ylabel("Total Kernel Cycles", labelpad=14)
    ax.set_xticks(centers)
    ax.set_xticklabels(run_b.DTYPE_ORDER)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_ylim(0, max_value * 1.55)
    ax.tick_params(axis="both", which="major", length=7, width=1.2)
    ax.grid(True, axis="y", linewidth=1.0, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="#333333",
        borderpad=0.7,
        handlelength=1.5,
        columnspacing=1.4,
    )

    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.2)

    fig.tight_layout()
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
