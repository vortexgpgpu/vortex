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
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_CUSTOM_DIR = SCRIPT_DIR.parent
DEFAULT_RUN_A_DIR = REPO_CUSTOM_DIR / "run_all" / "run_a"
DEFAULT_RUN_S_DIR = REPO_CUSTOM_DIR / "run_all" / "run_s"
RUN_A_PLOT_SCRIPT = DEFAULT_RUN_A_DIR / "plot_sparsity_sweep.py"


def load_run_a_module():
    spec = importlib.util.spec_from_file_location("run_a_plot_sparsity_sweep", RUN_A_PLOT_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {RUN_A_PLOT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the presentation version of the run_a/run_s sparsity sweep comparison."
    )
    parser.add_argument(
        "--run-a-dir",
        default=DEFAULT_RUN_A_DIR,
        type=Path,
        help="Directory containing run_a_*.stat files.",
    )
    parser.add_argument(
        "--run-s-dir",
        default=DEFAULT_RUN_S_DIR,
        type=Path,
        help="Directory containing run_s_*.stat files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=SCRIPT_DIR / "sparsity_sweep_comparison_presentation.png",
        type=Path,
        help="Output image path. A PDF copy is also written next to it.",
    )
    parser.add_argument(
        "--csv",
        default=SCRIPT_DIR / "sparsity_sweep_comparison_presentation.csv",
        type=Path,
        help="Output CSV path.",
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
        "--include-failed",
        action="store_true",
        help="Include stat files whose parsed status is not PASS.",
    )
    return parser.parse_args()


def save_figure(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=260, bbox_inches="tight")

    pdf_output = output.with_suffix(".pdf")
    if pdf_output != output:
        fig.savefig(pdf_output, bbox_inches="tight")
    return pdf_output


def plot_presentation(run_s_rows, run_a_rows, output, metric, b_sparsities, run_a):
    series = run_a.build_series(run_s_rows, run_a_rows, b_sparsities)
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
        "A uncompressed": {"linestyle": ":", "marker": "s", "alpha": 0.92},
    }

    plt.rcParams.update(
        {
            "font.size": 26,
            "axes.labelsize": 31,
            "xtick.labelsize": 26,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(17.2, 9.4))
    for index, (group_label, b_label, points) in enumerate(series):
        x_values = [point["a_label"] for point in points]
        y_values = [point[metric_key] for point in points]
        color = colors.get(b_label, fallback_colors[index % len(fallback_colors)])
        style = styles[group_label]
        ax.plot(
            x_values,
            y_values,
            marker=style["marker"],
            linewidth=3.0,
            markersize=8.0,
            linestyle=style["linestyle"],
            color=color,
            alpha=style["alpha"],
        )

    max_value = max(
        point[metric_key]
        for _group_label, _b_label, points in series
        for point in points
    )
    ax.set_xlabel("Matrix A Sparsity (%)", labelpad=14)
    ax.set_ylabel(metric_label, labelpad=14)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max_value * 1.16)
    ax.set_xticks(range(0, 101, 10))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.tick_params(axis="both", which="major", length=7, width=1.2)
    ax.grid(True, which="major", linewidth=1.0, alpha=0.35)
    ax.set_axisbelow(True)

    b_labels = sorted({b_label for _group_label, b_label, _points in series})
    color_handles = [
        Line2D(
            [0],
            [0],
            color=colors.get(b_label, fallback_colors[index % len(fallback_colors)]),
            marker="o",
            linewidth=3.0,
            markersize=8.0,
            label=f"B {b_label}%",
        )
        for index, b_label in enumerate(b_labels)
    ]
    storage_handles = [
        Line2D([0], [0], color="#222222", linewidth=3.0, linestyle="-", label="A compressed"),
        Line2D([0], [0], color="#222222", linewidth=3.0, linestyle=":", label="A uncompressed"),
    ]
    ax.legend(
        handles=color_handles + storage_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=True,
        fancybox=False,
        edgecolor="#333333",
        borderpad=0.7,
        handlelength=2.4,
        columnspacing=1.25,
    )

    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.2)

    fig.subplots_adjust(left=0.11, right=0.985, top=0.985, bottom=0.34)
    pdf_output = save_figure(fig, output)
    plt.close(fig)
    return pdf_output


def main():
    args = parse_args()
    run_a = load_run_a_module()
    run_s_rows = run_a.load_rows(
        args.run_s_dir.resolve(),
        "run_s_*.stat",
        "run_s",
        args.include_failed,
    )
    run_a_rows = run_a.load_rows(
        args.run_a_dir.resolve(),
        "run_a_*.stat",
        "run_a",
        args.include_failed,
    )
    run_a.write_csv(run_s_rows + run_a_rows, args.csv)
    pdf_output = plot_presentation(
        run_s_rows,
        run_a_rows,
        args.output,
        args.metric,
        args.b_sparsities,
        run_a,
    )
    print(f"wrote {args.output}")
    print(f"wrote {pdf_output}")
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
