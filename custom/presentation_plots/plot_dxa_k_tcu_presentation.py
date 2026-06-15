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
DEFAULT_DXA_STATS_DIR = REPO_CUSTOM_DIR / "run_all" / "run_d"
DEFAULT_K_STATS_DIR = REPO_CUSTOM_DIR / "run_all" / "run_k"
DXA_PLOT_SCRIPT = DEFAULT_DXA_STATS_DIR / "plot_dxa_transactions.py"
K_PLOT_SCRIPT = DEFAULT_K_STATS_DIR / "plot_k_tcu_utilization.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the combined presentation plot for DMA transfers and K-dimension TCU utilization."
    )
    parser.add_argument(
        "--dxa-stats-dir",
        default=DEFAULT_DXA_STATS_DIR,
        type=Path,
        help="Directory containing run_d_*.stat files.",
    )
    parser.add_argument(
        "--k-stats-dir",
        default=DEFAULT_K_STATS_DIR,
        type=Path,
        help="Directory containing run_k_1.stat through run_k_8.stat.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=SCRIPT_DIR / "dxa_k_tcu_presentation.png",
        type=Path,
        help="Output image path. A PDF copy is also written next to it.",
    )
    parser.add_argument(
        "--dxa-csv",
        default=SCRIPT_DIR / "dxa_transactions_presentation.csv",
        type=Path,
        help="Output CSV path for DMA data.",
    )
    parser.add_argument(
        "--k-csv",
        default=SCRIPT_DIR / "k_tcu_utilization_presentation.csv",
        type=Path,
        help="Output CSV path for K-utilization data.",
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


def style_axis(ax):
    ax.tick_params(axis="both", which="major", length=7, width=1.2)
    ax.grid(True, axis="y", linewidth=1.0, alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.2)


def annotate_values(ax, bars, labels, fontsize=22):
    for bar, label in zip(bars, labels):
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=90,
            clip_on=False,
        )


def plot_dxa(ax, rows):
    colors = {
        "Dense": "#4f5d75",
        "A compressed, B compressed": "#2f6f9f",
        "A uncompressed, B compressed": "#c7533b",
    }
    x_values = list(range(len(rows)))
    width = 0.34
    max_value = max(
        max(row["dxa_gmem_reads"], row["theoretical_min_gmem_reads"]) for row in rows
    )

    actual_bars = []
    theory_bars = []
    for x_value, row in zip(x_values, rows):
        actual_bars.append(
            ax.bar(
                x_value - width / 2,
                row["dxa_gmem_reads"],
                width,
                color=colors.get(row["mode_label"], "#7a5ea8"),
                edgecolor="#222222",
                linewidth=1.0,
            )[0]
        )
        theory_bars.append(
            ax.bar(
                x_value + width / 2,
                row["theoretical_min_gmem_reads"],
                width,
                facecolor="white",
                edgecolor="#222222",
                hatch="///",
                linewidth=1.1,
            )[0]
        )

    annotate_values(ax, actual_bars, [f"{row['dxa_gmem_reads']:.0f}" for row in rows])
    annotate_values(
        ax,
        theory_bars,
        [f"{row['theoretical_min_gmem_reads']:.0f}" for row in rows],
    )

    ax.set_ylabel("DMA Transactions", labelpad=14)
    ax.set_xticks(x_values)
    ax.set_xticklabels([f"{row['a_label']}/{row['b_label']}" for row in rows])
    ax.set_xlabel("A/B Sparsity (%)", labelpad=14)
    ax.set_ylim(0, max_value * 1.50)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    style_axis(ax)
    return colors


def plot_k_tcu(ax, rows):
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
    x_positions = list(range(len(k_values)))
    colors = ["#7a5ea8", "#b8792d", "#558b2f", "#2f6f9f"]
    group_width = min(0.76, 0.28 * len(step_sizes) + 0.20)
    bar_width = group_width / max(1, len(step_sizes))
    step_colors = {}

    for step_index, (block_m, block_n) in enumerate(step_sizes):
        color = colors[step_index % len(colors)]
        step_colors[(block_m, block_n)] = color
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
            color=color,
            edgecolor="#222222",
            linewidth=1.0,
        )
        annotate_values(ax, bars, [f"{value:.1f}%" for value in values])

    ax.set_xlabel("K Dimension", labelpad=14)
    ax.set_ylabel("TCU Utilization (%)", labelpad=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(k_value) for k_value in k_values])
    ax.set_ylim(0, min(112, max(y_values) * 1.28))
    style_axis(ax)
    return step_colors


def find_status_csv(stats_dir):
    default_status_csv = stats_dir / "run_all.csv"
    parent_status_csv = stats_dir.parent / "run_all.csv"
    if default_status_csv.exists():
        return default_status_csv
    if parent_status_csv.exists():
        return parent_status_csv
    return None


def plot_presentation(dxa_rows, k_rows, output):
    plt.rcParams.update(
        {
            "font.size": 25,
            "axes.labelsize": 30,
            "xtick.labelsize": 26,
            "ytick.labelsize": 23,
            "legend.fontsize": 24,
            "hatch.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, (dxa_ax, k_ax) = plt.subplots(
        2,
        1,
        figsize=(16.8, 16.2),
        gridspec_kw={"height_ratios": [1, 1]},
    )

    dxa_colors = plot_dxa(dxa_ax, dxa_rows)
    step_colors = plot_k_tcu(k_ax, k_rows)

    dxa_handles = [
        Patch(facecolor=dxa_colors["Dense"], edgecolor="#222222", label="DMA Dense"),
        Patch(
            facecolor=dxa_colors["A compressed, B compressed"],
            edgecolor="#222222",
            label="DMA A/B compressed",
        ),
        Patch(
            facecolor=dxa_colors["A uncompressed, B compressed"],
            edgecolor="#222222",
            label="DMA B compressed",
        ),
        Patch(facecolor="white", edgecolor="#222222", hatch="///", label="Theoretical min"),
    ]
    step_handles = [
        Patch(
            facecolor=color,
            edgecolor="#222222",
            label=f"Step {block_m}x{block_n}",
        )
        for (block_m, block_n), color in step_colors.items()
    ]

    dxa_ax.legend(
        handles=dxa_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=4,
        frameon=True,
        fancybox=False,
        edgecolor="#333333",
        borderpad=0.7,
        handlelength=1.4,
        columnspacing=1.25,
    )
    k_ax.legend(
        handles=step_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        fontsize=30,
        frameon=True,
        fancybox=False,
        edgecolor="#333333",
        borderpad=0.9,
        handlelength=1.7,
        handletextpad=0.7,
        columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.985, bottom=0.22, hspace=0.66)
    pdf_output = save_figure(fig, output)
    plt.close(fig)
    return pdf_output


def main():
    args = parse_args()
    dxa_module = load_module("run_d_plot_dxa_transactions", DXA_PLOT_SCRIPT)
    k_module = load_module("run_k_plot_tcu_utilization", K_PLOT_SCRIPT)

    dxa_rows = dxa_module.load_rows(args.dxa_stats_dir.resolve(), args.include_failed)
    k_stats_dir = args.k_stats_dir.resolve()
    k_rows = k_module.load_rows(
        k_stats_dir,
        1,
        8,
        args.include_failed,
        find_status_csv(k_stats_dir),
    )

    dxa_module.write_csv(dxa_rows, args.dxa_csv)
    k_module.write_csv(k_rows, args.k_csv)
    pdf_output = plot_presentation(dxa_rows, k_rows, args.output)
    print(f"wrote {args.output}")
    print(f"wrote {pdf_output}")
    print(f"wrote {args.dxa_csv}")
    print(f"wrote {args.k_csv}")


if __name__ == "__main__":
    main()
