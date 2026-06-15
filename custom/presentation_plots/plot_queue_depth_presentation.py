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


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_CUSTOM_DIR = SCRIPT_DIR.parent
DEFAULT_STATS_DIR = REPO_CUSTOM_DIR / "run_all" / "run_q"
RUN_Q_PLOT_SCRIPT = DEFAULT_STATS_DIR / "plot_queue_depth.py"


def load_run_q_module():
    spec = importlib.util.spec_from_file_location("run_q_plot_queue_depth", RUN_Q_PLOT_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {RUN_Q_PLOT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the presentation version of the run_q queue-depth plot."
    )
    parser.add_argument(
        "stats_dir",
        nargs="?",
        default=DEFAULT_STATS_DIR,
        type=Path,
        help="Directory containing run_q_1.stat through run_q_16.stat.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=SCRIPT_DIR / "queue_depth_results_presentation.png",
        type=Path,
        help="Output image path. A PDF copy is also written next to it.",
    )
    parser.add_argument(
        "--csv",
        default=SCRIPT_DIR / "queue_depth_crossbar_stalls_presentation.csv",
        type=Path,
        help="Output CSV path.",
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
        help="Manifest CSV used to skip stale/nonzero runs. Defaults to run_q/run_all.csv or run_all/run_all.csv when present.",
    )
    return parser.parse_args()


def save_figure(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=260, bbox_inches="tight")

    pdf_output = output.with_suffix(".pdf")
    if pdf_output != output:
        fig.savefig(pdf_output, bbox_inches="tight")
    return pdf_output


def plot_presentation(rows, output):
    queue_depths = sorted({row["queue_depth"] for row in rows})
    config_labels = sorted({row["config_label"] for row in rows})
    values = {
        (row["queue_depth"], row["config_label"]): row["crossbar_stall_percent"]
        for row in rows
    }

    width = 0.76 / max(1, len(config_labels))
    x_positions = list(range(len(queue_depths)))

    plt.rcParams.update(
        {
            "font.size": 26,
            "axes.labelsize": 31,
            "xtick.labelsize": 28,
            "ytick.labelsize": 28,
            "legend.fontsize": 24,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig_width = max(13.5, 2.25 * len(queue_depths) + 1.75 * len(config_labels))
    fig, ax = plt.subplots(figsize=(fig_width, 9.2))
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
            linewidth=1.0,
        )
        for bar, height in zip(bars, heights):
            if height <= 0:
                continue
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=25,
                rotation=90,
                clip_on=False,
            )

    ax.set_xlabel("Crossbar Queue Depth", labelpad=14)
    ax.set_ylabel("Crossbar Stall Cycles (%)", labelpad=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(queue_depth) for queue_depth in queue_depths])
    ax.set_ylim(0, max(row["crossbar_stall_percent"] for row in rows) * 1.55)
    ax.tick_params(axis="both", which="major", length=7, width=1.2)
    ax.grid(True, axis="y", linewidth=1.0, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(len(config_labels), 4),
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
    stats_dir = args.stats_dir.resolve()
    status_csv = args.status_csv
    if status_csv is None:
        default_status_csv = stats_dir / "run_all.csv"
        parent_status_csv = stats_dir.parent / "run_all.csv"
        if default_status_csv.exists():
            status_csv = default_status_csv
        elif parent_status_csv.exists():
            status_csv = parent_status_csv

    run_q = load_run_q_module()
    rows = run_q.load_rows(
        stats_dir,
        args.start_run,
        args.end_run,
        args.include_failed,
        status_csv,
    )
    run_q.write_csv(rows, args.csv)
    pdf_output = plot_presentation(rows, args.output)
    print(f"wrote {args.output}")
    print(f"wrote {pdf_output}")
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
