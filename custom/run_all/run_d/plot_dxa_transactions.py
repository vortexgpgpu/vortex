#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


RUN_RE = re.compile(r"(?:run_d_|run)(?P<run>\d+)")
META_RE = re.compile(
    r"Sparsity Mode:(?P<mode>\d+)\s*\|\s*"
    r"MxNxK=(?P<m>\d+)x(?P<n>\d+)x(?P<k>\d+)\s*\|\s*"
    r"A sparsity=(?P<a>[0-9.]+)%\s*\|\s*"
    r"B sparsity=(?P<b>[0-9.]+)%\s*\|\s*"
    r"It=(?P<dtype>\S+)"
)
STATUS_RE = re.compile(r"^(PASS|FAIL)$")
DXA_RE = re.compile(
    r"^PERF6: dxa: transfers=(?P<transfers>\d+), "
    r"gmem_reads=(?P<gmem_reads>\d+), "
    r"gmem_dedup=(?P<gmem_dedup>\d+) .*"
    r"smem_writes=(?P<smem_writes>\d+), "
    r"avg_gmem_lat=(?P<avg_gmem_lat>[0-9.]+)"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot DXA GMEM transactions against theoretical minima for run_d stats."
    )
    parser.add_argument(
        "stats_dir",
        nargs="?",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Directory containing run_d_*.stat files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=Path,
        help="Output image path. Defaults to <stats_dir>/dxa_transactions.png. A PDF copy is also written.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        type=Path,
        help="Output CSV path. Defaults to <stats_dir>/dxa_transactions.csv.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include stat files whose parsed status is not PASS.",
    )
    return parser.parse_args()


def rounded_percent(value):
    return int(round(value))


def input_ratio(dtype):
    ratios = {
        "fp32": 1,
        "fp16": 2,
        "bf16": 2,
        "fp8": 4,
    }
    if dtype not in ratios:
        raise ValueError(f"unsupported input dtype for theoretical DXA model: {dtype}")
    return ratios[dtype]


def tile_sizes(total, tile_size):
    return [min(tile_size, total - offset) for offset in range(0, total, tile_size)]


def tile_transactions(rows, cols, ratio):
    return math.ceil((rows * cols) / float(16 * ratio))


def sparse_transactions(dense_transactions, sparsity_percent, ratio):
    data_transactions = dense_transactions * (1.0 - sparsity_percent / 100.0)
    bitmap_transactions = dense_transactions / (32.0 / ratio)
    return data_transactions + bitmap_transactions


def theoretical_min_transactions(row):
    ratio = input_ratio(row["dtype"])
    m_tiles = tile_sizes(row["m"], 32)
    n_tiles = tile_sizes(row["n"], 32)
    k_tiles = tile_sizes(row["k"], 32 * ratio)

    a_dense_transactions = sum(
        tile_transactions(m_tile, k_tile, ratio) * len(n_tiles)
        for m_tile in m_tiles
        for k_tile in k_tiles
    )
    b_dense_transactions = sum(
        tile_transactions(k_tile, n_tile, ratio) * len(m_tiles)
        for n_tile in n_tiles
        for k_tile in k_tiles
    )

    if row["sparsity_mode"] == 0:
        return a_dense_transactions + b_dense_transactions

    a_transactions = a_dense_transactions
    if row["sparsity_mode"] == 2:
        a_transactions = sparse_transactions(
            a_dense_transactions, row["a_sparsity"], ratio
        )

    b_transactions = sparse_transactions(b_dense_transactions, row["b_sparsity"], ratio)
    return a_transactions + b_transactions


def case_label(row):
    return f"A: {row['a_label']}%, B: {row['b_label']}%"


def mode_label(row):
    if row["sparsity_mode"] == 0:
        return "Dense"
    if row["sparsity_mode"] == 1:
        return "A uncompressed, B compressed"
    if row["sparsity_mode"] == 2:
        return "A compressed, B compressed"
    return f"s{row['sparsity_mode']}"


def parse_stat(path):
    run_match = RUN_RE.search(path.name)
    if not run_match:
        raise ValueError(f"{path}: cannot parse run number")

    row = {
        "file": path.name,
        "run": int(run_match.group("run")),
        "status": "",
        "sparsity_mode": None,
        "m": None,
        "n": None,
        "k": None,
        "dtype": "",
        "a_sparsity": None,
        "b_sparsity": None,
        "a_label": None,
        "b_label": None,
        "dxa_transfers": None,
        "dxa_gmem_reads": None,
        "dxa_gmem_dedup": None,
        "dxa_smem_writes": None,
        "dxa_avg_gmem_lat": None,
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
                row["sparsity_mode"] = int(match.group("mode"))
                row["m"] = int(match.group("m"))
                row["n"] = int(match.group("n"))
                row["k"] = int(match.group("k"))
                row["a_sparsity"] = float(match.group("a"))
                row["b_sparsity"] = float(match.group("b"))
                row["a_label"] = rounded_percent(row["a_sparsity"])
                row["b_label"] = rounded_percent(row["b_sparsity"])
                row["dtype"] = match.group("dtype")
                continue

            match = DXA_RE.match(line)
            if match:
                row["dxa_transfers"] = int(match.group("transfers"))
                row["dxa_gmem_reads"] = int(match.group("gmem_reads"))
                row["dxa_gmem_dedup"] = int(match.group("gmem_dedup"))
                row["dxa_smem_writes"] = int(match.group("smem_writes"))
                row["dxa_avg_gmem_lat"] = float(match.group("avg_gmem_lat"))

    missing = [
        key
        for key in (
            "status",
            "sparsity_mode",
            "m",
            "n",
            "k",
            "dtype",
            "a_sparsity",
            "b_sparsity",
            "dxa_transfers",
            "dxa_gmem_reads",
        )
        if row[key] in ("", None)
    ]
    if missing:
        raise ValueError(f"{path}: missing required fields: {', '.join(missing)}")

    row["mode_label"] = mode_label(row)
    row["case_label"] = case_label(row)
    row["theoretical_min_gmem_reads"] = theoretical_min_transactions(row)
    row["actual_over_theory"] = (
        row["dxa_gmem_reads"] / row["theoretical_min_gmem_reads"]
        if row["theoretical_min_gmem_reads"]
        else math.nan
    )
    return row


def load_rows(stats_dir, include_failed):
    rows = []
    skipped = []
    for path in sorted(stats_dir.glob("run_d_*.stat")):
        row = parse_stat(path)
        if include_failed or row["status"] == "PASS":
            rows.append(row)
        else:
            skipped.append(path.name)

    if skipped:
        print(f"warning: skipped non-passing stats: {', '.join(skipped)}")
    if not rows:
        raise FileNotFoundError(f"no usable run_d_*.stat files found in {stats_dir}")

    return sorted(rows, key=lambda row: row["run"])


def write_csv(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "run",
        "status",
        "sparsity_mode",
        "mode_label",
        "m",
        "n",
        "k",
        "dtype",
        "a_sparsity",
        "b_sparsity",
        "a_label",
        "b_label",
        "dxa_transfers",
        "dxa_gmem_reads",
        "dxa_gmem_dedup",
        "dxa_smem_writes",
        "dxa_avg_gmem_lat",
        "theoretical_min_gmem_reads",
        "actual_over_theory",
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
    colors = {
        "Dense": "#4f5d75",
        "A compressed, B compressed": "#2f6f9f",
        "A uncompressed, B compressed": "#c7533b",
    }

    fig, ax = plt.subplots(figsize=(10.6, 5.9))
    x_values = list(range(len(rows)))
    width = 0.36

    for x, row in zip(x_values, rows):
        ax.bar(
            x - width / 2,
            row["dxa_gmem_reads"],
            width,
            color=colors.get(row["mode_label"], "#7a5ea8"),
            label=row["mode_label"],
        )
        ax.bar(
            x + width / 2,
            row["theoretical_min_gmem_reads"],
            width,
            facecolor="white",
            edgecolor="#222222",
            hatch="///",
            linewidth=1.1,
            label="Theoretical minimum",
        )
        ax.text(
            x + width / 2,
            row["theoretical_min_gmem_reads"] + 35,
            f"{row['theoretical_min_gmem_reads']:.0f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    max_value = max(
        max(row["dxa_gmem_reads"], row["theoretical_min_gmem_reads"]) for row in rows
    )
    shape = f"{rows[0]['m']}x{rows[0]['n']}x{rows[0]['k']}"
    dtype = rows[0]["dtype"]

    ax.set_title(f"DXA GMEM Transactions - Matrix operation: {shape}, Input type: {dtype}")
    ax.set_ylabel("DXA GMEM Transactions")
    ax.set_xticks(x_values)
    ax.set_xticklabels([row["case_label"] for row in rows], fontsize=9)
    ax.set_ylim(0, max_value * 1.18)
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ax.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=4,
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
    output = args.output or stats_dir / "dxa_transactions.png"
    csv_output = args.csv or stats_dir / "dxa_transactions.csv"

    rows = load_rows(stats_dir, args.include_failed)
    write_csv(rows, csv_output)
    pdf_output = plot(rows, output)
    print(f"wrote {output}")
    print(f"wrote {pdf_output}")
    print(f"wrote {csv_output}")


if __name__ == "__main__":
    main()
