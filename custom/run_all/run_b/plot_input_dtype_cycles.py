#!/usr/bin/env python3
import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


RUN_RE = re.compile(r"(?:run_b_|run)(?P<run>\d+)")
TESTBENCH_RE = re.compile(r"^Testbench:\s*(?P<testbench>\S+)")
META_RE = re.compile(
    r"(?:Sparsity Mode:(?P<sparsity_mode>\d+)\s*\|\s*)?"
    r"MxNxK=(?P<shape>\d+x\d+x\d+).*?\|\s*It=(?P<dtype>\S+)"
)
SPARSITY_RE = re.compile(
    r"A sparsity=(?P<a_sparsity>[0-9.]+)%\s*\|\s*"
    r"B sparsity=(?P<b_sparsity>[0-9.]+)%"
)
STATUS_RE = re.compile(r"^(PASS|FAIL)$")
TOTAL_INSTRUCTIONS_RE = re.compile(r"^Total kernel instructions:\s*(?P<value>\d+)")
KERNEL_BODY_RE = re.compile(r"^Kernel body cycles:\s*(?P<value>\d+)")
TCU_UTIL_RE = re.compile(r"^TCU utilization:\s*(?P<value>[0-9.]+)%")
MEMORY_REQS_RE = re.compile(
    r"^(?:PERF2:\s*)?memory:\s*reqs=(?P<total>\d+)\s*"
    r"\(r=(?P<reads>\d+),\s*w=(?P<writes>\d+)\)"
)
ICACHE_RE = re.compile(
    r"^(?:PERF2:\s*)?core\d+:\s*icache:\s*reads=(?P<reads>\d+),\s*"
    r"miss=(?P<misses>\d+)"
)
DCACHE_RE = re.compile(
    r"^(?:PERF2:\s*)?core\d+:\s*dcache:\s*reqs=(?P<reqs>\d+),\s*"
    r"miss_r=(?P<miss_reads>\d+).*?miss_w=(?P<miss_writes>\d+).*?"
    r"bank_st=(?P<bank_stalls>\d+)"
)

DTYPE_ORDER = ["fp8", "fp16", "fp32"]
VARIANT_ORDER = [
    "sgemm_tcu",
    "sgemm_tcu_sp",
    "sgemm_tcu_op_dense",
    "sgemm_tcu_op_s20",
    "sgemm_tcu_op_s50",
    "sgemm_tcu_op_s90",
]
VARIANT_LABELS = {
    "sgemm_tcu": "sgemm_tcu",
    "sgemm_tcu_sp": "sgemm_tcu_sp",
    "sgemm_tcu_op_dense": "sgemm_tcu_op dense",
    "sgemm_tcu_op_s20": "sgemm_tcu_op 20%",
    "sgemm_tcu_op_s50": "sgemm_tcu_op 50%",
    "sgemm_tcu_op_s90": "sgemm_tcu_op 90%",
}
VARIANT_COLORS = {
    "sgemm_tcu": "#4f5d75",
    "sgemm_tcu_sp": "#7a5ea8",
    "sgemm_tcu_op_dense": "#2f6f9f",
    "sgemm_tcu_op_s20": "#558b2f",
    "sgemm_tcu_op_s50": "#b8792d",
    "sgemm_tcu_op_s90": "#c7533b",
}
CACHE_PLOT_SPECS = {
    "icache": {
        "label": "ICache",
        "total": "icache_reqs",
        "components": [
            ("icache_hits", "Hits", "#558b2f"),
            ("icache_misses", "Misses", "#c7533b"),
        ],
    },
    "dcache": {
        "label": "DCache",
        "total": "dcache_reqs",
        "components": [
            ("dcache_hits", "Hits", "#558b2f"),
            ("dcache_miss_reads", "Read Misses", "#2f6f9f"),
            ("dcache_miss_writes", "Write Misses", "#c7533b"),
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot run_b stats for kernels, dtypes, and sgemm_tcu_op sparsity modes."
    )
    parser.add_argument(
        "stats_dir",
        nargs="?",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Directory containing run_b_1.stat through run_b_18.stat.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        default=True,
        help="Include failed stats when metrics are present. Default: enabled.",
    )
    return parser.parse_args()


def rounded_percent(value):
    return int(round(value))


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
        "sparsity_mode": None,
        "a_sparsity": None,
        "b_sparsity": None,
        "sparsity_label": "",
        "variant": "",
        "variant_label": "",
        "total_instructions": None,
        "kernel_body_cycles": None,
        "tcu_utilization": None,
        "memory_reqs": None,
        "memory_reads": None,
        "memory_writes": None,
        "icache_reqs": None,
        "icache_hits": None,
        "icache_misses": None,
        "dcache_reqs": None,
        "dcache_hits": None,
        "dcache_misses": None,
        "dcache_miss_reads": None,
        "dcache_miss_writes": None,
        "dcache_bank_stalls": None,
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
                if match.group("sparsity_mode") is not None:
                    row["sparsity_mode"] = int(match.group("sparsity_mode"))
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
                continue

            match = ICACHE_RE.match(line)
            if match and row["icache_reqs"] is None:
                row["icache_reqs"] = int(match.group("reads"))
                row["icache_misses"] = int(match.group("misses"))
                row["icache_hits"] = row["icache_reqs"] - row["icache_misses"]
                continue

            match = DCACHE_RE.match(line)
            if match and row["dcache_reqs"] is None:
                row["dcache_reqs"] = int(match.group("reqs"))
                row["dcache_miss_reads"] = int(match.group("miss_reads"))
                row["dcache_miss_writes"] = int(match.group("miss_writes"))
                row["dcache_misses"] = row["dcache_miss_reads"] + row["dcache_miss_writes"]
                row["dcache_hits"] = row["dcache_reqs"] - row["dcache_misses"]
                row["dcache_bank_stalls"] = int(match.group("bank_stalls"))

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
            "icache_reqs",
            "icache_hits",
            "icache_misses",
            "dcache_reqs",
            "dcache_hits",
            "dcache_misses",
            "dcache_miss_reads",
            "dcache_miss_writes",
            "dcache_bank_stalls",
        )
        if row[key] in ("", None)
    ]
    if missing:
        raise ValueError(f"{path}: missing required fields: {', '.join(missing)}")

    apply_variant(row)
    return row


def apply_variant(row):
    if row["testbench"] in ("sgemm_tcu", "sgemm_tcu_sp"):
        row["sparsity_mode"] = 0 if row["sparsity_mode"] is None else row["sparsity_mode"]
        row["a_sparsity"] = 0.0 if row["a_sparsity"] is None else row["a_sparsity"]
        row["b_sparsity"] = 0.0 if row["b_sparsity"] is None else row["b_sparsity"]
        row["sparsity_label"] = "0%"
        row["variant"] = row["testbench"]
        row["variant_label"] = VARIANT_LABELS[row["variant"]]
        return

    if row["testbench"] != "sgemm_tcu_op":
        raise ValueError(f"{row['file']}: unsupported testbench {row['testbench']}")

    if row["sparsity_mode"] == 0:
        row["sparsity_label"] = "0%"
        row["variant"] = "sgemm_tcu_op_dense"
    else:
        avg_sparsity = (row["a_sparsity"] + row["b_sparsity"]) / 2.0
        sparsity = rounded_percent(avg_sparsity)
        row["sparsity_label"] = f"{sparsity}%"
        row["variant"] = f"sgemm_tcu_op_s{sparsity}"

    if row["variant"] not in VARIANT_LABELS:
        raise ValueError(f"{row['file']}: unsupported run_b variant {row['variant']}")
    row["variant_label"] = VARIANT_LABELS[row["variant"]]


def load_rows(stats_dir, include_failed):
    rows = []
    skipped = []
    for run in range(1, 19):
        path = stats_dir / f"run_b_{run}.stat"
        if not path.exists():
            raise FileNotFoundError(f"missing required stat file: {path}")
        row = parse_stat(path)
        if include_failed or row["status"] == "PASS":
            rows.append(row)
        else:
            skipped.append(path.name)

    if skipped:
        print(f"warning: skipped non-passing stats: {', '.join(skipped)}")
    return sort_rows(rows)


def sort_rows(rows):
    return sorted(
        rows,
        key=lambda row: (
            DTYPE_ORDER.index(row["dtype"]),
            VARIANT_ORDER.index(row["variant"]),
        ),
    )


def save_figure(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    pdf_output = output.with_suffix(".pdf")
    if pdf_output != output:
        fig.savefig(pdf_output, bbox_inches="tight")
    return pdf_output


def metric_fieldnames():
    return [
        "file",
        "run",
        "testbench",
        "variant",
        "variant_label",
        "status",
        "shape",
        "dtype",
        "sparsity_mode",
        "a_sparsity",
        "b_sparsity",
        "sparsity_label",
        "total_instructions",
        "kernel_body_cycles",
        "tcu_utilization",
        "memory_reqs",
        "memory_reads",
        "memory_writes",
        "icache_reqs",
        "icache_hits",
        "icache_misses",
        "dcache_reqs",
        "dcache_hits",
        "dcache_misses",
        "dcache_miss_reads",
        "dcache_miss_writes",
        "dcache_bank_stalls",
    ]


def write_csv(rows, output, fieldnames=None):
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or metric_fieldnames()
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def grouped_positions():
    group_width = 0.84
    bar_width = group_width / len(VARIANT_ORDER)
    centers = list(range(len(DTYPE_ORDER)))
    positions = {}
    for dtype_index, dtype in enumerate(DTYPE_ORDER):
        for variant_index, variant in enumerate(VARIANT_ORDER):
            positions[(dtype, variant)] = (
                centers[dtype_index] - group_width / 2 + bar_width * (variant_index + 0.5)
            )
    return centers, positions, bar_width


def plot_metric(rows, output, metric, title, ylabel, percent=False):
    by_key = {(row["dtype"], row["variant"]): row for row in rows}
    centers, positions, bar_width = grouped_positions()
    values = [row[metric] for row in rows]
    shape = next((row["shape"] for row in rows if row["shape"]), "")

    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    for variant in VARIANT_ORDER:
        x_values = []
        y_values = []
        for dtype in DTYPE_ORDER:
            row = by_key.get((dtype, variant))
            if row is None:
                continue
            x_values.append(positions[(dtype, variant)])
            y_values.append(row[metric])
        ax.bar(
            x_values,
            y_values,
            width=bar_width * 0.88,
            label=VARIANT_LABELS[variant],
            color=VARIANT_COLORS[variant],
            edgecolor="#222222",
            linewidth=0.8,
        )

    ax.set_title(f"{title}\nMxNxK={shape}")
    ax.set_xlabel("Input Data Type")
    ax.set_ylabel(ylabel)
    ax.set_xticks(centers)
    ax.set_xticklabels(DTYPE_ORDER)
    ymax = max(values) * 1.18
    if percent:
        ymax = min(100, ymax)
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        title="Kernel / OP Sparsity Mode",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
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


def plot_memory(rows, output):
    by_key = {(row["dtype"], row["variant"]): row for row in rows}
    centers, positions, bar_width = grouped_positions()
    totals = [row["memory_reqs"] for row in rows]
    shape = next((row["shape"] for row in rows if row["shape"]), "")
    hatches = {
        "sgemm_tcu": "",
        "sgemm_tcu_sp": "..",
        "sgemm_tcu_op_dense": "//",
        "sgemm_tcu_op_s20": "\\\\",
        "sgemm_tcu_op_s50": "xx",
        "sgemm_tcu_op_s90": "++",
    }

    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    for variant in VARIANT_ORDER:
        x_values = []
        writes = []
        reads = []
        for dtype in DTYPE_ORDER:
            row = by_key.get((dtype, variant))
            if row is None:
                continue
            x_values.append(positions[(dtype, variant)])
            writes.append(row["memory_writes"])
            reads.append(row["memory_reads"])

        ax.bar(
            x_values,
            writes,
            width=bar_width * 0.88,
            color="#c7533b",
            edgecolor="#222222",
            linewidth=0.8,
            hatch=hatches[variant],
        )
        ax.bar(
            x_values,
            reads,
            width=bar_width * 0.88,
            bottom=writes,
            color="#2f6f9f",
            edgecolor="#222222",
            linewidth=0.8,
            hatch=hatches[variant],
        )

    ax.set_title(f"Memory Requests by Input Data Type and Kernel\nMxNxK={shape}")
    ax.set_xlabel("Input Data Type")
    ax.set_ylabel("Memory Requests")
    ax.set_xticks(centers)
    ax.set_xticklabels(DTYPE_ORDER)
    ax.set_ylim(0, max(totals) * 1.15)
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    request_handles = [
        Patch(facecolor="#2f6f9f", edgecolor="#222222", label="Reads"),
        Patch(facecolor="#c7533b", edgecolor="#222222", label="Writes"),
    ]
    variant_handles = [
        Patch(
            facecolor="white",
            edgecolor="#222222",
            hatch=hatches[variant],
            label=VARIANT_LABELS[variant],
        )
        for variant in VARIANT_ORDER
    ]
    ax.legend(
        handles=request_handles + variant_handles,
        title="Request Type",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
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


def plot_cache_requests(rows, output, cache_name):
    spec = CACHE_PLOT_SPECS[cache_name]
    by_key = {(row["dtype"], row["variant"]): row for row in rows}
    centers, positions, bar_width = grouped_positions()
    totals = [row[spec["total"]] for row in rows]
    shape = next((row["shape"] for row in rows if row["shape"]), "")
    hatches = {
        "sgemm_tcu": "",
        "sgemm_tcu_sp": "..",
        "sgemm_tcu_op_dense": "//",
        "sgemm_tcu_op_s20": "\\\\",
        "sgemm_tcu_op_s50": "xx",
        "sgemm_tcu_op_s90": "++",
    }

    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    for variant in VARIANT_ORDER:
        x_values = []
        component_values = [[] for _ in spec["components"]]
        for dtype in DTYPE_ORDER:
            row = by_key.get((dtype, variant))
            if row is None:
                continue
            x_values.append(positions[(dtype, variant)])
            for component_index, (field, _label, _color) in enumerate(spec["components"]):
                component_values[component_index].append(row[field])

        bottoms = [0] * len(x_values)
        for component_index, (_field, _label, color) in enumerate(spec["components"]):
            values = component_values[component_index]
            ax.bar(
                x_values,
                values,
                width=bar_width * 0.88,
                bottom=bottoms,
                color=color,
                edgecolor="#222222",
                linewidth=0.8,
                hatch=hatches[variant],
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_title(f"{spec['label']} Requests by Input Data Type and Kernel\nMxNxK={shape}")
    ax.set_xlabel("Input Data Type")
    ax.set_ylabel(f"{spec['label']} Requests")
    ax.set_xticks(centers)
    ax.set_xticklabels(DTYPE_ORDER)
    ax.set_ylim(0, max(totals) * 1.15)
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    request_handles = [
        Patch(facecolor=color, edgecolor="#222222", label=label)
        for _field, label, color in spec["components"]
    ]
    variant_handles = [
        Patch(
            facecolor="white",
            edgecolor="#222222",
            hatch=hatches[variant],
            label=VARIANT_LABELS[variant],
        )
        for variant in VARIANT_ORDER
    ]
    ax.legend(
        handles=request_handles + variant_handles,
        title="Request Type",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
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


def op_rows(rows):
    selected = [row for row in rows if row["testbench"] == "sgemm_tcu_op"]
    return sort_rows(selected)


def plot_op_sparsity_metric(rows, output, metric, title, ylabel, percent=False):
    selected = op_rows(rows)
    by_key = {(row["dtype"], row["variant"]): row for row in selected}
    variants = [
        "sgemm_tcu_op_dense",
        "sgemm_tcu_op_s20",
        "sgemm_tcu_op_s50",
        "sgemm_tcu_op_s90",
    ]
    x_positions = list(range(len(variants)))
    labels = ["0%", "20%", "50%", "90%"]
    values = [row[metric] for row in selected]
    shape = selected[0]["shape"]

    fig, ax = plt.subplots(figsize=(9.8, 5.9))
    colors = {"fp8": "#2f6f9f", "fp16": "#558b2f", "fp32": "#c7533b"}
    group_width = 0.72
    bar_width = group_width / len(DTYPE_ORDER)

    for dtype_index, dtype in enumerate(DTYPE_ORDER):
        offsets = [
            x - group_width / 2 + bar_width * (dtype_index + 0.5)
            for x in x_positions
        ]
        y_values = [by_key[(dtype, variant)][metric] for variant in variants]
        ax.bar(
            offsets,
            y_values,
            width=bar_width * 0.88,
            label=dtype,
            color=colors[dtype],
            edgecolor="#222222",
            linewidth=0.8,
        )

    ax.set_title(f"{title}\nMxNxK={shape}, Kernel=sgemm_tcu_op")
    ax.set_xlabel("Average A/B Sparsity")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ymax = max(values) * 1.18
    if percent:
        ymax = min(100, ymax)
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        title="Input Data Type",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
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


def plot_op_sparsity_metric_by_dtype(rows, output, metric, title, ylabel, percent=False):
    selected = op_rows(rows)
    by_key = {(row["dtype"], row["variant"]): row for row in selected}
    variants = [
        "sgemm_tcu_op_dense",
        "sgemm_tcu_op_s20",
        "sgemm_tcu_op_s50",
        "sgemm_tcu_op_s90",
    ]
    variant_labels = {
        "sgemm_tcu_op_dense": "0%",
        "sgemm_tcu_op_s20": "20%",
        "sgemm_tcu_op_s50": "50%",
        "sgemm_tcu_op_s90": "90%",
    }
    colors = {
        "sgemm_tcu_op_dense": "#2f6f9f",
        "sgemm_tcu_op_s20": "#558b2f",
        "sgemm_tcu_op_s50": "#b8792d",
        "sgemm_tcu_op_s90": "#c7533b",
    }
    centers = list(range(len(DTYPE_ORDER)))
    values = [row[metric] for row in selected]
    shape = selected[0]["shape"]
    group_width = 0.72
    bar_width = group_width / len(variants)

    fig, ax = plt.subplots(figsize=(9.8, 5.9))
    for variant_index, variant in enumerate(variants):
        offsets = [
            x - group_width / 2 + bar_width * (variant_index + 0.5)
            for x in centers
        ]
        y_values = [by_key[(dtype, variant)][metric] for dtype in DTYPE_ORDER]
        ax.bar(
            offsets,
            y_values,
            width=bar_width * 0.88,
            label=variant_labels[variant],
            color=colors[variant],
            edgecolor="#222222",
            linewidth=0.8,
        )

    ax.set_title(f"{title}\nMxNxK={shape}, Kernel=sgemm_tcu_op")
    ax.set_xlabel("Input Data Type")
    ax.set_ylabel(ylabel)
    ax.set_xticks(centers)
    ax.set_xticklabels(DTYPE_ORDER)
    ymax = max(values) * 1.18
    if percent:
        ymax = min(100, ymax)
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        title="Average A/B Sparsity",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
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


def plot_op_sparsity_memory(rows, output):
    selected = op_rows(rows)
    by_key = {(row["dtype"], row["variant"]): row for row in selected}
    variants = [
        "sgemm_tcu_op_dense",
        "sgemm_tcu_op_s20",
        "sgemm_tcu_op_s50",
        "sgemm_tcu_op_s90",
    ]
    variant_labels = {
        "sgemm_tcu_op_dense": "0%",
        "sgemm_tcu_op_s20": "20%",
        "sgemm_tcu_op_s50": "50%",
        "sgemm_tcu_op_s90": "90%",
    }
    hatches = {
        "sgemm_tcu_op_dense": "",
        "sgemm_tcu_op_s20": "\\\\",
        "sgemm_tcu_op_s50": "xx",
        "sgemm_tcu_op_s90": "++",
    }
    centers = list(range(len(DTYPE_ORDER)))
    totals = [row["memory_reqs"] for row in selected]
    shape = selected[0]["shape"]
    group_width = 0.72
    bar_width = group_width / len(variants)

    fig, ax = plt.subplots(figsize=(11.8, 5.9))
    for variant_index, variant in enumerate(variants):
        offsets = [
            x - group_width / 2 + bar_width * (variant_index + 0.5)
            for x in centers
        ]
        writes = [by_key[(dtype, variant)]["memory_writes"] for dtype in DTYPE_ORDER]
        reads = [by_key[(dtype, variant)]["memory_reads"] for dtype in DTYPE_ORDER]
        ax.bar(
            offsets,
            writes,
            width=bar_width * 0.88,
            color="#c7533b",
            edgecolor="#222222",
            linewidth=0.8,
            hatch=hatches[variant],
        )
        ax.bar(
            offsets,
            reads,
            width=bar_width * 0.88,
            bottom=writes,
            color="#2f6f9f",
            edgecolor="#222222",
            linewidth=0.8,
            hatch=hatches[variant],
        )

    ax.set_title(f"Memory Requests by Input Data Type and OP Sparsity\nMxNxK={shape}, Kernel=sgemm_tcu_op")
    ax.set_xlabel("Input Data Type")
    ax.set_ylabel("Memory Requests")
    ax.set_xticks(centers)
    ax.set_xticklabels(DTYPE_ORDER)
    ax.set_ylim(0, max(totals) * 1.15)
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    request_handles = [
        Patch(facecolor="#2f6f9f", edgecolor="#222222", label="Reads"),
        Patch(facecolor="#c7533b", edgecolor="#222222", label="Writes"),
    ]
    sparsity_handles = [
        Patch(
            facecolor="white",
            edgecolor="#222222",
            hatch=hatches[variant],
            label=variant_labels[variant],
        )
        for variant in variants
    ]
    ax.legend(
        handles=request_handles + sparsity_handles,
        title="Request Type / Average A/B Sparsity",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
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


def plot_op_sparsity_cache_requests(rows, output, cache_name):
    spec = CACHE_PLOT_SPECS[cache_name]
    selected = op_rows(rows)
    by_key = {(row["dtype"], row["variant"]): row for row in selected}
    variants = [
        "sgemm_tcu_op_dense",
        "sgemm_tcu_op_s20",
        "sgemm_tcu_op_s50",
        "sgemm_tcu_op_s90",
    ]
    variant_labels = {
        "sgemm_tcu_op_dense": "0%",
        "sgemm_tcu_op_s20": "20%",
        "sgemm_tcu_op_s50": "50%",
        "sgemm_tcu_op_s90": "90%",
    }
    hatches = {
        "sgemm_tcu_op_dense": "",
        "sgemm_tcu_op_s20": "\\\\",
        "sgemm_tcu_op_s50": "xx",
        "sgemm_tcu_op_s90": "++",
    }
    centers = list(range(len(DTYPE_ORDER)))
    totals = [row[spec["total"]] for row in selected]
    shape = selected[0]["shape"]
    group_width = 0.72
    bar_width = group_width / len(variants)

    fig, ax = plt.subplots(figsize=(11.8, 5.9))
    for variant_index, variant in enumerate(variants):
        offsets = [
            x - group_width / 2 + bar_width * (variant_index + 0.5)
            for x in centers
        ]
        component_values = [
            [by_key[(dtype, variant)][field] for dtype in DTYPE_ORDER]
            for field, _label, _color in spec["components"]
        ]

        bottoms = [0] * len(offsets)
        for component_index, (_field, _label, color) in enumerate(spec["components"]):
            values = component_values[component_index]
            ax.bar(
                offsets,
                values,
                width=bar_width * 0.88,
                bottom=bottoms,
                color=color,
                edgecolor="#222222",
                linewidth=0.8,
                hatch=hatches[variant],
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_title(f"{spec['label']} Requests by Input Data Type and OP Sparsity\nMxNxK={shape}, Kernel=sgemm_tcu_op")
    ax.set_xlabel("Input Data Type")
    ax.set_ylabel(f"{spec['label']} Requests")
    ax.set_xticks(centers)
    ax.set_xticklabels(DTYPE_ORDER)
    ax.set_ylim(0, max(totals) * 1.15)
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    request_handles = [
        Patch(facecolor=color, edgecolor="#222222", label=label)
        for _field, label, color in spec["components"]
    ]
    sparsity_handles = [
        Patch(
            facecolor="white",
            edgecolor="#222222",
            hatch=hatches[variant],
            label=variant_labels[variant],
        )
        for variant in variants
    ]
    ax.legend(
        handles=request_handles + sparsity_handles,
        title="Request Type / Average A/B Sparsity",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
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
    rows = load_rows(stats_dir, args.include_failed)
    op_selected = op_rows(rows)

    outputs = [
        (
            stats_dir / "fp16_total_instructions.png",
            plot_metric(
                rows,
                stats_dir / "fp16_total_instructions.png",
                "total_instructions",
                "Total Kernel Instructions by Input Data Type",
                "Total Kernel Instructions",
            ),
            stats_dir / "fp16_total_instructions.csv",
        ),
        (
            stats_dir / "input_dtype_kernel_body_cycles.png",
            plot_metric(
                rows,
                stats_dir / "input_dtype_kernel_body_cycles.png",
                "kernel_body_cycles",
                "Kernel Body Cycles by Input Data Type",
                "Kernel Body Cycles",
            ),
            stats_dir / "input_dtype_kernel_body_cycles.csv",
        ),
        (
            stats_dir / "input_dtype_tcu_utilization.png",
            plot_metric(
                rows,
                stats_dir / "input_dtype_tcu_utilization.png",
                "tcu_utilization",
                "TCU Utilization by Input Data Type",
                "TCU Utilization (%)",
                percent=True,
            ),
            stats_dir / "input_dtype_tcu_utilization.csv",
        ),
        (
            stats_dir / "memory_requests_read_write.png",
            plot_memory(rows, stats_dir / "memory_requests_read_write.png"),
            stats_dir / "memory_requests_read_write.csv",
        ),
        (
            stats_dir / "dcache_requests_hits_misses.png",
            plot_cache_requests(rows, stats_dir / "dcache_requests_hits_misses.png", "dcache"),
            stats_dir / "dcache_requests_hits_misses.csv",
        ),
        (
            stats_dir / "sparsity_total_instructions.png",
            plot_op_sparsity_metric_by_dtype(
                rows,
                stats_dir / "sparsity_total_instructions.png",
                "total_instructions",
                "Total Kernel Instructions by Input Data Type and OP Sparsity",
                "Total Kernel Instructions",
            ),
            None,
        ),
        (
            stats_dir / "sparsity_kernel_body_cycles.png",
            plot_op_sparsity_metric_by_dtype(
                rows,
                stats_dir / "sparsity_kernel_body_cycles.png",
                "kernel_body_cycles",
                "Kernel Body Cycles by Input Data Type and OP Sparsity",
                "Kernel Body Cycles",
            ),
            None,
        ),
        (
            stats_dir / "sparsity_tcu_utilization.png",
            plot_op_sparsity_metric_by_dtype(
                rows,
                stats_dir / "sparsity_tcu_utilization.png",
                "tcu_utilization",
                "TCU Utilization by Input Data Type and OP Sparsity",
                "TCU Utilization (%)",
                percent=True,
            ),
            None,
        ),
        (
            stats_dir / "sparsity_memory_requests.png",
            plot_op_sparsity_memory(rows, stats_dir / "sparsity_memory_requests.png"),
            None,
        ),
        (
            stats_dir / "sparsity_dcache_requests.png",
            plot_op_sparsity_cache_requests(
                rows,
                stats_dir / "sparsity_dcache_requests.png",
                "dcache",
            ),
            None,
        ),
    ]

    write_csv(rows, stats_dir / "fp16_total_instructions.csv")
    write_csv(rows, stats_dir / "input_dtype_kernel_body_cycles.csv")
    write_csv(rows, stats_dir / "input_dtype_tcu_utilization.csv")
    write_csv(rows, stats_dir / "memory_requests_read_write.csv")
    write_csv(rows, stats_dir / "dcache_requests_hits_misses.csv")
    write_csv(op_selected, stats_dir / "sparsity_sgemm_tcu_op.csv")

    for png_output, pdf_output, csv_output in outputs:
        print(f"wrote {png_output}")
        print(f"wrote {pdf_output}")
        if csv_output is not None:
            print(f"wrote {csv_output}")
    print(f"wrote {stats_dir / 'sparsity_sgemm_tcu_op.csv'}")


if __name__ == "__main__":
    main()
