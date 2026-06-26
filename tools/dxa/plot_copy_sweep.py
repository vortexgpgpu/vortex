#!/usr/bin/env python3
"""Plot DXA copy sweep CSVs as 3x3 hardware grids of 4x4 tile heatmaps."""

import argparse
import csv
import math
from pathlib import Path


def load_rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def finite_cycles(row):
    if row.get("status") != "PASS":
        return None
    value = row.get("cycles")
    if not value:
        return None
    return float(value)


def build_speedup(rows, figure):
    by_key = {}
    for row in rows:
        if row.get("figure") != figure:
            continue
        key = (
            int(row["warps"]),
            int(row["threads"]),
            int(row["tile_rows"]),
            int(row["tile_cols"]),
        )
        by_key.setdefault(key, {})[row["variant"]] = finite_cycles(row)

    result = {}
    for key, variants in by_key.items():
        if figure == "3b":
            numer = variants.get("lsu")
            denom = variants.get("dxa")
        else:
            numer = variants.get("percta")
            denom = variants.get("mcast")
        if numer and denom and denom > 0:
            result[key] = numer / denom
    return result


def plot_figure(rows, figure, output):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.patches import Rectangle

    plt.rcParams.update({
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    speedups = build_speedup(rows, figure)
    warps = sorted({int(r["warps"]) for r in rows if r.get("figure") == figure})
    threads = sorted({int(r["threads"]) for r in rows if r.get("figure") == figure})
    tile_rows = sorted({int(r["tile_rows"]) for r in rows if r.get("figure") == figure})
    tile_cols = sorted({int(r["tile_cols"]) for r in rows if r.get("figure") == figure})

    fig, axes = plt.subplots(len(warps), len(threads), figsize=(12, 12), squeeze=False)
    values = list(speedups.values())
    vmax = max(values) if values else 1.0
    vmin = min(values) if values else 0.0
    if math.isclose(vmin, vmax):
        vmin = 0.0

    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for i, nw in enumerate(warps):
        for j, nt in enumerate(threads):
            ax = axes[i][j]
            matrix = []
            for tr in tile_rows:
                row = []
                for tc in tile_cols:
                    row.append(speedups.get((nw, nt, tr, tc), math.nan))
                matrix.append(row)
            for r, row in enumerate(matrix):
                for c, value in enumerate(row):
                    facecolor = "#ffffff" if math.isnan(value) else cmap(norm(value))
                    ax.add_patch(Rectangle(
                        (c, r), 1, 1,
                        facecolor=facecolor,
                        edgecolor="#f2f2f2",
                        linewidth=0.5,
                    ))
            ax.set_xlim(0, len(tile_cols))
            ax.set_ylim(len(tile_rows), 0)
            ax.set_aspect("equal")
            ax.set_title(f"NW={nw}, NT={nt}", fontsize=9)
            ax.set_xticks([x + 0.5 for x in range(len(tile_cols))],
                          [str(x) for x in tile_cols], fontsize=8)
            ax.set_yticks([y + 0.5 for y in range(len(tile_rows))],
                          [str(y) for y in tile_rows], fontsize=8)
            if i == len(warps) - 1:
                ax.set_xlabel("tile cols")
            if j == 0:
                ax.set_ylabel("tile rows")
            for r, tr in enumerate(tile_rows):
                for c, tc in enumerate(tile_cols):
                    value = speedups.get((nw, nt, tr, tc))
                    label = "NA" if value is None else f"{value:.2f}"
                    color = "#555555" if value is None else (
                        "white" if value < (vmin + vmax) / 2 else "black"
                    )
                    ax.text(c + 0.5, r + 0.5, label, ha="center", va="center",
                            fontsize=7, color=color)

    title = "Figure 3(b): LSU / DXA speedup" if figure == "3b" else "Figure 3(c): per-CTA / multicast speedup"
    if any(math.isnan(speedups.get((nw, nt, tr, tc), math.nan))
           for nw in warps for nt in threads for tr in tile_rows for tc in tile_cols):
        title += " (NA = timeout or missing pair)"
    fig.suptitle(title)
    fig.subplots_adjust(right=0.86, top=0.92, wspace=0.25, hspace=0.35)
    if values:
        cax = fig.add_axes([0.89, 0.18, 0.025, 0.64])
        steps = 96
        span = vmax - vmin
        for k in range(steps):
            y0 = vmin + span * k / steps
            y1 = vmin + span * (k + 1) / steps
            cax.add_patch(Rectangle(
                (0, y0), 1, y1 - y0,
                facecolor=cmap(norm((y0 + y1) * 0.5)),
                edgecolor="none",
            ))
        cax.set_xlim(0, 1)
        cax.set_ylim(vmin, vmax)
        cax.set_xticks([])
        cax.yaxis.tick_right()
        cax.yaxis.set_label_position("right")
        cax.set_ylabel("speedup", rotation=270, labelpad=18)
    output = Path(output)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=200, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--figure", choices=("3b", "3c", "both"), default="both")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/results/dxa_copy_sweep"))
    args = parser.parse_args()

    rows = load_rows(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.figure in ("3b", "both"):
        plot_figure(rows, "3b", args.out_dir / "figure3b_speedup.png")
    if args.figure in ("3c", "both"):
        plot_figure(rows, "3c", args.out_dir / "figure3c_speedup.png")


if __name__ == "__main__":
    main()
