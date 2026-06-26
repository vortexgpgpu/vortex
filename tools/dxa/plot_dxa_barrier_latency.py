#!/usr/bin/env python3
"""Plot hard-vs-soft DXA barrier release-latency proxy sweeps."""

import argparse
import csv
from pathlib import Path


def load_rows(path):
    with Path(path).open(newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("status") == "PASS"]


def as_int(row, key, default=0):
    value = row.get(key)
    if value in (None, ""):
        return default
    return int(float(value))


def group_pairs(rows):
    pairs = {}
    for row in rows:
        payload = as_int(row, "payload_bytes")
        pairs.setdefault(payload, {})[row.get("mode", "")] = row
    return [(payload, modes) for payload, modes in sorted(pairs.items())]


def plot(rows, output):
    import matplotlib.pyplot as plt

    pairs = [
        (payload, modes)
        for payload, modes in group_pairs(rows)
        if "hard_dxa_completion" in modes and "soft_dxa_completion" in modes
    ]
    if not pairs:
        raise SystemExit("no complete hard/soft PASS pairs to plot")

    payloads = [payload for payload, _ in pairs]
    labels = [f"{payload // 1024}KB" for payload in payloads]
    hard_release = [as_int(modes["hard_dxa_completion"], "release_cycles") for _, modes in pairs]
    soft_release = [as_int(modes["soft_dxa_completion"], "release_cycles") for _, modes in pairs]
    extra = [soft - hard for hard, soft in zip(hard_release, soft_release)]
    ratio = [100.0 * e / max(1, hard) for e, hard in zip(extra, hard_release)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = list(range(len(labels)))
    width = 0.38

    axes[0].bar([i - width / 2 for i in x], hard_release, width, label="hard barrier",
                color="#4C78A8")
    axes[0].bar([i + width / 2 for i in x], soft_release, width, label="soft smem+atomic",
                color="#F58518")
    axes[0].set_title("DXA End-to-End Release Latency")
    axes[0].set_ylabel("cycles from expect_tx to release")
    axes[0].set_xlabel("DXA payload")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].bar(labels, extra, color="#54A24B", label="extra cycles")
    ax2 = axes[1].twinx()
    ax2.plot(labels, ratio, color="#E45756", marker="o", label="extra / hard")
    axes[1].set_title("Soft Extra End-to-End Release Latency")
    axes[1].set_ylabel("cycles")
    ax2.set_ylabel("extra / hard release latency (%)")
    axes[1].set_xlabel("DXA payload")
    axes[1].grid(axis="y", alpha=0.25)

    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper right")

    driver = rows[0].get("driver", "")
    title = "DXA Barrier Release-Latency Proxy, Not Pure Barrier Overhead"
    if driver:
        title += f" ({driver})"
    fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/results/dxa_barrier_overhead"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit("no PASS rows to plot")

    output = args.output
    if output is None:
        output = args.out_dir / (args.csv.stem + ".png")
    plot(rows, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
