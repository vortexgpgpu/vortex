#!/usr/bin/env python3
"""Render v1 ThreadSplit microbenchmark figures from final CSVs.

The synchronization figures aggregate parameter points within a benchmark: geometric
means for normalized time/instructions and arithmetic means for SIMD utilization.
ITS deliberately remains a label-only placeholder until measured rows are available.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SYNC_ORDER = ['tl_cg', 'tl_fg', 'am_cg', 'am_fg', 'lockht', 'atm', 'lclist', 'cp_ds', 'bh_st']
CONTROL_ORDER = ['atomicreduce', 'histogram', 'psort', 'pathfinder']
LABELS = {
    'tl_cg': 'TL-CG', 'tl_fg': 'TL-FG', 'am_cg': 'AM-CG', 'am_fg': 'AM-FG',
    'lockht': 'HT', 'atm': 'ATM', 'lclist': 'LCList', 'cp_ds': 'CP-DS',
    'bh_st': 'BH-ST', 'atomicreduce': 'Atomic\nReduce', 'histogram': 'Histogram',
    'psort': 'Rank\nSort', 'pathfinder': 'Pathfinder',
}
CONFIGS = ['IPDOM_native', 'IPDOM_SW', 'SCS', 'ITS']
CONFIG_LABELS = {
    'IPDOM_native': 'IPDOM-native', 'IPDOM_SW': 'IPDOM-SW',
    'SCS': 'SCS', 'ITS': 'ITS (pending)',
}
COLORS = {'IPDOM_native': '#0072B2', 'IPDOM_SW': '#D55E00', 'SCS': '#009E73'}
THREADS = [4, 8, 16, 32]

plt.rcParams.update({
    'font.size': 8, 'font.family': 'sans-serif', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.24,
    'grid.linewidth': 0.5, 'figure.dpi': 160, 'savefig.dpi': 300,
    'legend.frameon': False,
})


def read_csv(path):
    with path.open() as infile:
        return list(csv.DictReader(infile))


def aggregate(rows, workloads, threads, configuration, metric, geometric):
    """Return one value/status for each workload at a width/configuration."""
    result = {}
    for workload in workloads:
        matching = [r for r in rows if int(r['threads']) == threads
                    and r['benchmark'] == workload and r['configuration'] == configuration]
        passed = [float(r[metric]) for r in matching if r['status'] == 'PASS' and r[metric]]
        if passed:
            value = math.exp(sum(math.log(v) for v in passed) / len(passed)) if geometric else sum(passed) / len(passed)
            result[workload] = ('PASS', value)
        elif matching and all(r['status'] == 'DEADLOCK' for r in matching):
            result[workload] = ('DEADLOCK', None)
        else:
            result[workload] = ('PENDING', None)
    return result


def panel_limits(rows, workloads, metric, geometric, log_scale):
    values = []
    for threads in THREADS:
        for config in CONFIGS[:-1]:
            values.extend(value for status, value in aggregate(rows, workloads, threads, config, metric, geometric).values()
                          if status == 'PASS')
    if log_scale:
        return 10 ** math.floor(math.log10(min(values))), 10 ** math.ceil(math.log10(max(values) * 2.0))
    return 0.0, max(1.0, max(values) * 1.22)


def legend_handles():
    return [
        Line2D([], [], color=COLORS['IPDOM_native'], marker='$\mathrm{DNL}$', linestyle='none', markersize=12,
               label='IPDOM-native (DNL)'),
        Patch(facecolor=COLORS['IPDOM_SW'], label='IPDOM-SW'),
        Patch(facecolor=COLORS['SCS'], label='SCS'),
        Line2D([], [], color='#777777', marker='$\mathrm{ITS}$', linestyle='none', markersize=12,
               label='ITS (pending)'),
    ]


def draw_panel(ax, rows, workloads, threads, metric, geometric, ylabel, log_scale, limits):
    bar_width = 0.20
    xs = list(range(len(workloads)))
    offsets = {'IPDOM_native': -1.5 * bar_width, 'IPDOM_SW': -0.5 * bar_width,
               'SCS': 0.5 * bar_width, 'ITS': 1.5 * bar_width}
    low, high = limits
    marker_y = high / 1.32 if log_scale else high * 0.93
    for config in CONFIGS:
        values = aggregate(rows, workloads, threads, config, metric, geometric)
        for index, workload in enumerate(workloads):
            status, value = values[workload]
            x = xs[index] + offsets[config]
            if status == 'PASS':
                ax.bar(x, value, width=bar_width, color=COLORS[config], edgecolor='white', linewidth=0.35, zorder=3)
            elif status == 'DEADLOCK':
                ax.text(x, marker_y, 'DNL', ha='center', va='center', color=COLORS[config], rotation=90,
                        fontsize=6.2, fontweight='bold', clip_on=True)
            elif config == 'ITS':
                ax.text(x, marker_y, 'ITS', ha='center', va='center', color='#777777', rotation=90,
                        fontsize=6.2, fontweight='bold', clip_on=True)
    ax.set_title(f'$W={threads}$')
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[w] for w in workloads], rotation=38, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_ylim(low, high)
    if log_scale:
        ax.set_yscale('log')
        ax.axhline(1.0, color='#555555', linestyle=':', linewidth=0.7, zorder=2)
    ax.tick_params(axis='x', length=0)


def sync_figure(rows, output, metric, geometric, ylabel, filename, log_scale=True):
    limits = panel_limits(rows, SYNC_ORDER, metric, geometric, log_scale)
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.65), sharey=True)
    for ax, threads in zip(axes.flat, THREADS):
        draw_panel(ax, rows, SYNC_ORDER, threads, metric, geometric, ylabel, log_scale, limits)
    fig.legend(handles=legend_handles(), loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.01),
               columnspacing=1.5, handletextpad=0.45)
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.8, h_pad=1.3, w_pad=0.6)
    save(fig, output, filename)


def control_figure(rows, output):
    metric = 'normalized_time'
    limits = panel_limits(rows, CONTROL_ORDER, metric, True, True)
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.35), sharey=True)
    for ax, threads in zip(axes.flat, THREADS):
        draw_panel(ax, rows, CONTROL_ORDER, threads, metric, True, 'Normalized cycles', True, limits)
    fig.legend(handles=legend_handles(), loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.01),
               columnspacing=1.5, handletextpad=0.45)
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.8, h_pad=1.2, w_pad=0.6)
    save(fig, output, 'control_overhead_time')


def write_summary(rows, output):
    fields = ['threads', 'benchmark', 'configuration', 'status', 'normalized_time_geomean',
              'normalized_instructions_geomean', 'simd_utilization_mean']
    with (output / 'microbenchmark-aggregates.csv').open('w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fields)
        writer.writeheader()
        for threads in THREADS:
            for workload in SYNC_ORDER + CONTROL_ORDER:
                for config in CONFIGS:
                    time_status, time_value = aggregate(rows, [workload], threads, config, 'normalized_time', True)[workload]
                    instr_status, instr_value = aggregate(rows, [workload], threads, config, 'normalized_instructions', True)[workload]
                    util_status, util_value = aggregate(rows, [workload], threads, config, 'simd_utilization', False)[workload]
                    writer.writerow({
                        'threads': threads, 'benchmark': workload, 'configuration': config,
                        'status': time_status,
                        'normalized_time_geomean': '' if time_value is None else time_value,
                        'normalized_instructions_geomean': '' if instr_value is None else instr_value,
                        'simd_utilization_mean': '' if util_value is None else util_value,
                    })


def save(fig, output, filename):
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f'{filename}.pdf', bbox_inches='tight')
    fig.savefig(output / f'{filename}.png', bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path,
                        default=Path(__file__).resolve().parent / 'final-csvs' / 'threadsplit-microbench.csv')
    parser.add_argument('--output', type=Path,
                        default=Path('/home/nikhilrout97/-ASPLOS-27-ThreadSplit-Copy-/figs'))
    args = parser.parse_args()
    rows = read_csv(args.input)
    sync_figure(rows, args.output, 'normalized_time', True, 'Normalized cycles', 'microbench_time')
    sync_figure(rows, args.output, 'normalized_instructions', True, 'Normalized warp instructions',
                'microbench_instructions')
    sync_figure(rows, args.output, 'simd_utilization', False, 'Average SIMD utilization',
                'microbench_utilization', log_scale=False)
    control_figure(rows, args.output)
    write_summary(rows, args.output)


if __name__ == '__main__':
    main()
