#!/usr/bin/env python3
"""Create paper-facing ThreadSplit CSVs from completed sweep rows."""

import csv
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'microbench-results'
OUT = ROOT / 'final-csvs'
SOURCES = {4: 'rtlsim-t4.csv', 8: 'rtlsim-t8.csv', 16: 'rtlsim-t16.csv', 32: 'simx-t32.csv'}
FIELDS = [
    'benchmark', 'parameters', 'threads', 'warps', 'cores', 'configuration', 'status',
    'cycles', 'dynamic_instructions', 'issued_instructions', 'active_lane_instructions',
    'simd_utilization', 'vx_yield_count', 'subgroup_switch_count',
    'watchdog_switch_count', 'peak_live_split_contexts', 'normalized_time',
    'normalized_instructions', 'wall_seconds', 'notes',
]


def read_rows(path):
    with path.open() as infile:
        rows = list(csv.DictReader(infile))
    selected = {}
    for row in rows:
        if row['status'] in {'ERROR', 'TIMEOUT'}:
            continue
        key = (row['benchmark'], row['parameters'], row['configuration'])
        selected[key] = row
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, choices=tuple(SOURCES), nargs='+')
    args = parser.parse_args()
    selected_threads = args.threads or list(SOURCES)
    OUT.mkdir(exist_ok=True)
    all_rows = []
    for threads in selected_threads:
        filename = SOURCES[threads]
        rows = read_rows(RAW / filename)
        if len(rows) != 180:
            raise SystemExit(f'{filename}: expected 180 completed rows, found {len(rows)}')
        ordered = [rows[key] for key in sorted(rows)]
        target = OUT / f'threadsplit-microbench-t{threads}.csv'
        with target.open('w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=FIELDS, lineterminator='\n')
            writer.writeheader()
            writer.writerows({field: row[field] for field in FIELDS} for row in ordered)
        all_rows.extend(ordered)
    if args.threads is None:
        with (OUT / 'threadsplit-microbench.csv').open('w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=FIELDS, lineterminator='\n')
            writer.writeheader()
            writer.writerows({field: row[field] for field in FIELDS} for row in all_rows)


if __name__ == '__main__':
    main()
