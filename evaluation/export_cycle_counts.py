#!/usr/bin/env python3
import sys
import pandas as pd
from os import path
from glob import glob

if len(sys.argv) < 2:
    print('usage: python3 ' + sys.argv[0] + ' <path to test_outputs>')
    exit()

output_dir = sys.argv[1]


config_names = []
test_names = []
cycle_counts = []

for filename in glob(path.join(output_dir, '*.log')):
    cycle_line = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('[sim] total cycles:'):
                cycle_line = line
    print(filename, cycle_line)

    full_name, _, _ = path.basename(filename).partition('.') 

    if cycle_line is None:
        count = None
    else:
        _, _, count = cycle_line.partition(':')
        count = int(count.strip())

    config, test = full_name.rsplit('-', 1)
    config_names.append(config)
    test_names.append(test)
    cycle_counts.append(count)

df = pd.DataFrame({
    'config': config_names,
    'test': test_names,
    'cycle_count': cycle_counts,
})

print(df.head())

pivot = pd.pivot_table(df, values='cycle_count', index=['config'], columns=['test'])

print(pivot.head())

pivot.to_csv('results.csv')
print('Table written to results.csv')

