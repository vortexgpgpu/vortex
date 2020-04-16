#!/usr/bin/env python3
import os
import glob

config_location = 'configs'

name_template = '{clusters}cl-{cores}c-{warps}w-{threads}t-{l2}Kl2-{dcache}Kd-{icache}Ki{name_suffix}.sh'

template = """

export V_NT={threads}
export V_NW={warps}
export V_NUMBER_CORES_PER_CLUSTER={cores}
export V_NUMBER_CLUSTERS={clusters}
export V_DCACHE_SIZE_BYTES={dcachek}
export V_ICACHE_SIZE_BYTES={icachek}

# L2 Cache size
export V_L2CACHE_SIZE_BYTES={l2k}

{codegen}

"""

# cluster, cores, warps, threads, l2, dcache, icache
configs = [
    (1, 2, 8, 4, 8, 4, 1),
    (1, 2, 8, 8, 8, 4, 1),
    (1, 2, 8, 8, 16, 8, 1),

    (1, 4, 8, 8, 16, 4, 1),
    (1, 4, 8, 8, 16, 8, 1),
    (1, 4, 16, 8, 16, 8, 1),

    (2, 4, 8, 4, 8, 4, 1),
    (2, 4, 8, 8, 16, 8, 1),
]

files = glob.glob(config_location + '/*.sh')
for f in files:
    os.remove(f)

for clusters, cores, warps, threads, l2, dcache, icache in configs:
    l2k, dcachek, icachek = 1024 * l2, 1024 * dcache, 1024 * icache
    name_suffix = ''
    with open(config_location + '/' + name_template.format(**locals()), 'w') as f:
        codegen = ''
        f.write(template.format(**locals()))
