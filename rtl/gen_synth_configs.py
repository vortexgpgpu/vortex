#!/usr/bin/env python3
import os
import glob

config_location = 'configs'

name_template = '{clusters}cl-{cores}c-{warps}w-{threads}t-{l2}Kl2-{dcache}Kd-{icache}Ki{name_suffix}.v'

template = """
`ifndef VX_DEFINE_SYNTH
`define VX_DEFINE_SYNTH

`define NT {threads}
`define NW {warps}
`define NUMBER_CORES_PER_CLUSTER {cores}
`define NUMBER_CLUSTERS {clusters}
`define DCACHE_SIZE_BYTES {dcachek}
`define ICACHE_SIZE_BYTES {icachek}

// L2 Cache size
`define LLCACHE_SIZE_BYTES {l2k}

{codegen}

`endif
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

files = glob.glob(config_location + '/*')
for f in files:
    os.remove(f)

for clusters, cores, warps, threads, l2, dcache, icache in configs:
    l2k, dcachek, icachek = 1024 * l2, 1024 * dcache, 1024 * icache
    for force_mlab in [False]:
        name_suffix = ''
        if force_mlab:
            name_suffix += '-mlab'
        with open(config_location + '/' + name_template.format(**locals()), 'w') as f:
            codegen = ''
            if force_mlab:
                codegen += '\n`define QUEUE_FORCE_MLAB 1'
            else:
                codegen += '\n// `define QUEUE_FORCE_MLAB 1'

            codegen += '\n\n// Use l3 cache (required for cluster behavior)'

            f.write(template.format(**locals()))
