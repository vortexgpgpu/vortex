#!/usr/bin/env python3
"""Compile a target to a maxsim simulator with CIRCT.

usage: maxsim.py --srcs <srcs.f> --top <top> --out <dir> [--circt-bin <dir>]
                 <class> [<class> ...]

The target is imported with circt-verilog and flattened, but for the classes:
the modules simulated as models of their own, each class's instances threaded
through one copy (FAME-5). A class may contain others. Each module with class
instances, innermost first, is cut into a netlist of its own logic and the
models (maxsim-extract-models); the target's netlist is then composed into the
simulator (maxsim-compose): one threaded module per class, the rest of the
target as a model of one thread, and a fabric of token channels between them.

Writes <out>/maxsim_sim.sv, the module `maxsim_sim` with a channel per target
port (`<port>_valid`, `_ready`, `_bits`) and a `profile_finish` bit per class;
<out>/maxsim_sim.json, its port list and the classes' thread counts, in the
order of those bits; and <out>/maxsim_sim.vh, their number for the kernel.
"""
import argparse
import json
import os
import re
import subprocess
import sys

# Passes each model gets before it is threaded: register constants, shared
# adders and narrowed low bits matter for area once every register of a model
# is replicated per thread.
OPTIMIZE = ['hw.module(seq-reg-const-prop)', 'hw.module(comb-share-adders)', 'canonicalize',
            'hw.module(comb-low-bits-narrowing)', 'canonicalize', 'cse']


def run(cmd, **kw):
    print('+', ' '.join(cmd), file=sys.stderr)
    return subprocess.run(cmd, check=True, text=True, **kw)


def publish(src, public):
    """The design with only the `public` modules public."""
    s = re.sub(r'hw\.module @', 'hw.module private @', src)
    for m in public:
        s, n = re.subn(r'hw\.module private @' + re.escape(m) + r'\(', f'hw.module @{m}(', s, count=1)
        if not n:
            sys.exit(f'no module {m} in the design')
    return s


def instantiated(text, module, names):
    """The modules of `names` the module `module` instantiates."""
    m = re.search(r'hw\.module (?:private )?@' + re.escape(module) + r'\(.*?\n  \}\n', text, re.S)
    return {n for n in re.findall(r'hw\.instance "[^"]*" @(\w+)\(', m.group(0)) if n in names}


def signature(text, module):
    """A module's ports: [(direction, name, width)]."""
    m = re.search(r'hw\.module (?:private )?@' + re.escape(module) + r'\((.*?)\) (?:attributes .*?)?\{',
                  text, re.S)
    ports = []
    for d, n, t in re.findall(r'(in|out) %?(\w+) : ([^,)]+)', m.group(1)):
        w = re.fullmatch(r'i(\d+)', t.strip())
        ports.append((d, n, int(w.group(1)) if w else 0))
    return ports


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--srcs', required=True, help='source list: +define+, +incdir+ and files')
    ap.add_argument('--top', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--circt-bin', default='')
    ap.add_argument('classes', nargs='+')
    a = ap.parse_args()
    tool = lambda t: os.path.join(a.circt_bin, t) if a.circt_bin else t
    os.makedirs(a.out, exist_ok=True)
    out = lambda f: os.path.join(a.out, f)

    defines, incdirs, files = [], [], []
    for line in open(a.srcs).read().split('\n'):
        if line.startswith('+define+'):
            defines.append('-D' + line[len('+define+'):])
        elif line.startswith('+incdir+'):
            incdirs.append('-I' + line[len('+incdir+'):])
        elif line.strip():
            files.append(line.strip())
    design = out('design.mlir')
    run([tool('circt-verilog'), '--ir-hw', f'--top={a.top}', '--allow-use-before-declare',
         '--allow-hierarchical-const', '--ignore-uninstantiated-modules', '--detect-memories',
         '--preserve-unpacked-ports', *defines, *incdirs, *files, '-o', design])

    # Flat but for the classes.
    flat = out('flat.mlir')
    with open(flat + '.in', 'w') as f:
        f.write(publish(open(design).read(), [a.top] + a.classes))
    run([tool('circt-opt'), flat + '.in', '--symbol-dce', '--hw-flatten-modules', '--symbol-dce',
         '--canonicalize', '--cse', '-o', flat])
    os.remove(flat + '.in')
    text = open(flat).read()

    # Each module with class instances, innermost first, becomes a netlist.
    classes = set(a.classes)
    children = {m: instantiated(text, m, classes) for m in [a.top] + a.classes}
    order, seen = [], set()

    def visit(m):
        if m in seen:
            return
        seen.add(m)
        for c in sorted(children[m]):
            visit(c)
        if children[m]:
            order.append(m)
    visit(a.top)
    unused = classes - seen
    if unused:
        sys.exit(f'classes not in {a.top}: {" ".join(sorted(unused))}')
    passes = [f'maxsim-extract-models{{module={m} models={",".join(sorted(children[m]))} netlist}}'
              for m in order]
    pipeline = passes + OPTIMIZE + [f'maxsim-compose{{module={a.top} sim=maxsim_sim}}',
                                    'canonicalize', 'cse']
    sim = out('maxsim_sim.mlir')
    run([tool('circt-opt'), flat, f'--pass-pipeline=builtin.module({",".join(pipeline)})', '-o', sim])
    sim_text = open(sim).read()
    run([tool('firtool'), '--format=mlir', sim, '-o', out('maxsim_sim.sv')])

    threads = re.search(r'maxsim\.threads = \[(.*?)\]', sim_text).group(1)
    info = {
        'top': a.top,
        'threads': dict(t.strip('"').split('=') for t in re.findall(r'"[^"]*"', threads)),
        'ports': [{'dir': d, 'name': n, 'width': w} for d, n, w in signature(sim_text, 'maxsim_sim')],
    }
    for name, n in info['threads'].items():
        info['threads'][name] = int(n)
    with open(out('maxsim_sim.json'), 'w') as f:
        json.dump(info, f, indent=1)
    # The kernel's view of it: how many classes report their finishes, bit c
    # of `profile_finish` for the c-th class of the JSON's `threads`.
    with open(out('maxsim_sim.vh'), 'w') as f:
        f.write('// Generated by maxsim.py. Do not edit.\n'
                f'`define MAXSIM_NUM_CLASSES {len(info["threads"])}\n')
    print(f'maxsim: {", ".join(f"{k} x{v}" for k, v in info["threads"].items())}', file=sys.stderr)


if __name__ == '__main__':
    main()
