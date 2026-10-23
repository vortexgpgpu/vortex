#!/usr/bin/env python3
# Copyright © 2019-2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""timing_screen -- run foldsim's timing screen on a Vortex netlist, and read
its results by RTL unit or against Vivado.

  screen        map an elaborated CIRCT netlist and count the logic levels of
                every end point (foldsim-timing-map, foldsim-estimate-timing)
  summarize     the deep end points grouped by RTL unit, optionally beside
                Vivado's failing ones
  validate      agreement per register with a Vivado end point table
  retime-split  the same agreement, split by whether Vivado retimed the paths

  ci/timing_screen.py screen gen/design.mlir runs/full
  ci/timing_screen.py screen runs/full/mapped.mlirbc runs/socket \\
      top=VX_socket__lift black-boxes=VX_core__lift
  ci/timing_screen.py summarize runs/full --min-levels 12
  ci/timing_screen.py validate runs/socket ep.csv --native VX_socket__lift=dut

A Vivado end point table is a CSV of `slack,levels,logic,net,start,end`, one
row per failing end point (a gate's ep.csv). The DUT flow and the timing gate
screen with the Makefiles instead (make -C hw/syn/xilinx/dut <dut>-screen,
ci/timing_gate.py); see docs/designs/timing_screen.md.
"""
import argparse
import collections
import csv
import os
import re
import resource
import subprocess
import sys
import time

# Fold classes of the static fold: module in the design -> cell prefix in
# the folded netlist.
FOLD_CLASSES = {
    'VX_socket__lift': 'VX_socket__lift_model_copy',
    'VX_core__lift': 'VX_core__lift_model_copy',
    'VX_compute_slice__lift': 'VX_compute_slice__lift_copy',
    'foldsim_target': 'foldsim_target_model_copy',
}

VIVADO_CELL = re.compile(
    r'_reg((\[\d+\])*(_replica(_\d+)?)?(_[fb]ret(_\d+)?)*(_rep(_\d+)?)?'
    r'|_\d+_\d+_\d+_\d+|_bram_\d+)$')
FOLD_SUFFIX = re.compile(
    r'(_ext(_\d+)?|_fold|_read(_\d+)?|_[fb]ret(_\d+)?|_replica(_\d+)?)+$')
FOLD_SHELL = re.compile(r'^fold\d+_')
DSP_SUBCELL = re.compile(r'^DSP\w*(<\d+>)?_INST$')


def norm(s):
    return s.replace('.', '_').replace('[', '_').replace(']', '')


def ep_key(cell_path, native):
    """(class, key) of a Vivado end point pin, or None for fold-shell cells.

    Folded: sim/<class>/<path with 2F for />[/<primitive>]/<pin>.
    Native: <harness>/<path>/<cell>[/<primitive>]/<pin>, <class> being the
    harness prefix.
    """
    parts = cell_path.split('/')
    if native:
        cls = parts[0]
        hier = parts[1:-1]
        # Distributed RAM primitives below the memory cell, and the sub-cells
        # of a DSP macro (DSP58C<0>_INST/DSP_M_DATA58_INST,
        # DSP_OUTPUT_INST) below the register it absorbed.
        for i, p in enumerate(hier):
            if (p.startswith('Memory_reg') or re.match(r'RAM[A-H]', p)
                    or DSP_SUBCELL.match(p)):
                hier = hier[:i]
                break
    else:
        if parts[0] != 'sim' or len(parts) < 3:
            return None
        cls = parts[1]
        hier = parts[2].split('2F')
    if not hier or FOLD_SHELL.match(hier[-1]):
        return None
    leaf = VIVADO_CELL.sub('', hier[-1])
    leaf = FOLD_SUFFIX.sub('', leaf)
    return cls, norm('/'.join(hier[:-1] + [leaf]))


def numpy():
    # Only the comparisons with Vivado need numpy; screening does not.
    import numpy as np
    return np


def spearman(a, b):
    np = numpy()

    def rank(x):
        x = np.asarray(x, dtype=float)
        order = np.argsort(x, kind='mergesort')
        ranks = np.empty(len(x))
        ranks[order] = np.arange(len(x))
        # Average the ranks of ties.
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=ranks)
        return (sums / counts)[inv]
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def endpoints(screen):
    """The rows of a screen's end point table, as dicts."""
    with open(f'{screen}/endpoints.csv') as f:
        yield from csv.DictReader(f)


def load_instances(path):
    inst = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            inst[int(row['id'])] = (int(row['parent']), row['instance'],
                                    row['module'])
    return inst


def class_paths(inst, classes):
    """ctx -> (class cell prefix, path below the class root)."""
    memo = {}

    def get(ctx):
        if ctx in memo:
            return memo[ctx]
        parent, name, module = inst[ctx]
        if module in classes:
            res = (classes[module], '')
        elif ctx == 0:
            res = (None, '')
        else:
            cls, p = get(parent)
            res = (cls, f'{p}/{name}' if p else name)
        memo[ctx] = res
        return res

    sys.setrecursionlimit(10000)
    return {ctx: get(ctx) for ctx in inst}


INDEX = re.compile(r'_\d+(?=_|/|$)')


def unit(key):
    d = key.rpartition('/')[0]
    return INDEX.sub('_*', d)


def cmd_screen(args):
    """Map (unless the input is already mapped bytecode), then screen."""
    opt = os.environ.get("FOLDSIM_OPT") or "foldsim-opt"
    os.makedirs(args.out, exist_ok=True)
    asic = any(o == "family=asap7" for o in args.options)
    mapped = args.design
    steps = []
    if not args.design.endswith(".mlirbc"):
        mapped = os.path.join(args.out, "mapped.mlirbc")
        pipeline = "foldsim-timing-map-asic" if asic else "foldsim-timing-map"
        steps.append(("map", [opt, args.design, "--emit-bytecode", "-o", mapped,
                              "--pass-pipeline=builtin.module(%s)" % pipeline]))
    opts = " ".join(["output=%s/endpoints.csv" % args.out,
                     "instances=%s/instances.csv" % args.out,
                     "report=%s/report.txt" % args.out] + args.options)
    steps.append(("screen", [opt, mapped, "-o", os.devnull,
                             "--pass-pipeline=builtin.module("
                             "foldsim-estimate-timing{%s})" % opts]))
    for name, cmd in steps:
        start = time.time()
        rc = subprocess.call(cmd)
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
        print("%-7s %6.1f s  peak %.1f GB" % (name, time.time() - start, peak))
        if rc:
            return rc
    return 0


def cmd_validate(args):
    np = numpy()

    inst = load_instances(f'{args.screen}/instances.csv')
    classes = FOLD_CLASSES
    sub = None
    if args.native:
        module, prefix = args.native.split('=')
        prefix, _, sub = prefix.partition(':')
        classes = {module: prefix}
    paths = class_paths(inst, classes)

    ours = collections.defaultdict(lambda: [0, 0])  # key -> [max levels, n]
    # (class, instance path) -> {register base name: key}, for Vivado names
    # that place a register in the generate scope that assigns it rather
    # than the one that declares it.
    by_dir = collections.defaultdict(dict)
    for row in endpoints(args.screen):
        cls, p = paths[int(row['end_inst'])]
        if cls is None:
            continue
        # Registers of inlined modules are named <instance path>/<name>.
        base = row['end'].rsplit('.', 1)[0]
        full = f'{p}/{base}' if p else base
        if sub:
            if not full.startswith(sub + '/'):
                continue
            full = full[len(sub) + 1:]
        key = (cls, norm(full))
        v = ours[key]
        if not v[1]:
            d, _, base = full.rpartition('/')
            by_dir[(cls, norm(d))][norm(base.rsplit('.', 1)[-1])] = key
        v[0] = max(v[0], int(row[args.metric]))
        v[1] += 1

    def fallback(key):
        cls, k = key
        d, _, leaf = k.rpartition('/')
        best = None
        for name, cand in by_dir.get((cls, d), {}).items():
            if (leaf == name or leaf.endswith('_' + name)) and (
                    best is None or len(name) > len(best[0])):
                best = (name, cand)
        return best[1] if best else key

    viv = collections.defaultdict(lambda: [0, 0.0, 0])  # max lv, min slack, n
    rows = unmatched_rows = shell_rows = 0
    unmatched = collections.Counter()
    with open(args.ep) as f:
        r = csv.reader(f)
        next(r)
        for slack, levels, logic, net, start, end in r:
            rows += 1
            key = ep_key(end, args.native)
            if key is None or key[0] not in classes.values():
                shell_rows += 1
                continue
            if key not in ours and key[1].endswith('_rdata'):
                # The read register of a synchronous RAM is the memory's
                # read port in the design.
                key = (key[0], key[1][:-len('_rdata')])
            if key not in ours:
                key = fallback(key)
            if key not in ours:
                unmatched_rows += 1
                unmatched[key] += 1
                continue
            v = viv[key]
            v[0] = max(v[0], int(levels))
            v[1] = min(v[1], float(slack))
            v[2] += 1

    keys = sorted(viv)
    if not keys:
        print(f'ep rows {rows}: fold shell / channels {shell_rows}, '
              f'unmatched {unmatched_rows}, matched 0')
        for (cls, k), n in unmatched.most_common(args.top):
            print(f'  {n:6d} {cls}: {k}')
        return
    a = np.array([ours[k][0] for k in keys])
    lv = np.array([viv[k][0] for k in keys])
    sl = np.array([viv[k][1] for k in keys])
    print(f'ep rows {rows}: fold shell / channels {shell_rows}, '
          f'unmatched {unmatched_rows}, matched {rows - shell_rows - unmatched_rows} '
          f'in {len(keys)} registers/memories')
    print(f'screen keys in the classes: {len(ours)}')
    print(f'Spearman(screen levels, Vivado levels) = {spearman(a, lv):.3f}')
    print(f'Spearman(screen levels, -slack)        = {spearman(a, -sl):.3f}')
    ratio = a[lv > 0] / lv[lv > 0]
    print(f'screen/Vivado levels: median {np.median(ratio):.2f}, '
          f'p10 {np.percentile(ratio, 10):.2f}, p90 {np.percentile(ratio, 90):.2f}')

    print('\nVivado levels -> screen levels (registers/memories):')
    for v in range(0, int(lv.max()) + 1):
        sel = a[lv == v]
        if len(sel):
            print(f'  {v:3d}: n={len(sel):6d} screen median {np.median(sel):5.1f} '
                  f'p10 {np.percentile(sel, 10):5.1f} p90 {np.percentile(sel, 90):5.1f}')

    if args.native:
        # The ep.csv of a gate lists every failing end point, so a screened
        # register without one meets timing.
        scr = np.array([v[0] for v in ours.values()])
        fails = np.array([k in viv for k in ours])
        print('\nflag registers at >= T levels (gate lists all failing end '
              'points):')
        print('     T  flagged  precision  recall')
        for t in range(4, 21, 2):
            flagged = scr >= t
            if not flagged.any():
                break
            prec = fails[flagged].mean()
            rec = flagged[fails].mean()
            print(f'  {t:4d} {flagged.sum():8d} {prec:10.2f} {rec:7.2f}')

    if args.json:
        summary = {
            'rows': rows, 'shell': shell_rows, 'unmatched': unmatched_rows,
            'registers': len(keys), 'screened': len(ours),
            'rho_levels': round(spearman(a, lv), 3),
            'rho_slack': round(spearman(a, -sl), 3),
            'ratio': round(float(np.median(ratio)), 2),
            'pairs': [[int(x), int(y), n] for (x, y), n in
                      sorted(collections.Counter(zip(a, lv)).items())],
        }
        if args.native:
            summary['pr'] = [
                [t, int((scr >= t).sum()), round(float(fails[scr >= t].mean()), 3),
                 round(float((scr >= t)[fails].mean()), 3)]
                for t in range(2, 23) if (scr >= t).any()]
        with open(args.json, 'w') as f:
            json.dump(summary, f)

    print(f'\nunmatched Vivado cells (top {args.top} by rows):')
    for (cls, k), n in unmatched.most_common(args.top):
        print(f'  {n:6d} {cls}: {k}')

    order = np.argsort(-a, kind='mergesort')
    print(f'\ndeepest screened registers with failing Vivado end points:')
    for i in order[:args.top]:
        k = keys[i]
        print(f'  screen {a[i]:3d} vivado {lv[i]:3d} slack {sl[i]:7.3f} '
              f'n {viv[k][2]:5d} {k[0]}: {k[1]}')



def cmd_summarize(args):

    inst = load_instances(f'{args.screen}/instances.csv')
    classes = FOLD_CLASSES
    if args.native:
        module, prefix = args.native.split('=')
        classes = {module: prefix}
    paths = class_paths(inst, classes)

    groups = collections.defaultdict(
        lambda: {'n': 0, 'max': -1, 'hist': collections.Counter(),
                 'worst': None})
    for row in endpoints(args.screen):
        levels = int(row[args.metric])
        if levels < args.min_levels:
            continue
        cls, p = paths[int(row['end_inst'])]
        end, start = row['end'], row['start']
        full = f'{p}/{end}' if p else end
        g = groups[(cls, unit(norm(full)))]
        g['n'] += 1
        g['hist'][levels] += 1
        if levels > g['max']:
            sp = paths[int(row['start_inst'])][1]
            g['max'] = levels
            g['worst'] = (f'{sp}/{start}' if sp else start, full, row['fanout'])

    viv = collections.defaultdict(lambda: [0, 0])  # n, max levels
    if args.ep:
        with open(args.ep) as f:
            r = csv.reader(f)
            next(r)
            for slack, levels, logic, net, start, end in r:
                if int(levels) < args.min_levels:
                    continue
                key = ep_key(end, args.native)
                if key is None:
                    continue
                v = viv[(key[0], unit(key[1]))]
                v[0] += 1
                v[1] = max(v[1], int(levels))

    order = sorted(groups, key=lambda k: (-groups[k]['max'], -groups[k]['n']))
    print(f'end point bits of at least {args.min_levels} levels, by unit '
          f'({sum(g["n"] for g in groups.values())} in {len(groups)} units)')
    hdr = f'{"max":>4} {"mode":>4} {"bits":>8}'
    if args.ep:
        hdr += f' {"viv n":>7} {"viv max":>7}'
    print(hdr + '  unit')
    for k in order[:args.top]:
        g = groups[k]
        mode = g['hist'].most_common(1)[0][0]
        line = f'{g["max"]:4d} {mode:4d} {g["n"]:8d}'
        if args.ep:
            v = viv.get(k, [0, 0])
            line += f' {v[0]:7d} {v[1]:7d}'
        print(f'{line}  {k[0] or ""}: {k[1]}')
        s, e, fo = g['worst']
        print(f'{"":>20}from {s}\n{"":>20}to   {e}')
    if args.ep:
        missed = sorted((k for k in viv if k not in groups),
                        key=lambda k: -viv[k][0])
        print(f'\nVivado units at or above {args.min_levels} levels the '
              f'screen ranks below it ({len(missed)}):')
        for k in missed[:args.top]:
            print(f'{viv[k][0]:8d} {viv[k][1]:4d}  {k[0]}: {k[1]}')



RETIMED = re.compile(r'_[fb]ret')


def cmd_retime_split(args):
    np = numpy()
    screen, ep, native, metric = args.screen, args.ep, args.native, args.metric
    module, prefix = native.split('=')
    inst = load_instances(f'{screen}/instances.csv')
    paths = class_paths(inst, {module: prefix})
    ours = collections.defaultdict(int)
    for row in endpoints(screen):
        cls, p = paths[int(row['end_inst'])]
        base = row['end'].rsplit('.', 1)[0]
        key = (cls, norm(f'{p}/{base}' if p else base))
        ours[key] = max(ours[key], int(row[metric]))
    viv = collections.defaultdict(lambda: [0, False])
    with open(ep) as f:
        r = csv.reader(f)
        next(r)
        for slack, levels, logic, net, start, end in r:
            key = ep_key(end, True)
            if key not in ours:
                continue
            v = viv[key]
            v[0] = max(v[0], int(levels))
            v[1] |= bool(RETIMED.search(start) or RETIMED.search(end))
    for retimed in (False, True):
        keys = [k for k, v in viv.items() if v[1] == retimed]
        a = np.array([ours[k] for k in keys])
        b = np.array([viv[k][0] for k in keys])
        if len(keys) < 3:
            continue
        ratio = np.median(a[b > 0] / b[b > 0]) if (b > 0).any() else float('nan')
        print(f'retimed={retimed!s:5}: {len(keys):5d} registers, Spearman '
              f'{spearman(a, b):.3f}, median screen/Vivado {ratio:.2f}, '
              f'screen-Vivado median {np.median(a - b):+.1f}')



def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("screen", help="map and screen a netlist")
    p.add_argument("design", help="elaborated netlist (.mlir), or mapped bytecode (.mlirbc)")
    p.add_argument("out", help="output directory")
    p.add_argument("options", nargs="*",
                   help="foldsim-estimate-timing options (top=, family=, black-boxes=, "
                   "trace=, retime-levels=, ...); family=asap7 maps for the ASIC library")
    p.set_defaults(fn=cmd_screen)

    p = sub.add_parser("summarize", help="deep end points by RTL unit")
    p.add_argument('screen')
    p.add_argument('--min-levels', type=int, default=10)
    p.add_argument('--ep', help="a Vivado end point table to group beside")
    p.add_argument('--native', metavar='MODULE=PREFIX')
    p.add_argument('--top', type=int, default=40)
    p.add_argument('--metric', default='levels', choices=['levels', 'retimed'],
                   help="the screen's column to group by")
    p.set_defaults(fn=cmd_summarize)

    p = sub.add_parser("validate", help="agreement per register with Vivado")
    p.add_argument('screen')
    p.add_argument('ep')
    p.add_argument('--top', type=int, default=30)
    p.add_argument('--native', metavar='MODULE=PREFIX[:SUBPATH]',
                   help='native gate: the screened top and the harness '
                   'prefix of its cells (e.g. VX_socket__lift=dut); with '
                   'SUBPATH, only the screened end points below it, which '
                   'the harness instantiates as PREFIX')
    p.add_argument('--json', help='also write the agreement to this file')
    p.add_argument('--metric', default='levels', choices=['levels', 'retimed'],
                   help="the screen's column to compare")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("retime-split", help="agreement split by Vivado's retiming")
    p.add_argument('screen')
    p.add_argument('ep')
    p.add_argument('native', metavar='MODULE=PREFIX')
    p.add_argument('metric', nargs='?', default='levels', choices=['levels', 'retimed'])
    p.set_defaults(fn=cmd_retime_split)

    args = ap.parse_args()
    return args.fn(args) or 0


if __name__ == '__main__':
    sys.exit(main())
