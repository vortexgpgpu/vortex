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

"""beacon -- run Beacon, foldsim's logic-level and area count, on a Vortex
netlist, and read its results by RTL unit or against Vivado.

  run           map an elaborated CIRCT netlist and count the logic levels of
                every end point (foldsim-estimate-timing) and its LUTs,
                flip-flops, memories and DSPs (foldsim-estimate-area)
  summarize     the deep end points grouped by RTL unit, optionally beside
                Vivado's failing ones
  paths         every deep path op by op, by RTL unit (one per pair of
                registers, generate copies folded)
  diff          a change against its parent revision: the registers that got
                deeper or shallower, by register and by unit
  validate      agreement per register with a Vivado end point table
  retime-split  the same agreement, split by whether Vivado retimed the paths
  ports         the depth at the ports of chosen modules' instances, by unit
  buffers       which elastic buffers pass a deep ready chain or a computed
                output through: evidence for an OUT_BUF (OUT_REG) change
  calibrate     train the learned calibration (scikit-learn) on DUT builds:
                Beacon runs and the Vivado end point tables of the same sources
  score         rank a run's registers by the calibrated near-critical
                probability, with predicted slack (pure Python; advisory)

  ci/beacon.py run gen/design.mlir runs/full
  ci/beacon.py run runs/full/mapped.mlirbc runs/socket \\
      top=VX_socket__lift black-boxes=VX_core__lift
  ci/beacon.py summarize runs/full --min-levels 12
  ci/beacon.py paths runs/full --unit lsu_unit
  ci/beacon.py diff runs/parent runs/change --min-levels 8
  ci/beacon.py validate runs/socket ep.csv --native VX_socket__lift=dut
  ci/beacon.py run --keep VX_elastic_buffer design.mlir runs/eb \\
      port-modules=VX_elastic_buffer
  ci/beacon.py buffers runs/eb
  ci/beacon.py calibrate cal.json --build build_core/beacon ep.csv core
  ci/beacon.py score build_core/beacon cal.json --top 30

A Vivado end point table is a CSV of `slack,levels,logic,net,start,end`, one
row per failing end point (a gate's ep.csv). The DUT flow and logic_gate run
Beacon through the Makefiles instead (make -C hw/syn/xilinx/dut <dut>-beacon,
ci/logic_gate.py); see docs/designs/beacon.md.
"""
import argparse
import collections
import csv
import json
import math
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

# Vivado appends replication and retiming suffixes in any order
# (_rep__0_replica_1).
VIVADO_CELL = re.compile(
    r'_reg((\[\w+\])*(_replica(_\d+)?|_[fb]ret(_+\d+)?|_rep(_+\d+)?)*'
    r'|_\d+(_\d+)*|_bram_\d+)$')
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
    # The halves of a dual-port LUTRAM (RAM32X1D/RAM64X1D: DP, SP) sit
    # below the memory cell.
    cell_path = re.sub(r'/(DP|SP)(/[^/]+)$', r'\2', cell_path)
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
    # Only the comparisons with Vivado need numpy; counting levels does not.
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


def endpoints(run_dir):
    """The rows of a run's end point table, as dicts."""
    with open(f'{run_dir}/endpoints.csv') as f:
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


def cmd_run(args):
    """Map (unless the input is already mapped bytecode), then count levels;
    from a netlist, also map for area and count it."""
    opt = os.environ.get("FOLDSIM_OPT") or "foldsim-opt"
    os.makedirs(args.out, exist_ok=True)
    # Map to the cells of the target the levels are counted on.
    target = next((o for o in args.options if o.startswith("target=")),
                  "target=versal")
    mapped = args.design
    steps = []
    # MLIR maps modules in parallel: the full GPU maps in 53 s and 18 GB, or
    # 107 s and 9 GB on one thread.
    threads = ["--mlir-disable-threading"] if args.no_threads else []
    if not args.design.endswith(".mlirbc"):
        mapped = os.path.join(args.out, "mapped.mlirbc")
        pipeline = "foldsim-beacon-map{%s}" % target
        if args.keep:
            # Keep these modules' instances through inlining, so their ports
            # can be reported (ports=, port-modules=).
            pipeline = "foldsim-keep-modules{modules=%s},%s" % (
                ",".join(args.keep), pipeline)
        steps.append(("map", [opt, args.design, "--emit-bytecode", "-o", mapped,
                              "--pass-pipeline=builtin.module(%s)" % pipeline]
                      + threads))
    files = ["output=%s/endpoints.csv" % args.out,
             "instances=%s/instances.csv" % args.out,
             "report=%s/report.txt" % args.out,
             "paths=%s/paths.jsonl" % args.out]
    if any(o.startswith("port-modules=") for o in args.options):
        files.append("ports=%s/ports.csv" % args.out)
    opts = " ".join(files + args.options)
    steps.append(("levels", [opt, mapped, "-o", os.devnull,
                             "--pass-pipeline=builtin.module("
                             "foldsim-estimate-timing{%s})" % opts]))
    if not args.design.endswith(".mlirbc"):
        # Area reads the netlist mapped again, for area.
        top = [o for o in args.options if o.startswith(("top=", "black-boxes="))]
        area = " ".join([target, "output=%s/area.csv" % args.out,
                         "memories=%s/area_memories.csv" % args.out,
                         "report=%s/area.txt" % args.out] + top)
        steps.append(("area", [opt, args.design, "-o", os.devnull,
                               "--pass-pipeline=builtin.module("
                               "foldsim-beacon-map{%s area},"
                               "foldsim-estimate-area{%s})" % (target, area)]
                      + threads))
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

    inst = load_instances(f'{args.run_dir}/instances.csv')
    classes = FOLD_CLASSES
    sub = None
    harness = ''
    if args.native:
        module, prefix = args.native.split('=')
        prefix, _, sub = prefix.partition(':')
        # A DUT build's table names its cells from the DUT's top down, with
        # no harness prefix.
        harness = '' if prefix else 'dut/'
        prefix = prefix or 'dut'
        classes = {module: prefix}
    paths = class_paths(inst, classes)

    ours = collections.defaultdict(lambda: [0, 0])  # key -> [max levels, n]
    # (class, instance path) -> {register base name: key}, for Vivado names
    # that place a register in the generate scope that assigns it rather
    # than the one that declares it.
    by_dir = collections.defaultdict(dict)
    # Vendor IPs, black boxes to Beacon: Vivado's end points inside them have
    # no counterpart.
    ips = set()

    def class_key(inst_id, name):
        cls, p = paths[int(inst_id)]
        if cls is None:
            return None
        # Registers of inlined modules are named <instance path>/<name>.
        base = name.rsplit('.', 1)[0]
        full = f'{p}/{base}' if p else base
        if sub:
            if not full.startswith(sub + '/'):
                return None
            full = full[len(sub) + 1:]
        return cls, full

    for row in endpoints(args.run_dir):
        if row['start_kind'] == 'ip':
            ip = class_key(row['start_inst'], row['start'])
            if ip:
                ips.add((ip[0], norm(ip[1])))
        ck = class_key(row['end_inst'], row['end'])
        if ck is None:
            continue
        cls, full = ck
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

    def in_ip(key):
        cls, k = key
        parts = k.split('/')
        return any((cls, '/'.join(parts[:i])) in ips for i in range(1, len(parts)))

    viv = collections.defaultdict(lambda: [0, float('inf'), 0])  # max lv, min slack, n
    rows = unmatched_rows = shell_rows = ip_rows = 0
    unmatched = collections.Counter()
    with open(args.ep) as f:
        r = csv.reader(f)
        next(r)
        for slack, levels, logic, net, start, end in r:
            rows += 1
            key = ep_key(harness + end, args.native)
            if key is None or key[0] not in classes.values():
                shell_rows += 1
                continue
            if in_ip(key):
                ip_rows += 1
                continue
            if key not in ours and key[1].endswith('_rdata'):
                # The read register of a synchronous RAM is the memory's
                # read port in the design.
                key = (key[0], key[1][:-len('_rdata')])
            if (key not in ours and key[1].endswith('_rdata_r')
                    and (key[0], key[1][:-len('_rdata_r')] + '_ram') in ours):
                # So is the registered read of a synchronous LUTRAM.
                key = (key[0], key[1][:-len('_rdata_r')] + '_ram')
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
              f'inside vendor IPs {ip_rows}, unmatched {unmatched_rows}, '
              f'matched 0')
        for (cls, k), n in unmatched.most_common(args.top):
            print(f'  {n:6d} {cls}: {k}')
        return
    a = np.array([ours[k][0] for k in keys])
    lv = np.array([viv[k][0] for k in keys])
    sl = np.array([viv[k][1] for k in keys])
    print(f'ep rows {rows}: fold shell / channels {shell_rows}, '
          f'inside vendor IPs {ip_rows}, unmatched {unmatched_rows}, '
          f'matched {rows - shell_rows - ip_rows - unmatched_rows} '
          f'in {len(keys)} registers/memories')
    print(f'Beacon keys in the classes: {len(ours)}')
    print(f'Spearman(Beacon levels, Vivado levels) = {spearman(a, lv):.3f}')
    print(f'Spearman(Beacon levels, -slack)        = {spearman(a, -sl):.3f}')
    ratio = a[lv > 0] / lv[lv > 0]
    print(f'Beacon/Vivado levels: median {np.median(ratio):.2f}, '
          f'p10 {np.percentile(ratio, 10):.2f}, p90 {np.percentile(ratio, 90):.2f}')

    print('\nVivado levels -> Beacon levels (registers/memories):')
    for v in range(0, int(lv.max()) + 1):
        sel = a[lv == v]
        if len(sel):
            print(f'  {v:3d}: n={len(sel):6d} Beacon median {np.median(sel):5.1f} '
                  f'p10 {np.percentile(sel, 10):5.1f} p90 {np.percentile(sel, 90):5.1f}')

    if args.native:
        # The table lists the end points near or past the period; a measured
        # register fails when its worst slack is negative.
        scr = np.array([v[0] for v in ours.values()])
        fails = np.array([k in viv and viv[k][1] < 0 for k in ours])
        print('\nflag registers at >= T levels (failing: negative slack):')
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
            'rows': rows, 'shell': shell_rows, 'vendor_ip': ip_rows,
            'unmatched': unmatched_rows,
            'registers': len(keys), 'measured': len(ours),
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
    print(f'\ndeepest measured registers with failing Vivado end points:')
    for i in order[:args.top]:
        k = keys[i]
        print(f'  beacon {a[i]:3d} vivado {lv[i]:3d} slack {sl[i]:7.3f} '
              f'n {viv[k][2]:5d} {k[0]}: {k[1]}')



def cmd_summarize(args):

    inst = load_instances(f'{args.run_dir}/instances.csv')
    classes = FOLD_CLASSES
    if args.native:
        module, prefix = args.native.split('=')
        classes = {module: prefix}
    paths = class_paths(inst, classes)

    groups = collections.defaultdict(
        lambda: {'n': 0, 'max': -1, 'hist': collections.Counter(),
                 'worst': None})
    for row in endpoints(args.run_dir):
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
              f'Beacon ranks below it ({len(missed)}):')
        for k in missed[:args.top]:
            print(f'{viv[k][0]:8d} {viv[k][1]:4d}  {k[0]}: {k[1]}')



def cmd_paths(args):
    """Every path of at least --min-levels, by unit, op by op."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', 'hw', 'scripts'))
    import beacon_summary
    count, _ = beacon_summary.write_paths(sys.stdout, args.run_dir,
                                          args.min_levels, args.unit)
    if not count:
        print('(Beacon lists paths from its path-levels= on, 12 by '
              'default)')


def load_ports(run_dir):
    """instance -> {port: (direction, levels)} from a run's ports.csv."""
    ports = collections.defaultdict(dict)
    with open(f'{run_dir}/ports.csv') as f:
        for row in csv.DictReader(f):
            ports[(row['instance'], row['module'])][row['port']] = (
                row['direction'], int(row['levels']))
    return ports


def cmd_ports(args):
    """The deepest instance of each unit, port by port."""
    units = {}
    for (inst, module), ports in load_ports(args.run_dir).items():
        deepest = max(levels for _, levels in ports.values())
        key = (INDEX.sub('_*', norm(inst)), module.rstrip('0123456789').rstrip('_'))
        if key not in units or deepest > units[key][0]:
            units[key] = (deepest, inst, ports)
    for (unit, module), (deepest, inst, ports) in sorted(
            units.items(), key=lambda kv: -kv[1][0])[:args.top]:
        if deepest < args.min_levels:
            break
        print(f'{deepest:3d}  {module}  {unit}')
        for port, (direction, levels) in ports.items():
            print(f'       {direction:>3} {port:24s} {levels:3d}')


BIT_INDEX = re.compile(r'(\[\d+\])+$')


def register_depths(run_dir, metric):
    """register -> (levels, deepest start) over its bits, for a run."""
    inst = load_instances(f'{run_dir}/instances.csv')
    paths = class_paths(inst, FOLD_CLASSES)
    regs = {}
    for row in endpoints(run_dir):
        levels = int(row[metric])
        cls, p = paths[int(row['end_inst'])]
        end = row['end']
        full = f'{p}/{end}' if p else end
        key = (cls or '', BIT_INDEX.sub('', full))
        if key not in regs or levels > regs[key][0]:
            sp = paths[int(row['start_inst'])][1]
            regs[key] = (levels, f'{sp}/{row["start"]}' if sp else row['start'])
    return regs


def cmd_diff(args):
    """A change's effect on depth: every register at or above --min-levels in
    either run whose depth moved, worst first. Exits 1 on a regression: a
    register --fail-delta or more levels deeper, one that reaches --fix-levels
    from below, or one only the change has at --fix-levels. If both runs
    counted area, also the design's resources and the units whose area moved;
    exits 3 if only area regressed: a resource of the design more than
    --area-tolerance larger.

    Against routed Vivado runs of the same DUT's revisions, a register that
    got four or more levels deeper lost slack every time, while a rise of one
    to three is within Vivado's own spread from run to run. The mean depth of
    the deepest registers tracks the change in worst slack; the single deepest
    one does not."""
    before = register_depths(args.before, args.metric)
    after = register_depths(args.after, args.metric)

    def deepest(regs):
        return max((v[0] for v in regs.values()), default=0)

    def watched(regs):
        return sum(1 for v in regs.values() if v[0] >= args.min_levels)

    def top_mean(regs, n=50):
        depths = sorted((v[0] for v in regs.values()), reverse=True)[:n]
        return sum(depths) / len(depths) if depths else 0.0

    area_before, area_after = load_area(args.before), load_area(args.after)
    unit_name = 'stages' if _asic_area(args.after) else 'levels'
    area_head = ''
    if bool(area_before) != bool(area_after):
        print('area: only the %s run counted it; not compared'
              % ('reference' if area_before else 'changed'))
    if area_before and area_after:
        tb, ta = area_before[''], area_after['']
        keys = area_keys(tb)
        area_head = '; ' + ', '.join(
            f'{area_label(k, tb)} {tb[k]:g} -> {ta[k]:g}'
            for k in (keys if 'macros' in tb else ('lut', 'bram', 'dsp')))
    head = (f'deepest path {deepest(before)} -> {deepest(after)} {unit_name}; '
            f'mean of the 50 deepest {top_mean(before):.1f} -> {top_mean(after):.1f}; '
            f'registers at or above {args.min_levels} {unit_name} '
            f'{watched(before)} -> {watched(after)}{area_head}')
    print(head)

    deeper, shallower, new, gone = [], [], [], []
    for key, (lv, start) in after.items():
        if key not in before:
            if lv >= args.min_levels:
                new.append((lv, key, start))
            continue
        old = before[key][0]
        if lv > old and lv >= args.min_levels:
            deeper.append((lv, old, key, start))
        elif lv < old and old >= args.min_levels:
            shallower.append((lv, old, key, start))
    for key, (lv, start) in before.items():
        if key not in after and lv >= args.min_levels:
            gone.append((lv, key, start))

    def name(key):
        return f'{key[0]}: {key[1]}' if key[0] else key[1]

    def show(title, rows, moved):
        if not rows:
            return
        print(f'\n{title} ({len(rows)})')
        for row in rows[:args.top]:
            if moved:
                lv, old, key, start = row
                print(f'{old:4d} -> {lv:<4d} {name(key)}')
            else:
                lv, key, start = row
                print(f'{lv:12d} {name(key)}')
            print(f'{"":>13}from {start}')

    deeper.sort(key=lambda r: (-r[0], r[1] - r[0]))
    shallower.sort(key=lambda r: (-r[1], r[0] - r[1]))
    new.sort(key=lambda r: -r[0])
    gone.sort(key=lambda r: -r[0])
    regress = [r for r in deeper if r[0] - r[1] >= args.fail_delta
               or (r[0] >= args.fix_levels > r[1])]
    regress_keys = {r[2] for r in regress}
    show(f'REGRESSION: {args.fail_delta} or more levels deeper, or now at '
         f'{args.fix_levels}', regress, True)
    show('deeper by less (watch)', [r for r in deeper if r[2] not in regress_keys], True)
    show('new at or above the watch', new, False)
    show('shallower', shallower, True)
    show('no longer present', gone, False)

    # The same movement by unit: the deepest register of each unit.
    units = collections.defaultdict(lambda: [0, 0])
    for i, regs in enumerate((before, after)):
        for (cls, reg), (lv, _) in regs.items():
            u = units[(cls, unit(norm(reg)))]
            u[i] = max(u[i], lv)
    moved = sorted(((b, a, k) for k, (b, a) in units.items()
                    if a != b and max(a, b) >= args.min_levels),
                   key=lambda r: (-max(r[0], r[1]), r[0] - r[1]))
    if moved:
        print(f'\nunits whose deepest register moved ({len(moved)})')
        for b, a, (cls, u) in moved[:args.top]:
            print(f'{b:4d} -> {a:<4d} {cls + ": " if cls else ""}{u}')

    area_regress = []
    if area_before and area_after:
        area_regress = show_area(area_before, area_after, args)

    # A register the reference lacks may be a renamed one, so it fails only at
    # the fix threshold, as a new unit does in logic_gate.
    new_regress = [r for r in new if r[0] >= args.fix_levels]
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'headline': head,
                       'deeper': [{'register': name(k), 'before': old, 'after': lv}
                                  for lv, old, k, _ in regress],
                       'new': [{'register': name(k), 'after': lv}
                               for lv, k, _ in new_regress],
                       'area': [{'resource': k, 'before': b, 'after': a}
                                for k, b, a in area_regress]},
                      fh, indent=2)
            fh.write('\n')
    if regress or new_regress:
        return 1
    return 3 if area_regress else 0


AREA_KEYS = ('lut', 'ff', 'lutram', 'bram', 'uram', 'dsp')
# An ASIC run (target=asap7) counts standard cells in the `lut` column,
# flip-flops, and SRAM macros; it has no distributed or block RAM or DSPs.
# Its macros are shown, not held to the tolerance: a memory that moves into
# a macro adds one and takes its flip-flops out.
ASIC_AREA_KEYS = ('lut', 'ff', 'macros')
ASIC_GATED = ('lut', 'ff')


def area_keys(res):
    return ASIC_AREA_KEYS if 'macros' in res else AREA_KEYS


def area_label(key, res):
    return 'cells' if key == 'lut' and 'macros' in res else key


def load_area(run_dir):
    """The design's resources ('') and each folded unit's own, summed over its
    copies, from a run's area.csv, or None. A unit's own resources leave out
    its children's, so a change shows at the unit it is in. `lut` counts
    distributed RAM too, as Vivado's utilization does."""
    path = f'{run_dir}/area.csv'
    if not os.path.exists(path):
        return None
    units = {}
    with open(path) as fh:
        keys = AREA_KEYS + (('macros',) if _asic_area(run_dir) else ())
        for row in csv.DictReader(fh):
            prefix = 'total_' if not row['path'] else ''
            res = {k: float(row[prefix + k]) for k in keys}
            res['lut'] += res['lutram']
            key = unit(norm(row['path']) + '/x') if row['path'] else ''
            if key in units:
                for k in keys:
                    units[key][k] += res[k]
            else:
                units[key] = res
    return units if '' in units else None


def _asic_area(run_dir):
    """Whether the run's area count is an ASIC one: area.txt's first line
    reads `<top> on <pdk>: N cells, ...` there, and LUTs on an FPGA."""
    try:
        with open(f'{run_dir}/area.txt') as fh:
            return ASIC_AREA_RE.search(fh.readline()) is not None
    except OSError:
        return False


def show_area(before, after, args):
    """The design's resources before and after, and the units whose own LUTs
    (cells), memories or DSPs moved most; the resources of the design that
    grew more than --area-tolerance, as (resource, before, after)."""
    tb, ta = before[''], after['']
    keys = area_keys(tb)
    print('\narea (design)')
    regress = []
    for k in keys:
        b, a = tb[k], ta.get(k, 0.0)
        pct = (a - b) / b * 100 if b else (100.0 if a else 0.0)
        flag = ''
        if pct > 100 * args.area_tolerance and \
                ('macros' not in tb or k in ASIC_GATED):
            flag = '  REGRESSION'
            regress.append((area_label(k, tb), b, a))
        print(f'  {area_label(k, tb):7s} {b:10g} -> {a:<10g} {pct:+6.1f}%{flag}')
    # Units rank by LUTs (cells) moved, a memory or DSP counting as 100, and
    # on an ASIC by flip-flops too: a memory there is either a macro or flops.
    weight = {'lut': 1, 'bram': 100, 'uram': 100, 'dsp': 100, 'macros': 100,
              'ff': 1 if 'macros' in tb else 0}
    moved = []
    zero = dict.fromkeys(keys, 0.0)
    for key in set(before) | set(after):
        if not key:
            continue
        b, a = before.get(key, zero), after.get(key, zero)
        if any(abs(a.get(k, 0.0) - b.get(k, 0.0)) >= 0.5 for k in keys):
            moved.append((sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) * weight.get(k, 0)
                              for k in keys), key, b, a))
    if moved:
        moved.sort(key=lambda r: -r[0])
        print(f'\nunits whose own area moved ({len(moved)}), all copies')
        for _, key, b, a in moved[:args.top]:
            parts = [f'{area_label(k, tb)} {b.get(k, 0.0):g} -> {a.get(k, 0.0):g}'
                     for k in keys if abs(a.get(k, 0.0) - b.get(k, 0.0)) >= 0.5]
            print(f'  {key}: ' + ', '.join(parts))
    return regress


def cmd_buffers(args):
    """VX_elastic_buffer instances whose ready chain or output is deep.

    ready_out is the input from downstream and ready_in the output upstream:
    a buffer that passes ready combinationally drives ready_in at least as
    deep as ready_out arrives, and a skid buffer (SIZE 2) or OUT_BUF cuts the
    chain. An output (data_out, valid_out) above one level is computed, not
    registered: OUT_REG would register it.
    """
    rows = []
    for (inst, module), ports in load_ports(args.run_dir).items():
        if not module.startswith('VX_elastic_buffer'):
            continue
        get = lambda p: ports.get(p, ('', -1))[1]
        chain = get('ready_in')
        through = chain >= 1 and chain >= get('ready_out')
        out = max(get('data_out'), get('valid_out'))
        advice = []
        if through and chain >= args.min_levels:
            advice.append(f'ready chain of {chain} levels passes through')
        if out >= args.min_levels:
            advice.append(f'output computed at {out} levels')
        if advice:
            rows.append((max(chain if through else 0, out), inst, module,
                         get('valid_in'), get('ready_out'), chain, out,
                         '; '.join(advice)))
    units = {}
    for r in rows:
        key = INDEX.sub('_*', norm(r[1]))
        if key not in units or r[0] > units[key][0][0]:
            units[key] = (r, 1 + (units[key][1] if key in units else 0))
        else:
            units[key] = (units[key][0], units[key][1] + 1)
    print(f'elastic buffers at or above {args.min_levels} levels: {len(rows)} '
          f'in {len(units)} units')
    print(f'{"lv":>3} {"n":>4} {"valid_in":>8} {"ready_out":>9} '
          f'{"ready_in":>8} {"out":>4}  unit / advice')
    for key, (r, n) in sorted(units.items(), key=lambda kv: -kv[1][0][0]):
        print(f'{r[0]:3d} {n:4d} {r[3]:8d} {r[4]:9d} {r[5]:8d} {r[6]:4d}  '
              f'{key}\n{"":32}{r[7]}')


RETIMED = re.compile(r'_[fb]ret')


def cmd_retime_split(args):
    np = numpy()
    run_dir, ep, native, metric = args.run_dir, args.ep, args.native, args.metric
    module, prefix = native.split('=')
    inst = load_instances(f'{run_dir}/instances.csv')
    paths = class_paths(inst, {module: prefix})
    ours = collections.defaultdict(int)
    for row in endpoints(run_dir):
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
              f'{spearman(a, b):.3f}, median Beacon/Vivado {ratio:.2f}, '
              f'Beacon-Vivado median {np.median(a - b):+.1f}')


# Learned calibration (calibrate, score): per-register features of a run,
# all from Beacon's own outputs. The p_*/n_* columns come from the deepest
# path in paths.jsonl, for registers at or above the model's path levels;
# below that they are missing, as in training.
CAL_FEATURES = (
    'lev', 'lev_mean', 'log2_bits', 'frac_deep', 'log2_nbits_deep',
    'k_d', 'k_wd', 'k_wctl', 'k_rd', 'k_port',
    's_reg', 's_port', 's_bram', 's_ip', 's_other',
    'log2_sfo', 'log2_sfomax', 'log2_in_fo', 'log2_in_fo_deep', 'cross', 'xmax',
    'hier_depth', 'log2_lca_bits', 'lca_up',
    'p_lut', 'p_carry', 'p_cmp', 'p_lutram', 'p_bram', 'p_dsp',
    'n_carry', 'n_cmp', 'p_maxfo', 'p_sumlog_fo', 'p_nfo64', 'p_ninst', 'p_nsrc',
    'hand',
)
CAL_OPS = ('lut', 'carry', 'cmp', 'lutram', 'bram', 'dsp')
CAL_KINDS = {'d': 'd', 'wd': 'wd', 'wa': 'wctl', 'we': 'wctl', 'wm': 'wctl',
             'ra': 'rd', 're': 'rd'}
CAL_NEAR = 0.3     # ns: the near-critical set the model ranks
CAL_PERIOD = 3.333  # ns: the clock the end point tables' slacks are at
CAL_CENSOR = 1.0   # ns: a gate's ep.csv lists end points under 1 ns only


def log2p(x):
    return math.log2(1 + max(0, x))


def register_features(run_dir, path_levels):
    """(features, by_dir, ips) of a run: register -> {feature: value}, all
    end point bits under one name (a memory's ports included), keyed by its
    normalized path below the run's top; the instance -> {base name: key}
    map and the vendor IP instances, for matching Vivado's names."""
    inst = load_instances(f'{run_dir}/instances.csv')
    memo = {}

    def ipath(ctx):
        if ctx not in memo:
            parent, name, _ = inst[ctx]
            p = '' if ctx == 0 else ipath(parent)
            memo[ctx] = '' if ctx == 0 else (f'{p}/{name}' if p else name)
        return memo[ctx]

    def full(ctx, name):
        p = ipath(int(ctx))
        return f'{p}/{name}' if p else name

    sys.setrecursionlimit(10000)
    regs, ips = {}, set()
    by_dir = collections.defaultdict(dict)
    for row in endpoints(run_dir):
        if row['start_kind'] == 'ip':
            ips.add(norm(full(row['start_inst'], row['start'].rsplit('.', 1)[0])))
        f = full(row['end_inst'], row['end'].rsplit('.', 1)[0])
        key = norm(f)
        lv, fo, sfo = int(row['levels']), int(row.get('in_fanout') or 0), int(row['fanout'])
        r = regs.get(key)
        if r is None:
            r = regs[key] = dict(lev=-1, bits=0, sum=0, sfomax=0, fo=0, xmax=0, lvs=[])
            d, _, b = f.rpartition('/')
            by_dir[norm(d)][norm(b.rsplit('.', 1)[-1])] = key
        r['bits'] += 1
        r['sum'] += lv
        r['lvs'].append(lv)
        r['fo'] = max(r['fo'], fo)
        r['xmax'] = max(r['xmax'], int(row['crossings']))
        if row['start_kind'] == 'reg':
            r['sfomax'] = max(r['sfomax'], sfo)
        if lv > r['lev']:
            r.update(lev=lv, kind=row['kind'], skind=row['start_kind'], sfo=sfo,
                     fo_deep=fo, cross=int(row['crossings']),
                     start=norm(full(row['start_inst'], row['start'])))

    paths = {}
    if os.path.exists(f'{run_dir}/paths.jsonl'):
        with open(f'{run_dir}/paths.jsonl') as f:
            for line in f:
                p = json.loads(line)
                key = norm(full(p['end_inst'], p['end'].rsplit('.', 1)[0]))
                if key not in paths or p['levels'] > paths[key]['levels']:
                    paths[key] = p

    # Bits below each instance: the span of a register and its deepest start.
    under = collections.Counter()
    for k, r in regs.items():
        parts = k.split('/')[:-1]
        for i in range(len(parts) + 1):
            under['/'.join(parts[:i])] += r['bits']

    feats = {}
    for k, r in regs.items():
        lv = r['lev']
        a, c = k.split('/')[:-1], r['start'].split('/')[:-1]
        n = 0
        while n < min(len(a), len(c)) and a[n] == c[n]:
            n += 1
        deep = sum(1 for v in r['lvs'] if v >= lv - 1)
        kind = CAL_KINDS.get(r['kind'], 'port')
        skind = r['skind'] if r['skind'] in ('reg', 'port', 'bram', 'ip') else 'other'
        x = dict(lev=lv, lev_mean=r['sum'] / r['bits'], log2_bits=log2p(r['bits']),
                 frac_deep=deep / r['bits'], log2_nbits_deep=log2p(deep),
                 log2_sfo=log2p(r['sfo']), log2_sfomax=log2p(r['sfomax']),
                 log2_in_fo=log2p(r['fo']), log2_in_fo_deep=log2p(r['fo_deep']),
                 cross=r['cross'], xmax=r['xmax'], hier_depth=len(a),
                 log2_lca_bits=log2p(under['/'.join(a[:n])]), lca_up=len(a) - n)
        for v in ('d', 'wd', 'wctl', 'rd', 'port'):
            x['k_' + v] = float(kind == v)
        for v in ('reg', 'port', 'bram', 'ip', 'other'):
            x['s_' + v] = float(skind == v)
        p = paths.get(k) if lv >= path_levels else None
        if p is None:
            x.update((f, float('nan')) for f in CAL_FEATURES if f.startswith(('p_', 'n_')))
            x['hand'] = lv
        else:
            ops = p['ops']  # [levels, kind, fanout, instance, name, source], end first
            cost, count = collections.Counter(), collections.Counter()
            for i in range(len(ops) - 1):
                op = ops[i][1] if ops[i][1] in CAL_OPS else 'lut'
                cost[op] += ops[i][0] - ops[i + 1][0]
                count[op] += 1
            fos = [o[2] for o in ops[1:-1]]
            x.update((f'p_{op}', cost[op]) for op in CAL_OPS)
            x.update(n_carry=count['carry'], n_cmp=count['cmp'],
                     p_maxfo=log2p(max(fos, default=0)),
                     p_sumlog_fo=sum(log2p(v) for v in fos),
                     p_nfo64=sum(1 for v in fos if v >= 64),
                     p_ninst=len({o[3] for o in ops}),
                     p_nsrc=len({o[5].split(':')[0] for o in ops}))
            # Against Vivado's path delay, a carry level costs about 0.42 of
            # a LUT level and a compare 0.15.
            x['hand'] = lv - 0.58 * cost['carry'] - 0.85 * cost['cmp']
        feats[k] = x
    return feats, by_dir, ips


def vivado_slacks(ep, regs, by_dir, ips):
    """register -> worst slack of a DUT build's end point table (cells named
    from the DUT's top down), matched to a run's registers as validate does."""
    def in_ip(k):
        parts = k.split('/')
        return any('/'.join(parts[:i]) in ips for i in range(1, len(parts)))

    def fallback(k):
        d, _, leaf = k.rpartition('/')
        best = None
        for name, cand in by_dir.get(d, {}).items():
            if (leaf == name or leaf.endswith('_' + name)) and (
                    best is None or len(name) > len(best[0])):
                best = (name, cand)
        return best[1] if best else k

    slack = {}
    with open(ep) as f:
        r = csv.reader(f)
        next(r)
        for s, levels, logic, net, start, end in r:
            key = ep_key('top/' + end, True)
            if key is None or in_ip(key[1]):
                continue
            k = key[1]
            if k not in regs and k.endswith('_rdata'):
                k = k[:-len('_rdata')]
            if k not in regs and k.endswith('_rdata_r') and k[:-len('_rdata_r')] + '_ram' in regs:
                k = k[:-len('_rdata_r')] + '_ram'
            if k not in regs:
                k = fallback(k)
            if k in regs:
                slack[k] = min(slack.get(k, float('inf')), float(s))
    return slack


def average_precision(score, positive):
    """Average precision of a ranking, tied scores taken together."""
    np = numpy()
    order = np.argsort(-np.asarray(score, float), kind='mergesort')
    s, y = np.asarray(score, float)[order], np.asarray(positive, bool)[order]
    tp = np.cumsum(y)
    last = np.r_[s[1:] != s[:-1], True]
    prec, rec = tp[last] / (np.nonzero(last)[0] + 1), tp[last] / max(y.sum(), 1)
    return float(np.sum(prec * np.diff(np.r_[0.0, rec])))


def hgb_json(m, link):
    """A fitted HistGradientBoosting model as plain trees: a node goes left
    when its feature is <= t, or is missing and m is set; a leaf has f -1."""
    trees = []
    for (pred,) in m._predictors:
        n = pred.nodes
        leaf = n['is_leaf'].astype(bool)
        trees.append(dict(
            f=[-1 if lf else int(i) for i, lf in zip(n['feature_idx'], leaf)],
            t=[0 if lf else float(t) for t, lf in zip(n['num_threshold'], leaf)],
            m=[int(v) for v in n['missing_go_to_left']],
            l=[int(v) for v in n['left']], r=[int(v) for v in n['right']],
            v=[float(f'{v:.7g}') if lf else 0 for v, lf in zip(n['value'], leaf)]))
    return dict(link=link, base=float(m._baseline_prediction.ravel()[0]), trees=trees)


def hgb_predict(model, x):
    """Evaluate hgb_json trees on one feature row (a list)."""
    raw = model['base']
    for t in model['trees']:
        i = 0
        f = t['f']
        while f[i] >= 0:
            v = x[f[i]]
            i = t['l'][i] if (t['m'][i] if v != v else v <= t['t'][i]) else t['r'][i]
        raw += t['v'][i]
    return 1 / (1 + math.exp(-raw)) if model['link'] == 'logistic' else raw


def cal_models(X, slack, weight, features):
    """Fit the slack regressor and the near-critical and failing classifiers:
    gradient-boosted trees, never ranking a deeper register (lev, hand) safer."""
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50,
              l2_regularization=1.0, early_stopping=False, random_state=0)
    up = [1 if f in ('lev', 'hand') else 0 for f in features]
    reg = HistGradientBoostingRegressor(monotonic_cst=[-v for v in up], **kw)
    near = HistGradientBoostingClassifier(monotonic_cst=up, **kw)
    fail = HistGradientBoostingClassifier(monotonic_cst=up, **kw)
    return (reg.fit(X, slack, sample_weight=weight),
            near.fit(X, slack < CAL_NEAR, sample_weight=weight),
            fail.fit(X, slack < 0, sample_weight=weight))


def cmd_calibrate(args):
    """Train the calibration on DUT builds: each a Beacon run and the Vivado
    end point table of the same sources. Writes the model as JSON."""
    np = numpy()
    builds = []
    for b in args.build:
        if len(b) not in (2, 3):
            sys.exit('--build RUN_DIR EP_CSV [GROUP]')
        run_dir, ep = b[:2]
        feats, by_dir, ips = register_features(run_dir, args.path_levels)
        slacks = vivado_slacks(ep, feats, by_dir, ips)
        keys = sorted(feats)
        X = np.array([[feats[k][f] for f in CAL_FEATURES] for k in keys], float)
        y = np.array([min(slacks.get(k, CAL_CENSOR), CAL_CENSOR) for k in keys])
        measured = np.array([k in slacks for k in keys])
        builds.append(dict(run=run_dir, group=b[2] if len(b) == 3 else run_dir,
                           X=X, y=y, measured=measured))
        print(f'{run_dir}: {len(keys)} registers, {measured.sum()} in the table, '
              f'{(y < 0).sum()} failing, {(y < CAL_NEAR).sum()} under {CAL_NEAR} ns')

    def fit(train):
        X = np.vstack([b['X'] for b in train])
        y = np.concatenate([b['y'] for b in train])
        # Every build weighs the same, whatever its register count.
        w = np.concatenate([np.full(len(b['y']), len(y) / len(train) / len(b['y']))
                            for b in train])
        return cal_models(X, y, w, CAL_FEATURES)

    if args.holdout:
        print('\nleave one build out (builds of its group left out of training):')
        print(f'{"":40} near-critical AP      Spearman vs -slack')
        print(f'{"build":40} {"levels":>7} {"model":>7}   {"levels":>7} {"model":>7}')
        for b in builds:
            train = [o for o in builds if o['group'] != b['group']]
            if not train:
                continue
            reg, near, _ = fit(train)
            lev, m, y = b['X'][:, 0], b['measured'], b['y']
            ap = [average_precision(s, y < CAL_NEAR) if (y < CAL_NEAR).any() else float('nan')
                  for s in (lev, near.predict_proba(b['X'])[:, 1])]
            rho = [spearman(s[m], -y[m]) for s in (lev, -reg.predict(b['X']))]
            name = '/'.join(os.path.normpath(b['run']).split('/')[-2:])
            print(f'{name[-40:]:40} {ap[0]:7.2f} {ap[1]:7.2f}   {rho[0]:7.2f} {rho[1]:7.2f}')

    import sklearn
    reg, near, fail = fit(builds)
    model = dict(
        features=list(CAL_FEATURES), path_levels=args.path_levels, near=CAL_NEAR,
        censor=CAL_CENSOR, period_ns=CAL_PERIOD, sklearn=sklearn.__version__,
        builds=[dict(run=b['run'], group=b['group'], registers=len(b['y']),
                     failing=int((b['y'] < 0).sum())) for b in builds],
        slack=hgb_json(reg, 'identity'), p_near=hgb_json(near, 'logistic'),
        p_fail=hgb_json(fail, 'logistic'))
    with open(args.model, 'w') as f:
        json.dump(model, f, separators=(',', ':'))
    print(f'\nwrote {args.model} ({os.path.getsize(args.model) // 1024} KB, '
          f'{len(builds)} builds)')


def score_run(run_dir, model):
    """(P(near), predicted slack, P(fail), levels, register) for every register
    of a run, near-critical first; the slack is at the model's period. Also
    whether registers at the model's path depth lack their path."""
    feats, _, _ = register_features(run_dir, model['path_levels'])
    names = model['features']
    pathless = any(math.isnan(x['p_lut']) for x in feats.values()
                   if x['lev'] >= model['path_levels'])
    rows = []
    for k, x in feats.items():
        v = [x[n] for n in names]
        rows.append((hgb_predict(model['p_near'], v), hgb_predict(model['slack'], v),
                     hgb_predict(model['p_fail'], v), x['lev'], k))
    rows.sort(key=lambda r: (-r[0], r[1]))
    return rows, pathless


def cmd_score(args):
    """Rank a run's registers by the calibrated near-critical probability,
    with the predicted worst slack."""
    with open(args.model) as f:
        model = json.load(f)
    rows, pathless = score_run(args.run_dir, model)
    if pathless:
        print(f'warning: registers at or above {model["path_levels"]} levels lack '
              f'a path in paths.jsonl; rerun with path-levels={model["path_levels"]}')
    low = sorted(r[1] for r in rows)[:50]
    print(f'{len(rows)} registers; P(slack < {model["near"]} ns) >= 0.5: '
          f'{sum(r[0] >= 0.5 for r in rows)}; mean of the 50 lowest predicted '
          f'slacks {sum(low) / max(1, len(low)):+.3f} ns (model trained on '
          f'{len(model["builds"])} builds)')
    print(f'{"P(near)":>7} {"slack":>6} {"P(fail)":>7} {"lv":>3}  register')
    for p, s, pf, lv, k in [r for r in rows if r[3] >= args.min_levels][:args.top]:
        print(f'{p:7.2f} {s:+6.2f} {pf:7.2f} {lv:3d}  {k}')
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['register', 'levels', 'p_near', 'slack', 'p_fail'])
            for p, s, pf, lv, k in rows:
                w.writerow([k, lv, f'{p:.4f}', f'{s:.4f}', f'{pf:.4f}'])



# report.txt of an ASIC run: `deepest path N stages: clk2q + path + N x cell
# + setup = P ps (F MHz)`.
PERIOD_RE = re.compile(r'^deepest path (\d+) stages: (\d+) \+ (\d+) \+ \d+ x (\d+) '
                       r'\+ (\d+) = (\d+) ps')
ASIC_AREA_RE = re.compile(r' on \S+: (\d+) cells, (\d+) flip-flops, (\d+) macros')


def cmd_asic_summary(args):
    """One CSV row for an ASIC run (target=$(PDK)): the deepest path's stages,
    the period report.txt converts them to and its Fmax, the per-stage and
    fixed delays of that conversion, and the cells, flip-flops and SRAM
    macros of area.txt."""
    row = {}
    with open(f'{args.run_dir}/report.txt') as fh:
        for line in fh:
            m = PERIOD_RE.match(line)
            if m:
                row.update(max_stages=m[1], period_ps=m[6],
                           fmax_mhz=f'{1e6 / int(m[6]):.1f}',
                           stage_ps=m[4],
                           overhead_ps=int(m[2]) + int(m[3]) + int(m[5]))
                break
    with open(f'{args.run_dir}/area.txt') as fh:
        m = ASIC_AREA_RE.search(fh.readline())
        if m:
            row.update(cells=m[1], ff=m[2], macros=m[3])
    if 'period_ps' not in row or 'ff' not in row:
        sys.exit(f'{args.run_dir}: no period in report.txt or no area in area.txt')
    row['endpoints'] = sum(1 for _ in endpoints(args.run_dir))
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)


# --- harness constant ties ---------------------------------------------------

INT_TYPE = re.compile(r'^i(\d+)$')
SSA = r'%[\w.]+'


def _split_top(text, sep=','):
    """`text` split at `sep` outside <> and ()."""
    out, depth, cur = [], 0, ''
    for ch in text:
        if ch in '<(':
            depth += 1
        elif ch in '>)':
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _type_bits(t):
    t = t.strip()
    m = INT_TYPE.match(t)
    if m:
        return int(m[1])
    if t.startswith('!hw.array<'):
        n, elem = t[len('!hw.array<'):-1].split('x', 1)
        return int(n) * _type_bits(elem)
    if t.startswith('!hw.struct<'):
        return sum(_type_bits(f.split(':', 1)[1])
                   for f in _split_top(t[len('!hw.struct<'):-1]))
    return 1 if t == '!seq.clock' else 0


def netlist_top(run_dir):
    """The run's top module: the design row of area.csv, else the netlist's
    last module."""
    try:
        with open(f'{run_dir}/area.csv') as fh:
            for row in csv.DictReader(fh):
                if not row['path']:
                    return row['module']
    except (OSError, KeyError):
        pass
    return None


def const_ties(netlist, top=None):
    """The instance-input bits `top` ties to constants, as (bits, width, port)
    rows, most first. Constants are followed through concatenation, bitcast
    and struct and array creation, which is how a harness zero-extends or
    ties off a port.

    A harness that ties an input lets Vivado and Yosys fold the logic behind
    it, while Beacon maps module by module and keeps it, so the synthesis
    gates and Beacon then measure different designs. The count moves when
    the harness ties more or fewer bits, and when the RTL widens or narrows
    a port the harness ties: either way the synthesis gates now fold a
    different amount of logic."""
    mods, cur = {}, None
    with open(netlist) as fh:
        for line in fh:
            m = re.match(r'\s*hw\.module (?:private )?@(\S+)\(', line)
            if m:
                cur = m[1]
                mods[cur] = []
            elif cur and re.match(r'\s*} loc', line):
                cur = None
            elif cur:
                mods[cur].append(line)
    if not mods:
        raise ValueError(f'{netlist}: no hw.module')
    top = top or list(mods)[-1]
    if top not in mods:
        raise ValueError(f'{netlist}: no module {top}')

    const, rows = {}, []
    for line in mods[top]:
        line = re.sub(r' loc\(#loc\d*\)\s*$', '', line.strip())
        m = re.match(rf'({SSA}) = hw\.constant (?:true|false)\b', line)
        if m:
            const[m[1]] = 1
            continue
        m = re.match(rf'({SSA}) = hw\.constant \S+ : (\S+)', line)
        if m:
            const[m[1]] = _type_bits(m[2])
            continue
        m = re.match(rf'({SSA}) = comb\.concat (.*) : ', line)
        if m:
            const[m[1]] = sum(const.get(o.strip(), 0) for o in m[2].split(','))
            continue
        m = re.match(rf'({SSA}) = hw\.bitcast ({SSA})', line)
        if m:
            const[m[1]] = const.get(m[2], 0)
            continue
        m = re.match(rf'({SSA}) = hw\.(?:struct|array)_create \((.*)\) :', line)
        if m:
            const[m[1]] = sum(const.get(o.strip(), 0) for o in m[2].split(','))
            continue
        m = re.search(r'hw\.instance "(\S+)" @(\S+)\((.*)\) -> ', line)
        if m:
            for arg in _split_top(m[3]):
                port, rest = arg.split(':', 1)
                val, ty = rest.split(':', 1)
                bits = const.get(val.strip(), 0)
                if bits:
                    rows.append((bits, _type_bits(ty), f'{m[1]}.{port.strip()}'))
    rows.sort(key=lambda r: (-r[0], r[2]))
    return rows


def cmd_ties(args):
    """Instance-input bits the DUT top ties to constants, per port."""
    rows = const_ties(f'{args.run_dir}/netlist.mlir',
                      args.top or netlist_top(args.run_dir))
    for bits, width, port in rows:
        print(f'{bits:8d} of {width:6d}  {port}')
    print(f'{sum(r[0] for r in rows)} instance input bits tied to constants')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="map a netlist and count its levels")
    p.add_argument("design", help="elaborated netlist (.mlir), or mapped bytecode (.mlirbc)")
    p.add_argument("out", help="output directory")
    p.add_argument("options", nargs="*",
                   help="foldsim-estimate-timing options (top=, target=, black-boxes=, "
                   "trace=, retime-levels=, ...); the design is mapped to the "
                   "cells of target= (versal by default)")
    p.add_argument("--no-threads", action="store_true",
                   help="map on one thread: half the peak memory, twice the time")
    p.add_argument("--keep", action="append", default=[], metavar="MODULE",
                   help="keep MODULE's instances through inlining, to report "
                   "their ports with port-modules=MODULE (repeatable)")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("summarize", help="deep end points by RTL unit")
    p.add_argument('run_dir')
    p.add_argument('--min-levels', type=int, default=10)
    p.add_argument('--ep', help="a Vivado end point table to group beside")
    p.add_argument('--native', metavar='MODULE=PREFIX')
    p.add_argument('--top', type=int, default=40)
    p.add_argument('--metric', default='levels', choices=['levels', 'retimed'],
                   help="the run's column to group by")
    p.set_defaults(fn=cmd_summarize)

    p = sub.add_parser("paths", help="deep paths op by op, by RTL unit")
    p.add_argument('run_dir')
    p.add_argument('--min-levels', type=int, default=12)
    p.add_argument('--unit', help="only the units whose name contains this")
    p.set_defaults(fn=cmd_paths)

    p = sub.add_parser("validate", help="agreement per register with Vivado")
    p.add_argument('run_dir')
    p.add_argument('ep')
    p.add_argument('--top', type=int, default=30)
    p.add_argument('--native', metavar='MODULE=PREFIX[:SUBPATH]',
                   help='native gate: the measured top and the harness '
                   'prefix of its cells (e.g. VX_socket__lift=dut); with '
                   'SUBPATH, only the measured end points below it, which '
                   'the harness instantiates as PREFIX; an empty PREFIX (VX_core_top=) for '
                   'a DUT build, whose cells have no harness prefix')
    p.add_argument('--json', help='also write the agreement to this file')
    p.add_argument('--metric', default='levels', choices=['levels', 'retimed'],
                   help="the run's column to compare")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("ports", help="port depths of chosen modules, by unit")
    p.add_argument('run_dir')
    p.add_argument('--min-levels', type=int, default=8)
    p.add_argument('--top', type=int, default=40)
    p.set_defaults(fn=cmd_ports)

    p = sub.add_parser("diff", help="a change's effect on depth, register by "
                       "register, and on area, by unit")
    p.add_argument('before', help="run directory of the reference (the parent revision)")
    p.add_argument('after', help="run directory of the change")
    p.add_argument('--min-levels', type=int, default=8,
                   help="watch registers at or above this depth in either run")
    p.add_argument('--fix-levels', type=int, default=12,
                   help="a register that reaches this depth from below, or "
                   "only the change has at it, fails")
    p.add_argument('--fail-delta', type=int, default=4,
                   help="a register this many levels deeper fails; a smaller "
                   "rise is listed to watch")
    p.add_argument('--top', type=int, default=40)
    p.add_argument('--metric', default='levels', choices=['levels', 'retimed'],
                   help="the runs' column to compare")
    p.add_argument('--area-tolerance', type=float, default=0.05,
                   help="a resource of the design that grows by more than this "
                   "fraction fails (exit 3)")
    p.add_argument('--json', metavar='FILE',
                   help="also write the headline and the failing registers "
                   "and resources here")
    p.set_defaults(fn=cmd_diff)

    p = sub.add_parser("buffers", help="elastic buffers that pass deep paths")
    p.add_argument('run_dir')
    p.add_argument('--min-levels', type=int, default=8)
    p.set_defaults(fn=cmd_buffers)

    p = sub.add_parser("retime-split", help="agreement split by Vivado's retiming")
    p.add_argument('run_dir')
    p.add_argument('ep')
    p.add_argument('native', metavar='MODULE=PREFIX')
    p.add_argument('metric', nargs='?', default='levels', choices=['levels', 'retimed'])
    p.set_defaults(fn=cmd_retime_split)

    p = sub.add_parser("calibrate", help="train the learned slack calibration")
    p.add_argument('model', help="the model JSON to write")
    p.add_argument('--build', nargs='+', action='append', required=True,
                   metavar='RUN_DIR EP_CSV [GROUP]',
                   help="a DUT build: its Beacon run and Vivado end point table; "
                   "builds of one GROUP (one RTL lineage) are held out together "
                   "(repeatable)")
    p.add_argument('--path-levels', type=int, default=8,
                   help="path features from this depth on (the runs' path-levels)")
    p.add_argument('--holdout', action='store_true',
                   help="also print a leave-one-build-out check against levels")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("score", help="registers by calibrated near-critical risk")
    p.add_argument('run_dir')
    p.add_argument('model', help="a model JSON from calibrate")
    p.add_argument('--min-levels', type=int, default=0)
    p.add_argument('--top', type=int, default=40)
    p.add_argument('--csv', help="also write every register's prediction here")
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("asic-summary", help="period and area of an ASIC run, as CSV")
    p.add_argument('run_dir')
    p.add_argument('out')
    p.set_defaults(fn=cmd_asic_summary)

    p = sub.add_parser("ties", help="instance-input bits the DUT top ties to "
                       "constants")
    p.add_argument('run_dir')
    p.add_argument('--top', help="top module (default: the run's design)")
    p.set_defaults(fn=cmd_ties)

    args = ap.parse_args()
    return args.fn(args) or 0


if __name__ == '__main__':
    # Exit codes 1 and 3 are regressions to logic_gate (DEEPER, AREA); a crash
    # must read as a failed run, not as one.
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(4)
