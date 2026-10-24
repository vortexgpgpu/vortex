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
"""The delays the logic screen's ASIC profile (foldsim-estimate-timing
family=asap7) converts stages with, read from a Liberty library.

usage: liberty_stage.py <lib> [--fanout N] [--suffix _ASAP7_75t_R]

For a representative set of gates, the delay of one stage driving `fanout`
copies of a NAND2's input, at the slew such a stage produces (iterated to a
fixed point), and the flip-flop's clock-to-output delay and setup time at
that slew. Prints one line per cell and the three numbers the asap7 profile
is built with; re-derive them when the library or its corner changes.
"""
import argparse
import bisect
import re
import sys

GATES = ['INVx1', 'NAND2xp5', 'NOR2xp33', 'AOI22xp5', 'OAI21xp5', 'XOR2xp5',
         'MAJIxp5', 'AO21x1']
NAND = 'NAND2xp5'
FLOP = 'DFFHQNx1'


def cell_block(text, name):
    """The text of `cell (name) { ... }`, by brace matching."""
    m = re.search(r'\bcell\s*\(\s*"?%s"?\s*\)\s*\{' % re.escape(name), text)
    if not m:
        sys.exit(f'no cell {name}')
    depth, i = 1, m.end()
    while depth:
        c = text[i]
        depth += c == '{'
        depth -= c == '}'
        i += 1
    return text[m.end():i - 1]


def groups(text, kind):
    """(argument, body) of each `kind (arg) { ... }` at any depth."""
    for m in re.finditer(r'\b%s\s*\(\s*"?([^")]*)"?\s*\)\s*\{' % kind, text):
        depth, i = 1, m.end()
        while depth:
            c = text[i]
            depth += c == '{'
            depth -= c == '}'
            i += 1
        yield m.group(1), text[m.end():i - 1]


def numbers(s):
    return [float(x) for x in re.findall(r'-?[0-9.]+(?:e-?[0-9]+)?', s)]


def table(body, kind):
    """(index_1, index_2, rows) of the first `kind` table in `body`."""
    for _, t in groups(body, kind):
        i1 = numbers(re.search(r'index_1\s*\(\s*"([^"]*)"', t).group(1))
        m2 = re.search(r'index_2\s*\(\s*"([^"]*)"', t)
        i2 = numbers(m2.group(1)) if m2 else [0.0]
        vals = numbers(re.search(r'values\s*\((.*?)\)\s*;', t, re.S).group(1))
        return i1, i2, [vals[k * len(i2):(k + 1) * len(i2)]
                        for k in range(len(i1))]
    return None


def interp(tab, x, y):
    i1, i2, v = tab

    def bracket(idx, q):
        k = min(max(bisect.bisect_left(idx, q), 1), len(idx) - 1)
        lo, hi = idx[k - 1], idx[k]
        return k - 1, k, (q - lo) / (hi - lo) if hi != lo else 0.0

    if len(i2) == 1:
        a, b, t = bracket(i1, x)
        return v[a][0] + t * (v[b][0] - v[a][0])
    a, b, t = bracket(i1, x)
    c, d, u = bracket(i2, y)
    top = v[a][c] + u * (v[a][d] - v[a][c])
    bot = v[b][c] + u * (v[b][d] - v[b][c])
    return top + t * (bot - top)


def pin_cap(cell, pin):
    for name, body in groups(cell, 'pin'):
        if name == pin:
            return float(re.search(r'\bcapacitance\s*:\s*([0-9.]+)', body).group(1))
    sys.exit(f'no pin {pin}')


def arcs(cell, out, related=None):
    """Timing groups of output pin `out`, optionally for one related pin."""
    for name, body in groups(cell, 'pin'):
        if name != out:
            continue
        for _, t in groups(body, 'timing'):
            rel = re.search(r'related_pin\s*:\s*"?(\w+)"?', t).group(1)
            if related is None or rel == related:
                yield rel, t


def gate_delay(cell, out, slew, load):
    """Mean of rise and fall delay and transition over every input arc."""
    d, s, n = 0.0, 0.0, 0
    for _, t in arcs(cell, out):
        if 'timing_type' in t and 'combinational' not in t:
            continue
        for kind, tkind in (('cell_rise', 'rise_transition'),
                            ('cell_fall', 'fall_transition')):
            dt, st = table(t, kind), table(t, tkind)
            if dt and st:
                d += interp(dt, slew, load)
                s += interp(st, slew, load)
                n += 1
    return d / n, s / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lib')
    ap.add_argument('--fanout', type=float, default=3)
    ap.add_argument('--suffix', default='_ASAP7_75t_R')
    args = ap.parse_args()
    text = open(args.lib).read()
    nand = cell_block(text, NAND + args.suffix)
    load = args.fanout * pin_cap(nand, 'A')

    slew = 20.0
    for _ in range(20):
        delays = {}
        out_slews = []
        for g in GATES:
            cell = cell_block(text, g + args.suffix)
            d, s = gate_delay(cell, 'Y', slew, load)
            delays[g] = d
            out_slews.append(s)
        new = sum(out_slews) / len(out_slews)
        if abs(new - slew) < 0.01:
            break
        slew = new

    flop = cell_block(text, FLOP + args.suffix)
    clk2q = sum(interp(table(t, k), slew, load)
                for _, t in arcs(flop, 'QN', 'CLK')
                for k in ('cell_rise', 'cell_fall')) / 2
    setup = 0.0
    for _, t in arcs(flop, 'D', 'CLK'):
        if 'setup_rising' in t:
            setup = (interp(table(t, 'rise_constraint'), slew, slew) +
                     interp(table(t, 'fall_constraint'), slew, slew)) / 2
    for g in GATES:
        print(f'{g:10s} {delays[g]:6.1f} ps')
    stage = sum(delays.values()) / len(delays)
    print(f'load {load:.2f} fF (fanout {args.fanout:g}), slew {slew:.1f} ps')
    print(f'stage {stage:.1f} ps, clk->q {clk2q:.1f} ps, setup {setup:.1f} ps')


if __name__ == '__main__':
    main()
