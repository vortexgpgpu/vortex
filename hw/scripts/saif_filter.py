#!/usr/bin/env python3
# Extract an instance subtree from a Verilator master SAIF and re-root it under a
# synthesis top module, so the result can be annotated directly by the hw/syn flows
# (Vivado read_saif, Synopsys read_saif, OpenSTA read_saif).
#
# The master SAIF produced by Verilator (--trace-saif) is rooted at the simulation
# top (e.g. TOP/.../core/execute/tcu_unit/...) and contains the full design.  The
# synthesis DUT is a standalone wrapper (e.g. VX_tcu_unit_top) whose top-level child
# instance matches the extracted scope.  This tool slices the requested subtree
# verbatim -- preserving every nested generate scope and net, regardless of tile size
# or NUM_THREADS -- and wraps it under the named top module.
#
# Usage:
#   saif_filter.py --instance <path> [--top <module>] [-o out.saif] master.saif
#   saif_filter.py --list [--list-depth N] master.saif
#
#   --instance   slash-separated instance path to extract.  Matched as a suffix of the
#                full hierarchy path, so "execute/tcu_unit" or just "tcu_unit" both work.
#   --top        wrapper instance name placed above the extracted subtree (and written
#                as the SAIF DESIGN).  Omit to emit the extracted instance as the root.
#   --all        extract every match instead of only the first (names disambiguated).
#   --list       print the instance hierarchy (for discovering paths) and exit.

import argparse
import re
import sys

INSTANCE_RE = re.compile(r'^(\s*)\(INSTANCE\s+(\S+)\s*$')
HEADER_KEYS = ('DIVIDER', 'TIMESCALE', 'DURATION', 'DIRECTION', 'SAIFVERSION')


def paren_delta(line):
    """Net change in paren depth on a line, ignoring anything inside double quotes."""
    delta = 0
    in_str = False
    for c in line:
        if c == '"':
            in_str = not in_str
        elif in_str:
            continue
        elif c == '(':
            delta += 1
        elif c == ')':
            delta -= 1
    return delta


def iter_instances(lines):
    """Yield (start_idx, path) for every (INSTANCE ...) opener, tracking hierarchy.

    Depth is the running paren balance *before* the line.  An instance opens one paren
    on its own line, so its body lives one level deeper and closes when the balance
    returns to the opener's level.
    """
    balance = 0
    stack = []  # (name, level) for currently-open instances
    for idx, line in enumerate(lines):
        while stack and balance <= stack[-1][1]:
            stack.pop()
        m = INSTANCE_RE.match(line)
        if m:
            name = m.group(2)
            path = [s[0] for s in stack] + [name]
            yield idx, path
            stack.append((name, balance))
        balance += paren_delta(line)


def capture_subtree(lines, start_idx):
    """Return the list of lines forming the instance subtree opened at start_idx."""
    depth = 0
    out = []
    for line in lines[start_idx:]:
        out.append(line)
        depth += paren_delta(line)
        if depth == 0:
            break
    else:
        raise ValueError("unterminated INSTANCE subtree (unbalanced parens)")
    return out


def dedent(block, pad):
    """Strip the leading indentation of the root line from every line, then re-pad."""
    root_indent = len(block[0]) - len(block[0].lstrip())
    out = []
    for line in block:
        stripped = line[root_indent:] if line[:root_indent].isspace() else line.lstrip()
        out.append(pad + stripped if stripped.strip() else stripped)
    return out


def grab_header(lines):
    """Collect header fields to carry over into the emitted SAIF."""
    hdr = {}
    for line in lines:
        s = line.strip()
        if s.startswith('(INSTANCE'):
            break
        for key in HEADER_KEYS:
            if s.startswith('(' + key):
                hdr[key] = s
    return hdr


def emit(out, header, top, subtrees):
    w = out.write
    w('(SAIFILE\n')
    w(header.get('SAIFVERSION', '(SAIFVERSION "2.0")') + '\n')
    w(header.get('DIRECTION', '(DIRECTION "backward")') + '\n')
    if top:
        w('(DESIGN "%s")\n' % top)
    w('(VENDOR "Verilator")\n')
    w('(PROGRAM_NAME "Verilator")\n')
    w(header.get('DIVIDER', '(DIVIDER / )') + '\n')
    w(header.get('TIMESCALE', '(TIMESCALE 1ps)') + '\n')
    w(header.get('DURATION', '(DURATION 0)') + '\n')
    if top:
        w(' (INSTANCE %s\n' % top)
        pad = '  '
    else:
        pad = ' '
    for block in subtrees:
        for line in dedent(block, pad):
            w(line if line.endswith('\n') else line + '\n')
    if top:
        w(' )\n')
    w(')\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('saif', help='master SAIF produced by Verilator --trace-saif')
    ap.add_argument('--instance', help='slash-separated instance path to extract (suffix match)')
    ap.add_argument('--top', help='wrapper top-module/instance name for the extracted subtree')
    ap.add_argument('--all', action='store_true', help='extract every match, not just the first')
    ap.add_argument('-o', '--output', help='output SAIF path (default: stdout)')
    ap.add_argument('--list', action='store_true', help='print the instance hierarchy and exit')
    ap.add_argument('--list-depth', type=int, default=6, help='max depth for --list (default 6)')
    args = ap.parse_args()

    with open(args.saif) as f:
        lines = f.readlines()

    if args.list:
        for idx, path in iter_instances(lines):
            if len(path) <= args.list_depth:
                print('  ' * (len(path) - 1) + path[-1])
        return 0

    if not args.instance:
        ap.error('--instance is required unless --list is given')

    target = [p for p in args.instance.strip('/').split('/') if p]
    n = len(target)
    matches = [idx for idx, path in iter_instances(lines) if path[-n:] == target]
    if not matches:
        sys.stderr.write('ERROR: instance path not found: %s\n' % args.instance)
        return 1
    if not args.all:
        matches = matches[:1]

    subtrees = [capture_subtree(lines, idx) for idx in matches]
    header = grab_header(lines)

    out = open(args.output, 'w') if args.output else sys.stdout
    try:
        emit(out, header, args.top, subtrees)
    finally:
        if args.output:
            out.close()

    fedp = sum(1 for b in subtrees for ln in b if INSTANCE_RE.match(ln) and INSTANCE_RE.match(ln).group(2) == 'fedp')
    sys.stderr.write('saif_filter: extracted %d instance(s) under "%s"; %d fedp leaf instance(s)\n'
                     % (len(subtrees), args.top or target[-1], fedp))
    return 0


if __name__ == '__main__':
    sys.exit(main())
