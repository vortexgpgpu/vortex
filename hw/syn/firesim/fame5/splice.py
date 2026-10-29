#!/usr/bin/env python3
"""Replace placeholder extmodules in a Chisel-emitted .fir with FIRRTL modules.

usage: splice.py <circuit.fir> <module.fir>...

Each <module.fir> is a Yosys circuit whose main module is M; all its modules
are inserted. Chisel
emits one extmodule per BlackBox instance (M, M_1, M_2, ...), all with
`defname = M`. Every one of them is removed, the first is replaced by M's
body, and every `inst ... of <placeholder>` is pointed at M, so that the
wrappers around them become identical and dedup to a single module.
"""
import re
import sys


def module_of(path):
    """(main module name, every module of the file) -- the main one replaces the placeholders."""
    lines = open(path).read().splitlines()
    name = next(re.match(r"circuit (\S+?)\s*:", l).group(1) for l in lines if l.startswith("circuit "))
    start = next(i for i, l in enumerate(lines) if re.match(r"  (ext)?module \S+\s*:", l))
    return name, lines[start:]


def stanzas(circuit):
    """Split into (header, [(kind, name, lines)])."""
    head, out, cur = [], [], None
    for l in circuit:
        m = re.match(r"  (extmodule|module) (\S+?)\s*:", l)
        if m:
            cur = (m.group(1), m.group(2), [l])
            out.append(cur)
        elif cur is None:
            head.append(l)
        else:
            cur[2].append(l)
    return head, out


def splice(circuit, name, body):
    head, mods = stanzas(circuit)
    placeholders = {
        n for k, n, ls in mods
        if k == "extmodule" and any(re.match(rf"\s+defname = {re.escape(name)}\s*$", l) for l in ls)
    }
    if not placeholders:
        sys.exit(f"splice: no extmodule with defname {name}")
    out, placed = list(head), False
    for k, n, ls in mods:
        if n in placeholders:
            if not placed:
                out += body + [""]
                placed = True
            continue
        for l in ls:
            m = re.match(r"(\s+inst \S+ of )(\S+)(.*)$", l)
            if m and m.group(2) in placeholders:
                l = m.group(1) + name + m.group(3)
            out.append(l)
    return out, len(placeholders)


circuit = open(sys.argv[1]).read().splitlines()
for m in sys.argv[2:]:
    name, body = module_of(m)
    circuit, n = splice(circuit, name, body)
    print(f"splice: {name} <- {m} ({n} placeholders)")
open(sys.argv[1], "w").write("\n".join(circuit) + "\n")
