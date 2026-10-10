#!/usr/bin/env python3
"""Tag the wires whose bits are driven by other bits of the same wire.

usage: selfref_wires.py <design.il>   (rewritten in place; prints the count)

Yosys reuses a bit of a wire to drive other bits of it (a write-enable vector
built by replicating one enable bit, say). That is acyclic bit by bit, but
FIRRTL checks loops per wire, so the exported module fails the combinational
loop check. Each such wire gets the attribute fame5_selfref, for
`splitnets a:fame5_selfref` to split it into single bits; nothing else is
touched.
"""

import re
import sys

il = sys.argv[1]
lines = open(il).read().splitlines(keepends=True)

selfref = set()  # (module, wire)
mod = None
for line in lines:
    if line.startswith("module "):
        mod = line.split()[1]
    elif line.startswith("  connect "):
        m = re.match(r"  connect (\\\S+|\$\S+)(?: \[[0-9:]+\])?\s+(.*)", line)
        if m and m.group(1) in re.findall(r"(\\\S+|\$\S+)", m.group(2)):
            selfref.add((mod, m.group(1)))

if selfref:
    out = []
    mod = None
    for line in lines:
        if line.startswith("module "):
            mod = line.split()[1]
        elif line.startswith("  wire "):
            name = line.split()[-1]
            if (mod, name) in selfref:
                out.append("  attribute \\fame5_selfref 1\n")
        out.append(line)
    open(il, "w").write("".join(out))
print(len(selfref))
