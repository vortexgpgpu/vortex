#!/usr/bin/env python3
"""Give kept parameterized modules public names in an RTLIL design.

usage: name_kept.py <design.il>   (rewritten in place)

After flattening, the only modules left are the top, the kept units (FAME-5
classes) and the black boxes. A kept unit that was instantiated with
parameters is named $paramod...\\<module>, and Yosys write_firrtl names its
instances by the cell's hdlname attribute rather than the module, so they
would not refer to the module it emits. Renaming each non-blackbox $paramod
module to \\<module>__p<n> (module and every cell of that type) makes the
writer use the module's own name.
"""

import re
import sys

il = sys.argv[1]
text = open(il).read()

mods = []
attrs = []
for line in text.splitlines():
    s = line.strip()
    if s.startswith("attribute "):
        attrs.append(s)
        continue
    if s.startswith("module "):
        name = s.split(None, 1)[1]
        if name.startswith("$paramod") and not any(a.startswith("attribute \\blackbox") for a in attrs):
            mods.append(name)
    attrs = []

if mods:
    ren = {}
    count = {}
    for m in mods:
        base = re.sub(r"[^A-Za-z0-9_]", "_", m.split("\\")[1]) if "\\" in m else "mod"
        n = count.get(base, 0)
        count[base] = n + 1
        ren[m] = f"\\{base}__p{n}"
    out = []
    for line in text.splitlines(keepends=True):
        s = line.lstrip()
        if s.startswith("module ") or s.startswith("cell "):
            parts = line.split(" ")
            for i, p in enumerate(parts):
                key = p.rstrip("\n")
                if key in ren:
                    parts[i] = ren[key] + ("\n" if p.endswith("\n") else "")
            line = " ".join(parts)
        out.append(line)
    open(il, "w").write("".join(out))
print(f"name_kept: {len(mods)} kept parameterized modules renamed")
