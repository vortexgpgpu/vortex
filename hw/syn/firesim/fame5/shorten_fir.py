#!/usr/bin/env python3
"""Shorten internal identifiers in Yosys-written FIRRTL.

usage: shorten_fir.py <in.fir> <out.fir>

Flattening leaves every wire of the top region carrying its full
hierarchical path as its name, and splitnets multiplies those wires per
bit: the identifiers become most of the file's bytes, and a full-GPU
export exceeds the FIRRTL reader's 2 GiB input limit. This pass renames
module-local identifiers (wire/node/reg/cmem/smem/mem/inst names) longer
than a threshold to sequential short names, per module. Module names and
port names are never touched, so instantiation, splicing and annotation
are unaffected. Renaming is semantics-preserving: FIRRTL identifiers are
local to their module.
"""

import re
import sys

THRESHOLD = 8  # keep short names for readability; rename only long ones

DECL = re.compile(
    r"^\s*(?:wire|node|reg|cmem|smem|inst)\s+([A-Za-z_][A-Za-z0-9_$]*)"
)
MEM = re.compile(r"^\s*mem\s+([A-Za-z_][A-Za-z0-9_$]*)\s*:")
MODULE = re.compile(r"^  (?:ext)?module\s+(\S+?)\s*:")
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")

src, dst = sys.argv[1], sys.argv[2]

# Pass 1: per module, map long declared names to short ones.
maps = {}  # module -> {long: short}
mod = None
with open(src, "r", errors="strict") as fh:
    for line in fh:
        m = MODULE.match(line)
        if m:
            mod = m.group(1)
            maps[mod] = {}
            continue
        if mod is None:
            continue
        d = DECL.match(line) or MEM.match(line)
        if d:
            name = d.group(1)
            if len(name) > THRESHOLD and name not in maps[mod]:
                maps[mod][name] = f"sn{len(maps[mod])}"

# Safety: no existing short identifier may collide with generated ones.
collide = re.compile(r"^sn\d+$")
for mod_name, mp in maps.items():
    for long_name in mp:
        assert not collide.match(long_name), f"collision risk: {long_name}"

LOCATOR = re.compile(r"\s*@\[[^\]]*\]")

def rename_line(line, mp):
    line = LOCATOR.sub("", line)
    return TOKEN.sub(lambda m: mp.get(m.group(0), m.group(0)), line)

# Pass 2: rewrite.
mod = None
mp = {}
with open(src, "r", errors="strict") as fh, open(dst, "w") as out:
    for line in fh:
        m = MODULE.match(line)
        if m:
            mod = m.group(1)
            mp = maps.get(mod, {})
            out.write(LOCATOR.sub("", line))
            continue
        out.write(rename_line(line, mp))

total = sum(len(m) for m in maps.values())
print(f"shorten_fir: renamed {total} identifiers across {len(maps)} modules")
