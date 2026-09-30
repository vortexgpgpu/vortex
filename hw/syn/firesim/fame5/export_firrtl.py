#!/usr/bin/env python3
"""FIRRTL export with selective memory lowering.

usage: export_firrtl.py <design.il> <out.fir> <log>

FIRRTL write ports carry one enable and one uniform mask, so write_firrtl
rejects the occasional $mem whose ports encode read-modify-write, per-field
or shifting stores ('Complex write enable on port N on memory M'). Every
memory it can express must stay a memory -- FAME-5 threading turns those
into dense deep LUTRAM/BRAM -- so only the memories the writer itself names
are memory_map'ed to FF logic, and the export is retried until it goes
through. The rejected arrays are register-class storage in every flow
(async reads, a few entries deep); this just makes that explicit.
"""

import os
import re
import subprocess
import sys

GLOB = re.compile(r"[][*?]")
ERR = re.compile(r"Complex write enable on port \d+ on memory (\S+?)\.?\s*$", re.M)

il, out, log = sys.argv[1:4]
yosys = os.environ.get("YOSYS", "yosys")

mapped = []  # selections accumulated across attempts
with open(log, "w") as lf:
    for attempt in range(30):
        script = f"read_rtlil {il}\n"
        if mapped:
            script += "memory_map " + " ".join(mapped) + "\n"
        script += f"write_firrtl {out}\n"
        r = subprocess.run([yosys, "-q", "-p", script], capture_output=True, text=True)
        lf.write(f"--- attempt {attempt}: mapped={mapped}\n{r.stdout}{r.stderr}\n")
        lf.flush()
        if r.returncode == 0:
            sys.exit(0)
        m = ERR.search(r.stdout + r.stderr)
        if not m:
            lf.write("unrecognized failure\n")
            print(f"export_firrtl: unrecognized yosys failure (see {log})", file=sys.stderr)
            sys.exit(1)
        path = m.group(1)
        module, tail = path.split(".", 1)[0], path.rsplit(".", 1)[-1]
        if GLOB.search(module + tail):
            # fall back to the exact path with glob metachars neutralized
            sel = [f"{module}/c:" + GLOB.sub("?", path.split('.', 1)[1])]
        else:
            # map every instance of this array kind in one step
            sel = [f"{module}/c:*.{tail}", f"{module}/c:{tail}"]
        print(f"export_firrtl: memory_map {' '.join(sel)}  ({path})")
        mapped += sel

print(f"export_firrtl: did not converge in 30 attempts (see {log})", file=sys.stderr)
sys.exit(1)
