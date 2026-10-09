#!/usr/bin/env python3
"""Point Golden Gate at the Verilog of the FIRRTL black boxes the flow emits.

usage: annotate_blackboxes.py <circuit.fir> <circuit.anno.json>

Adds a BlackBoxPathAnno for every FameRam extmodule (gen_rams with
FAME5_FAMERAM=1), so FameRam.v is copied next to the generated design and
appended to it, threaded or not.
"""
import json
import os
import re
import sys

fir, anno = sys.argv[1], sys.argv[2]
src = open(fir).read()
circuit = re.search(r"^circuit (\S+?)\s*:", src, re.M).group(1)
names = re.findall(r"^  extmodule (\S+?)\s*:\n(?:    .*\n)*?    defname = FameRam$", src, re.M)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FameRam.v")
annos = json.load(open(anno))
annos += [{"class": "firrtl.transforms.BlackBoxPathAnno", "target": f"{circuit}.{n}", "path": path} for n in names]
json.dump(annos, open(anno, "w"), indent=1)
print(f"annotate_blackboxes: {len(names)} FameRam black boxes")
