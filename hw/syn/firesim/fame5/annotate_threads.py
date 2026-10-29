#!/usr/bin/env python3
"""Annotate every instance of a module for FAME-5 threading, in the anno file.

usage: annotate_threads.py <circuit.fir> <anno.json> <module>

Finds each `inst <name> of <module>` in <circuit.fir> and appends a
FirrtlEnableModelMultiThreadingAnnotation targeting it -- what Chisel's
EnableModelMultiThreadingAnnotation would emit, for designs with no Chisel.
"""
import json
import re
import sys

fir, anno, mod = sys.argv[1:]
circuit, parent, targets = None, None, []
for line in open(fir):
    m = re.match(r"circuit (\S+?)\s*:", line)
    if m:
        circuit = m.group(1)
    m = re.match(r"  (?:ext)?module (\S+?)\s*:", line)
    if m:
        parent = m.group(1)
    m = re.match(rf"\s+inst (\S+) of {re.escape(mod)}\b", line)
    if m:
        targets.append(f"~{circuit}|{parent}/{m.group(1)}:{mod}")
if not targets:
    sys.exit(f"annotate_threads: no instance of {mod}")
annos = json.load(open(anno))
annos = [a for a in annos if not (a["class"].endswith("FirrtlEnableModelMultiThreadingAnnotation")
                                  and a["target"] in targets)]
annos += [{"class": "midas.targetutils.FirrtlEnableModelMultiThreadingAnnotation", "target": t} for t in targets]
json.dump(annos, open(anno, "w"), indent=2)
print(f"annotate_threads: {len(targets)} instances of {mod}")
