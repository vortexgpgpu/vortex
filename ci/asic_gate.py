#!/usr/bin/env python3
"""asic_gate — ASIC synthesis-regression gate (Yosys + OpenSTA, ASAP7).

The open-source counterpart of ci/fpga_gate.py: the same golden-baseline
discipline, over a flow that needs no licence and no dedicated machine, gating
post-synthesis Fmax and standard-cell area. The gating machinery is shared and
lives in ci/synth_gate.py; this entry point only pins the tool.

  Spec      : ci/testcases/asic_gate.yaml                (what to build)
  Baselines : ci/baselines/synthesis/yosys/<group>.json  (last accepted metrics)
  Builds    : hw/syn/yosys/asic_gate_<top>               (out-of-tree, per DUT)
  Metrics   : <build>/synth_summary.csv, written by hw/syn/yosys/synth_summary.py
              from the reports/ set the flow already produces

Unlike the FPGA gate this one runs on a hosted runner as a normal pytest cell
(`check: asic_gate`, tier `full`), because yosys/sv2v/OpenSTA ship in the
prebuilt toolchain and ASAP7 is a 59 MB fetch.

Usage (from a source checkout):
  ci/asic_gate.py --list
  ci/asic_gate.py                       # all builds, gate against baselines
  ci/asic_gate.py -b om -b tex -j 4     # explicit builds, 4 in parallel
  ci/asic_gate.py -b graphics           # a whole group
  ci/asic_gate.py --update-baseline     # re-record (human-reviewed, never in CI)

The same DUTs are runnable by hand through the flow's own entry point --
`make -C hw/syn/yosys/dut om` -- so the gate adds no second way to build.

See docs/designs/continuous_integration.md 3.5.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_gate  # noqa: E402

if __name__ == "__main__":
    sys.exit(synth_gate.main(default_tool="yosys"))
