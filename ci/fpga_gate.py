#!/usr/bin/env python3
"""fpga_gate — FPGA synthesis-regression gate (Xilinx/Vivado DUT builds).

Compares THIS commit's post-implementation Fmax and LUT count against a
checked-in golden baseline, so an RTL change that quietly costs timing closure
or area cannot land green. The gating machinery is shared with the ASIC gate
(ci/asic_gate.py) and lives in ci/synth_gate.py; this entry point only pins the
tool.

  Spec      : ci/testcases/fpga_gate.yaml                 (what to build)
  Baselines : ci/baselines/synthesis/xilinx/<group>.json  (last accepted metrics)
  Builds    : hw/syn/xilinx/dut/fpga_gate_<id>            (Vivado, out-of-tree)
  Metrics   : <build>/synth_summary.csv, timing.rpt, high_fanout_nets.rpt
              (all three already written by hw/syn/xilinx/dut/project.tcl)

Usage (from a source checkout, with the Xilinx env sourced):
  source ~/dev/xilinx_setup.sh
  ci/fpga_gate.py --list
  ci/fpga_gate.py                       # all builds, gate against baselines
  ci/fpga_gate.py -b tcu -b rtu -j 2    # explicit builds, 2 in parallel
  ci/fpga_gate.py -b graphics           # a whole group
  ci/fpga_gate.py --update-baseline     # re-record (human-reviewed, never in CI)

See docs/designs/continuous_integration.md 3.5.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_gate  # noqa: E402

if __name__ == "__main__":
    sys.exit(synth_gate.main(default_tool="xilinx"))
