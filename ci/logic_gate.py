#!/usr/bin/env python3
"""logic_gate — logic-level regression gate (the logic screen).

Gates the depth of every RTL unit of the DUT catalog: foldsim-estimate-timing
counts the logic levels of every register-to-register path of the CIRCT netlist
the Xilinx flow elaborates, as Vivado counts them on DEVICE's family, without
running Vivado. A unit that gets one level deeper than its baseline fails, by name, in
seconds -- where ci/fpga_gate.py finds the Fmax it costs the next night. The
gating machinery is shared with the synthesis gates and lives in
ci/synth_gate.py; this entry point only pins the tool.

  Spec      : ci/testcases/logic_gate.yaml                (what to screen)
  Baselines : ci/baselines/synthesis/screen/<group>.json   (last accepted levels)
  Builds    : hw/syn/xilinx/dut/logic_gate_<dut>          (no Vivado)
  Metrics   : <build>/screen_summary.csv, screen_units.json
              (written by hw/syn/xilinx/dut's `screen` target)

It needs CIRCT and foldsim (ci/toolchain_install.sh --circt --foldsim); the
Xilinx environment is not used.

Usage (from a source checkout):
  ci/logic_gate.py --list
  ci/logic_gate.py                       # all builds, gate against baselines
  ci/logic_gate.py -b cache -b tcu       # explicit builds
  ci/logic_gate.py --update-baseline     # re-record (human-reviewed, never in CI)

The same screen runs by hand through the flow's own entry point --
`make -C hw/syn/xilinx/dut cache-screen` -- so the gate adds no second way to
build.

See docs/designs/continuous_integration.md 3.5.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_gate  # noqa: E402

if __name__ == "__main__":
    sys.exit(synth_gate.main(default_tool="screen"))
