#!/usr/bin/env python3
"""logic_gate — logic-level and area regression gate (Beacon).

Gates the depth of every RTL unit of the DUT catalog, and each DUT's area:
foldsim-estimate-timing counts the logic levels of every register-to-register
path of the CIRCT netlist the Xilinx flow elaborates, as Vivado counts them on
DEVICE's family, and foldsim-estimate-area its LUTs, flip-flops, distributed
and block RAM and DSPs, without running Vivado. A unit that gets one level
deeper than its baseline fails, by name, and a resource more than 5% larger,
in seconds -- where ci/fpga_gate.py finds the Fmax or area it costs the next
night. The
gating machinery is shared with the synthesis gates and lives in
ci/synth_gate.py; this entry point only pins the tool.

  Spec      : ci/testcases/logic_gate.yaml                (what to measure)
  Baselines : ci/baselines/synthesis/beacon/<group>.json   (last accepted levels)
  Builds    : hw/syn/xilinx/dut/logic_gate_<dut>          (no Vivado)
  Metrics   : <build>/beacon_summary.csv, beacon_units.json, beacon_area.txt
              (written by hw/syn/xilinx/dut's `beacon` target)

It needs CIRCT and foldsim (ci/toolchain_install.sh --circt --foldsim); the
Xilinx environment is not used.

Usage (from a source checkout):
  ci/logic_gate.py --list
  ci/logic_gate.py                       # all builds, gate against baselines
  ci/logic_gate.py -b cache -b tcu       # explicit builds
  ci/logic_gate.py --update-baseline     # re-record (human-reviewed, never in CI)

Beacon runs by hand through the flow's own entry point --
`make -C hw/syn/xilinx/dut cache-beacon` -- so the gate adds no second way to
build.

See docs/designs/continuous_integration.md 3.5.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth_gate  # noqa: E402

if __name__ == "__main__":
    sys.exit(synth_gate.main(default_tool="beacon"))
