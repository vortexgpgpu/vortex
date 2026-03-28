# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -----------------------------------------------------------------------------
# power_analysis.tcl — Standalone VCD-annotated power analysis
#
# Intended for use with "make power VCD=<path>" after a successful build.
# Must be run from the DUT build directory (e.g. hw/syn/xilinx/dut/vortex/build/)
# so that project_1/post_impl.dcp is visible.
#
# Required environment variable:
#   VCD_FILE        Path to the VCD produced by rtlsim (Verilator --trace)
#
# Optional environment variables:
#   VCD_INST        Instance path of the DUT in the simulation hierarchy.
#                   Stripped from VCD signal paths so they match the synthesized
#                   netlist.  Mirrors SAIF_INST in the Synopsys flow.
#                   Verilator default scope is "TOP.<top_module>", e.g.:
#                     VCD_INST=TOP.Vortex
#                   Leave unset/empty if the VCD root scope already matches
#                   the top module name.
#   TOOL_DIR        Path to hw/scripts (set automatically by common.mk)
#
# Outputs written to the build directory:
#   power_vectorless.rpt   Baseline vectorless estimate (no VCD)
#   power_vcd.rpt          VCD-annotated estimate (per-module, depth 6)
# -----------------------------------------------------------------------------

# ---- Validate environment ---------------------------------------------------

set checkpoint "project_1/post_impl.dcp"
if {![file exists $checkpoint]} {
  puts "ERROR: $checkpoint not found."
  puts "       Run 'make build' in this directory before running 'make power'."
  exit 1
}

if {![info exists ::env(VCD_FILE)] || $::env(VCD_FILE) eq ""} {
  puts "ERROR: VCD_FILE environment variable is not set."
  puts "       Usage: make power VCD=<path/to/sim.vcd>"
  exit 1
}

set vcd_file $::env(VCD_FILE)
if {![file exists $vcd_file]} {
  puts "ERROR: VCD file not found: $vcd_file"
  exit 1
}

set strip_path ""
if {[info exists ::env(VCD_INST)]} {
  set strip_path $::env(VCD_INST)
}

# ---- Open the post-implementation checkpoint --------------------------------

puts "INFO: Opening checkpoint: $checkpoint"
open_checkpoint $checkpoint

# ---- Vectorless baseline (no VCD) -------------------------------------------
# Provides a reference so VCD and vectorless results can be compared side-by-side.

puts "INFO: Generating vectorless baseline power report..."
reset_switching_activity -all
set_switching_activity -default_toggle_rate 0.125 -default_static_probability 0.5
set_switching_activity -deassert_resets
report_power -file power_vectorless.rpt
puts "INFO: Vectorless report written to: power_vectorless.rpt"

# ---- VCD-annotated power ----------------------------------------------------
# read_vcd annotates toggle rates and static probabilities for every signal
# present in the VCD.  Signals absent from the VCD retain the Vivado default
# activity model, so partial VCDs are still useful.

puts "INFO: Reading VCD: $vcd_file"
reset_switching_activity -all
if {$strip_path ne ""} {
  puts "INFO: Stripping VCD path prefix: $strip_path"
  read_vcd -strip_path $strip_path $vcd_file
} else {
  read_vcd $vcd_file
}

# Override reset nets to deasserted — prevents reset pulses at the start of
# simulation from inflating the steady-state power estimate.
set_switching_activity -deassert_resets

puts "INFO: Generating VCD-annotated power report..."
report_power -file power_vcd.rpt
puts "INFO: VCD-annotated report written to: power_vcd.rpt"

# ---- Summary ----------------------------------------------------------------

puts ""
puts "Power analysis complete."
puts "  Vectorless baseline : power_vectorless.rpt"
puts "  VCD-annotated       : power_vcd.rpt"
puts ""
puts "Tip: diff the two reports to see which modules' activity differs most"
puts "     from the 12.5% vectorless assumption."
