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
# xilinx_power_analysis.tcl — Shared SAIF-annotated power analysis
#
# Used by both the DUT standalone flow and the XRT Vitis flow.
# Run via "make power SAIF_FILE=<path>" after a successful build.
#
# Required environment variable:
#   SAIF_FILE       Path to the SAIF produced by rtlsim/xrtsim with SAIF=1
#                   (verilator --trace-saif)
#
# Checkpoint resolution (first match wins):
#   DCP_FILE        Explicit path to the post-implementation checkpoint.
#   BUILD_DIR       Vitis build root — script searches for the routed DCP under
#                   <BUILD_DIR>/_x/link/vivado/vpl/prj/prj.runs/impl_1/.
#   (fallback)      post_impl.dcp in the current working directory (DUT flow).
#
# Optional environment variables:
#   SAIF_INST       Instance path prefix to strip from SAIF signal names so they
#                   match the synthesized netlist hierarchy.  E.g.:
#                     DUT flow: TOP.rtlsim_shim.vortex
#                     XRT flow: TOP.vortex_afu_shim.vortex_afu
#                   Leave unset if the SAIF root scope already matches.
#   OUT_DIR         Directory for output reports.
#                   Defaults to <BUILD_DIR>/bin if BUILD_DIR is set,
#                   otherwise the current working directory.
#
# Outputs:
#   power_vectorless.rpt   Baseline vectorless estimate (12.5% toggle rate)
#   power_saif.rpt         SAIF-annotated estimate
# -----------------------------------------------------------------------------

# ---- Resolve checkpoint -----------------------------------------------------

set checkpoint ""

if {[info exists ::env(DCP_FILE)] && $::env(DCP_FILE) ne ""} {
  set checkpoint $::env(DCP_FILE)
} elseif {[info exists ::env(BUILD_DIR)] && $::env(BUILD_DIR) ne ""} {
  set impl_dir "$::env(BUILD_DIR)/_x/link/vivado/vpl/prj/prj.runs/impl_1"
  set candidates [glob -nocomplain "$impl_dir/*.dcp"]
  if {[llength $candidates] == 0} {
    puts "ERROR: No implementation checkpoint found under: $impl_dir"
    puts "       Run 'make all TARGET=hw' before running 'make power'."
    exit 1
  }
  # Prefer the routed checkpoint if multiple exist.
  set routed [lsearch -inline $candidates "*routed*"]
  set checkpoint [expr {$routed ne "" ? $routed : [lindex $candidates 0]}]
} else {
  set checkpoint "post_impl.dcp"
}

if {![file exists $checkpoint]} {
  puts "ERROR: Checkpoint not found: $checkpoint"
  puts "       Run 'make build' (DUT) or 'make all TARGET=hw' (XRT) first."
  exit 1
}

# ---- Resolve SAIF file -------------------------------------------------------

if {![info exists ::env(SAIF_FILE)] || $::env(SAIF_FILE) eq ""} {
  puts "ERROR: SAIF_FILE environment variable is not set."
  puts "       Usage: make power SAIF_FILE=<path/to/trace.saif>"
  exit 1
}
set saif_file $::env(SAIF_FILE)
if {![file exists $saif_file]} {
  puts "ERROR: SAIF file not found: $saif_file"
  exit 1
}

set strip_path ""
if {[info exists ::env(SAIF_INST)] && $::env(SAIF_INST) ne ""} {
  set strip_path $::env(SAIF_INST)
}

# ---- Resolve output directory -----------------------------------------------

if {[info exists ::env(OUT_DIR)] && $::env(OUT_DIR) ne ""} {
  set out_dir $::env(OUT_DIR)
} elseif {[info exists ::env(BUILD_DIR)] && $::env(BUILD_DIR) ne ""} {
  set out_dir "$::env(BUILD_DIR)/bin"
} else {
  set out_dir "."
}
file mkdir $out_dir

# ---- Open the post-implementation checkpoint --------------------------------

puts "INFO: Opening checkpoint: $checkpoint"
open_checkpoint $checkpoint

# ---- Vectorless baseline ----------------------------------------------------
# Provides a reference so SAIF and vectorless results can be compared.

puts "INFO: Generating vectorless baseline power report..."
reset_switching_activity -all
set_switching_activity -default_toggle_rate 0.125 -default_static_probability 0.5
set_switching_activity -deassert_resets
report_power -file $out_dir/power_vectorless.rpt
puts "INFO: Vectorless report written to: $out_dir/power_vectorless.rpt"

# ---- SAIF-annotated power ---------------------------------------------------
# read_saif annotates toggle rates for every signal present in the SAIF file.
# Signals absent from the SAIF retain Vivado's default activity model.

puts "INFO: Reading SAIF: $saif_file"
reset_switching_activity -all
if {$strip_path ne ""} {
  puts "INFO: Stripping SAIF path prefix: $strip_path"
  read_saif -strip_path $strip_path $saif_file
} else {
  read_saif $saif_file
}

# Deassert resets to avoid inflating steady-state power from reset pulses.
set_switching_activity -deassert_resets

puts "INFO: Generating SAIF-annotated power report..."
report_power -file $out_dir/power_saif.rpt
puts "INFO: SAIF-annotated report written to: $out_dir/power_saif.rpt"

# ---- Summary ----------------------------------------------------------------

puts ""
puts "Power analysis complete."
puts "  Vectorless baseline : $out_dir/power_vectorless.rpt"
puts "  SAIF-annotated      : $out_dir/power_saif.rpt"
puts ""
puts "Tip: diff the two reports to see which modules' activity differs most"
puts "     from the 12.5% vectorless assumption."
