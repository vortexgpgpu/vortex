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
# so that post_impl.dcp is visible.
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

# ---- Helpers ----------------------------------------------------------------

# Scan the VCD header and collect all scope paths (depth-first, dot-separated).
# Stops at $enddefinitions or after collecting max_scopes entries.
proc vcd_list_scopes {vcd_file {max_scopes 64}} {
    set f [open $vcd_file r]
    set stack {}
    set scopes {}
    while {[gets $f line] >= 0} {
        if {[string match {*$enddefinitions*} $line]} break
        foreach token [split $line] {
            # accumulate tokens into a simple state machine
        }
        if {[regexp {\$scope\s+\w+\s+(\S+)\s+\$end} $line -> name]} {
            lappend stack $name
            lappend scopes [join $stack "."]
            if {[llength $scopes] >= $max_scopes} break
        }
        if {[regexp {\$upscope\s+\$end} $line]} {
            set stack [lrange $stack 0 end-1]
        }
    }
    close $f
    return $scopes
}

# Resolve VCD_INST to an exact scope path.
# - Exact path  → returned unchanged.
# - Glob pattern → first VCD scope whose full path OR leaf name matches.
# - Empty string → returned unchanged.
# Errors and exits if a glob pattern is given but no scope matches.
proc resolve_vcd_inst {vcd_file pattern} {
    if {$pattern eq "" || ![string match {*[*?]*} $pattern]} {
        return $pattern
    }
    set f [open $vcd_file r]
    set stack {}
    set found ""
    while {[gets $f line] >= 0} {
        if {[string match {*$enddefinitions*} $line]} break
        if {[regexp {\$scope\s+\w+\s+(\S+)\s+\$end} $line -> name]} {
            lappend stack $name
            set full [join $stack "."]
            if {[string match $pattern $full] || [string match $pattern $name]} {
                set found $full
                break
            }
        }
        if {[regexp {\$upscope\s+\$end} $line]} {
            set stack [lrange $stack 0 end-1]
        }
    }
    close $f
    if {$found eq ""} {
        puts "ERROR: VCD_INST pattern '$pattern' did not match any scope in: $vcd_file"
        puts "       Available scopes:"
        foreach s [vcd_list_scopes $vcd_file] { puts "         $s" }
        exit 1
    }
    return $found
}

# ---- Validate environment ---------------------------------------------------

set checkpoint "post_impl.dcp"
if {![file exists $checkpoint]} {
  puts "ERROR: $checkpoint not found."
  puts "       Run 'make build' in this directory before running 'make power'."
  exit 1
}

if {![info exists ::env(VCD_FILE)] || $::env(VCD_FILE) eq ""} {
  puts "ERROR: VCD_FILE environment variable is not set."
  puts "       Usage: make power VCD_FILE=<path/to/sim.vcd>"
  exit 1
}

set vcd_file $::env(VCD_FILE)
if {![file exists $vcd_file]} {
  puts "ERROR: VCD file not found: $vcd_file"
  exit 1
}

set strip_path ""
if {[info exists ::env(VCD_INST)] && $::env(VCD_INST) ne ""} {
  set strip_path [resolve_vcd_inst $vcd_file $::env(VCD_INST)]
  if {[string match {*[*?]*} $::env(VCD_INST)]} {
    puts "INFO: VCD_INST '$::env(VCD_INST)' resolved to: $strip_path"
  }
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

# Check annotation coverage: capture the report as a string and scan for
# the "X% of nets annotated" line that Vivado emits during report_power.
set pwr_str [report_power -return_string]
if {[regexp {(\d+)%\s+of nets annotated} $pwr_str -> annot_pct]} {
  if {$annot_pct == 0} {
    puts ""
    puts "ERROR: 0% of nets were annotated from the VCD."
    if {$strip_path ne ""} {
      puts "       Strip path '$strip_path' did not match the synthesized netlist hierarchy."
    } else {
      puts "       No VCD_INST strip path was given — VCD signal names may not match the netlist."
    }
    puts "       Available VCD scopes:"
    foreach s [vcd_list_scopes $vcd_file] { puts "         $s" }
    puts "       Re-run with VCD_INST set to the scope containing the DUT signals."
    exit 1
  }
  puts "INFO: VCD annotation coverage: ${annot_pct}% of nets"
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
