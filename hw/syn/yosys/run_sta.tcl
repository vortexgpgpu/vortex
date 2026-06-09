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

# run_sta.tcl — OpenSTA 2.7.0 compatible (no update_timing, no report_clocks)

# ---- env ----
proc getenv {name {default ""}} {
  if {[info exists ::env($name)] && $::env($name) ne ""} { return $::env($name) }
  return $default
}
proc ensure_file {path msg} { if {![file exists $path]} { puts stderr "FATAL: $msg ($path)"; exit 1 } }
proc has_cmd {name} { expr {[llength [info commands $name]] > 0} }

set TOP      [getenv TOP]
set NETLIST  [getenv NETLIST]
set LIB_TGT  [getenv LIB_TGT]
set LIB_ROOT [getenv LIB_ROOT]
set SDC_FILE [getenv SDC_FILE]
set RPT_DIR  [getenv RPT_DIR "."]
set SAIF_FILE [getenv SAIF_FILE]
set SAIF_INST [getenv SAIF_INST]

if {$TOP eq ""}     { puts stderr "FATAL: env TOP is required"; exit 1 }
if {$NETLIST eq ""} { puts stderr "FATAL: env NETLIST is required"; exit 1 }
ensure_file $NETLIST "NETLIST not found"

file mkdir $RPT_DIR

# ---- (optional) reporting units; you can omit to silence unit-scale warnings
# set_cmd_units -time ns -capacitance fF -voltage V -current mA -resistance kOhm -power mW -distance um

# ---- liberty ----
if {$LIB_ROOT ne "" && [file isdirectory $LIB_ROOT]} {
  foreach lib [split [exec bash -lc "shopt -s nullglob globstar; printf '%s\n' $LIB_ROOT/**/*.lib | sort -u"] "\n"] {
    if {$lib ne ""} { puts "read_liberty $lib"; read_liberty $lib }
  }
}
if {$LIB_TGT ne "" && [file exists $LIB_TGT]} {
  puts "read_liberty $LIB_TGT"
  read_liberty $LIB_TGT
}

# ---- netlist ----
puts "read_verilog $NETLIST"
read_verilog $NETLIST
puts "link_design $TOP"
link_design $TOP

# ---- SDC ----
if {$SDC_FILE ne "" && [file exists $SDC_FILE]} {
  puts "read_sdc $SDC_FILE"
  read_sdc $SDC_FILE
}

# IMPORTANT: do NOT propagate virtual clocks
# If you use set_propagated_clock, restrict to real on-chip clocks, e.g.:
# set_propagated_clock [get_clocks core_clk]

# ---- timing reports (no 'update_timing' needed) ----
report_units
set clks [get_clocks *]
if {[llength $clks] > 0} { report_clock_properties $clks } else { puts "No clocks defined." }

report_wns
report_tns

# Keep report_checks options conservative for 2.7.0 compatibility
report_checks -path_delay max -digits 3 -format full_clock_expanded
report_checks -path_delay min -digits 3 -format full_clock_expanded

# If SAIF is not provided (or unsupported), we intentionally fall back to vectorless/default power.
if {$SAIF_FILE ne "" && [file exists $SAIF_FILE]} {
  if {[has_cmd read_saif]} {
    if {$SAIF_INST ne ""} {
      puts "read_saif -instance $SAIF_INST $SAIF_FILE"
      read_saif -instance $SAIF_INST $SAIF_FILE
    } else {
      puts "read_saif $SAIF_FILE"
      read_saif $SAIF_FILE
    }
  } else {
    puts "WARNING: 'read_saif' not available in this STA build; cannot annotate SAIF (power will be vectorless/default)."
  }
} else {
  puts "INFO: SAIF_FILE not provided (or not found); power will be vectorless/default."
}

# Optional: help diagnose annotation coverage when SAIF is used
if {[has_cmd report_switching_activity]} {
  catch { report_switching_activity -list_not_annotated > [file join $RPT_DIR "saif_unannotated.rpt"] }
}

# ---- power reports (always) ----
if {[has_cmd report_power]} {
  report_power > [file join $RPT_DIR "power.rpt"]
  catch { report_power -hier > [file join $RPT_DIR "power_hier.rpt"] }
} else {
  puts "WARNING: 'report_power' not available in this STA build."
}

puts "STA done."

# exit the application
exit
