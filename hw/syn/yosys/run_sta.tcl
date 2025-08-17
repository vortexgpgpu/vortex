# run_sta.tcl — OpenSTA 2.7.0 compatible (no update_timing, no report_clocks)

# ---- env ----
proc getenv {name {default ""}} {
  if {[info exists ::env($name)] && $::env($name) ne ""} { return $::env($name) }
  return $default
}
proc ensure_file {path msg} { if {![file exists $path]} { puts stderr "FATAL: $msg ($path)"; exit 1 } }

set TOP      [getenv TOP]
set NETLIST  [getenv NETLIST]
set LIB_TGT  [getenv LIB_TGT]
set LIB_ROOT [getenv LIB_ROOT]
set SDC_FILE [getenv SDC_FILE]
set RPT_DIR  [getenv RPT_DIR "."]

if {$TOP eq ""}     { puts stderr "FATAL: env TOP is required"; exit 1 }
if {$NETLIST eq ""} { puts stderr "FATAL: env NETLIST is required"; exit 1 }
ensure_file $NETLIST "NETLIST not found"

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

# ---- reports (no 'update_timing' needed) ----
report_units
set clks [get_clocks *]
if {[llength $clks] > 0} { report_clock_properties $clks } else { puts "No clocks defined." }

report_wns
report_tns

# Keep report_checks options conservative for 2.7.0 compatibility
report_checks -path_delay max -digits 3 -format full_clock_expanded
report_checks -path_delay min -digits 3 -format full_clock_expanded

# Optional:
# report_unconstrained
# write_sdf [file join $RPT_DIR "$TOP.sdf"]

puts "STA done."

# exit the application
exit
