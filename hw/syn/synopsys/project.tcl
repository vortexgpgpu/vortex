###############################################################################
# Synopsys DC — Generic Synthesis (SystemVerilog) using .db libraries
# Inputs via environment variables:
#   TOP            : top module name (required)
#   SRC_FILE       : path to sources.txt (VCS-style +incdir/+define) (required)
#   LIB_ROOT       : root folder to recursively discover *.db (or)
#   LIB_TGT        : explicit target .db (optional)
#   SDC_FILE       : constraints file (optional)
#   BB_MODULES     : comma-separated list of modules to set as dont_touch (optional)
#   OUT_DIR        : output netlist/sdf folder (default: ./out)
#   RPT_DIR        : reports folder (default: ./reports)
#   TOOL_DIR       : folder containing parse_vcs_list.tcl (default: script dir)
###############################################################################

# ---------------- helpers ----------------
proc getenv {name default} {
  if {[info exists ::env($name)] && $::env($name) ne ""} { return $::env($name) }
  return $default
}
proc DIE {msg}  { puts stderr "FATAL: $msg"; exit 1 }
proc WARN {msg} { puts "WARN: $msg" }

# Detects if a SystemVerilog file defines a package (returns 1/0)
proc sv_file_has_package {path} {
  set ret 0

  if {[file exists $path]} {
    # Read file
    set ch [open $path r]
    fconfigure $ch -encoding utf-8 -translation auto
    set code [read $ch]
    close $ch

    # Strip /* ... */ block comments (multi-pass, multiline safe)
    while {[regsub -all {/\*[^*]*\*+([^/*][^*]*\*+)*/} $code "" code]} {}

    # Strip // line comments
    regsub -all {//[^\n]*} $code "" code

    # Primary: line-anchored 'package' (handles 'automatic' and macro’d names)
    if {[regexp -line -- {^\s*package\s+(?:automatic\s+)?([A-Za-z_`][A-Za-z0-9_`$]*)} $code]} {
      set ret 1
    } elseif {[regexp -- {package\s+(?:automatic\s+)?([A-Za-z_`][A-Za-z0-9_`$]*)\s*(?:#\s*\([^;]*\)\s*)?;} $code]} {
      # Fallback: tolerant match anywhere, including parameterized headers before ';'
      set ret 1
    }
  }

  # puts "sv_file_has_package: $path -> $ret"
  return $ret
}

# Robust recursive file finder (dbs only)
proc find_files {root patterns} {
  set out {}
  if {![file exists $root]} { return $out }
  set cmd [list find $root -type f]
  lappend cmd \(
  set first 1
  foreach pat $patterns {
    if {$first} { lappend cmd -name $pat; set first 0 } else { lappend cmd -o -name $pat }
  }
  lappend cmd \)
  if {[catch {eval exec $cmd} lines]} { return $out }
  foreach f [split $lines "\n"] { if {$f ne ""} { lappend out $f } }
  return $out
}

# Filter likely std-cell db’s (avoid IO/SRAM/PLL/etc.)
proc filter_stdcell {files} {
  set keep {std rvt svt sc hvt lvt slvt base cell}
  set drop {io pad sram pll adpll phy rf adc dac mem memory macro}
  set out {}
  foreach f $files {
    set lf [string tolower $f]
    set k 0; foreach t $keep { if {[string first $t $lf] >= 0} { set k 1; break } }
    if {!$k} { continue }
    set d 0; foreach t $drop { if {[string first $t $lf] >= 0} { set d 1; break } }
    if {$d} { continue }
    lappend out $f
  }
  return $out
}

# Corner classification & TT picker
proc classify_corner {f} {
  set lf [string tolower $f]
  if {[regexp {\btt\b|_tt} $lf]} { return TT }
  if {[regexp {\bss\b|_ss} $lf]} { return SS }
  if {[regexp {\bff\b|_ff} $lf]} { return FF }
  return OTHER
}
proc pick_tt_target {files} {
  set tt {}; foreach f $files { if {[classify_corner $f] eq "TT"} { lappend tt $f } }
  if {[llength $tt]==0} { return "" }
  set ranks [dict create rvt 1 std 1 svt 1 lvt 2 slvt 2 hvt 3 base 2]
  set best ""; set best_score 99
  foreach f $tt {
    set lf [string tolower $f]; set score 50
    foreach {k v} $ranks { if {[string first $k $lf] >= 0} { set score $v; break } }
    if {$score < $best_score} { set best $f; set best_score $score }
  }
  return $best
}

# Small utilities
proc uniq {lst} {
    array set seen {}
    set out {}
    foreach x $lst {
        if {![info exists seen($x)]} {
            set seen($x) 1
            lappend out $x
        }
    }
    return $out
}
proc basename {p} { return [file tail $p] }

# ------------------------- Main Flow ------------------------- #

# Setup environment
set TOP          [getenv TOP          ""]
set SRC_FILE [getenv SRC_FILE ""]
set SDC_FILE     [getenv SDC_FILE     ""]
set LIB_ROOT     [getenv LIB_ROOT     ""]
set LIB_TGT_HINT [getenv LIB_TGT      ""]
set BB_MODULES   [getenv BB_MODULES   ""]
set TOOL_DIR     [getenv TOOL_DIR     ""]
set OUT_DIR      [file normalize [getenv OUT_DIR "./out"]]
set RPT_DIR      [file normalize [getenv RPT_DIR "./reports"]]

# Validate environment
if {$TOP eq ""}          { DIE "TOP not set" }
if {$SRC_FILE eq ""} { DIE "SRC_FILE not set" }
if {![file exists $SRC_FILE]} { DIE "SRC_FILE not found: $SRC_FILE" }

# Create output directories
file mkdir $OUT_DIR $RPT_DIR

# Parse source list
source [file join $TOOL_DIR "parse_vcs_list.tcl"]
lassign [parse_vcs_list $SRC_FILE] v_files incdirs defines

# Validate all source files exist
set missing [list]
foreach f $v_files {
  if {![file exists $f]} { lappend missing $f }
}
if {[llength $missing]} {
  DIE "Missing source files:\n  [join $missing "\n  "]"
}

# Filter header-only files; order packages first
set hdr_ext { .vh .svh }
set pkg_files {}
set other_files {}

foreach f $v_files {
  set ext [string tolower [file extension $f]]
  if {$ext in $hdr_ext} continue

  set is_pkg_name [expr {[string match "*_pkg.sv" [string tolower [file tail $f]]] ? 1 : 0}]
  set is_pkg_file [expr {$is_pkg_name || [sv_file_has_package $f]}]

  if {$is_pkg_file} {
    lappend pkg_files $f
  } else {
    lappend other_files $f
  }
}

set v_files [concat [uniq $pkg_files] [uniq $other_files]]
set incdirs [uniq $incdirs]
set defines [uniq $defines]

# Add critical missing define for Vortex
lappend defines "FPGA_TARGET_SAED"

puts "\n===== SOURCES ====="
puts "Top         : $TOP"
puts "SV Files    : [llength $v_files]"
puts "First file  : [lindex $v_files 0]"
puts "Last file   : [lindex $v_files end]"
puts "Incdirs     : [join $incdirs "\n"]"
puts "Defines     : [join $defines " "]"  ;# Show all defines
puts "===================\n"

# ---------------- library setup (.db only) ----------------
set target_db ""; set link_dbs {}
if {$LIB_TGT_HINT ne ""} {
  if {![file exists $LIB_TGT_HINT]} { DIE "LIB_TGT points to missing file: $LIB_TGT_HINT" }
  set target_db $LIB_TGT_HINT
  set link_dbs  [list $target_db]
} else {
  if {$LIB_ROOT eq ""} { DIE "Set LIB_ROOT or LIB_TGT in environment" }
  if {![file exists $LIB_ROOT]} { DIE "LIB_ROOT does not exist: $LIB_ROOT" }
  set dbs [find_files $LIB_ROOT {*.db *.DB}]
  if {[llength $dbs]==0} { DIE "No .db files found under $LIB_ROOT" }
  set std_dbs [filter_stdcell $dbs]
  if {[llength $std_dbs]==0} {
    WARN "No std-cell-looking .db found; using all .db"
    set std_dbs $dbs
  }
  set target_db [pick_tt_target $std_dbs]
  if {$target_db eq ""} {
    WARN "No TT .db detected; falling back to first .db"
    set target_db [lindex $std_dbs 0]
  }
  set link_dbs [list $target_db]
  foreach f $std_dbs { if {$f ne $target_db} { lappend link_dbs $f } }
}

puts "===== LIBRARIES ====="
puts "Target DB : $target_db"
puts "Link DBs  : [join $link_dbs { }]"
puts "=====================\n"

# DC variables
set_app_var search_path [concat $incdirs [getenv search_path ""] .]
set_app_var target_library $target_db
set_app_var link_library   [concat {*}{"*"} $link_dbs]

# ---------------- WORK library ----------------
if {![file exists WORK]} { file mkdir WORK }
define_design_lib WORK -path WORK

# ---------------- SystemVerilog knobs ----------------
catch { set hdlin_sv on }
catch { set hdlin_sv_2009 true }
catch { set_app_var hdlin_check_no_latch true }

# ---------------- analyze ----------------
# Check file existence first
foreach f $v_files {
  if {![file exists $f]} { WARN "Source file not found: $f" }
}

if {[llength $defines]} {
  puts "Analyzing (SV) with defines: $defines"
  analyze -format sverilog -work WORK -define $defines $v_files
} else {
  puts "Analyzing (SV)..."
  analyze -format sverilog -work WORK $v_files
}

# ---------------- elaborate / link ----------------
if {[catch {elaborate $TOP -work WORK} elaberr]} {
  DIE "Elaborate failed for '$TOP': $elaberr"
}
current_design $TOP
link

# BB_MODULES: comma/space-separated module names
set _bbmods [split [string map {, " "} [string trim $BB_MODULES]]]
foreach m $_bbmods {
  if {$m eq ""} continue
  set cells [get_cells -hier -filter "ref_name==$m" -quiet]
  if {[llength $cells]} {
    puts "INFO: Applying 'dont_touch' to cell $m"
    set_dont_touch $cells
  } else {
    puts "WARNING: Could not find cell '$m' to set 'dont_touch'."
  }
}

# Set operating conditions if available (library-dependent)
catch { set_operating_conditions TT }

# Pre-compile checks and uniquify
check_design > [file join $RPT_DIR "check_design.rpt"]
uniquify
link

# Optional: multi-core
catch { set_host_options -max_cores [getenv DC_CORES 8] }

# ---------------- constraints ----------------
if {$SDC_FILE ne "" && [file exists $SDC_FILE]} {
  puts "INFO: Loading SDC: $SDC_FILE"
  source $SDC_FILE
} else {
  WARN "No SDC provided/found; creating default 10ns clock if 'clk' exists"
  set clk_ports [get_ports clk -quiet]
  if {[sizeof_collection $clk_ports]} {
    create_clock -name clk -period 10.0 $clk_ports
    set_clock_uncertainty 0.05 [get_clocks clk]
    set_input_delay  0.2 -clock clk [remove_from_collection [all_inputs] $clk_ports]
    set_output_delay 0.2 -clock clk [all_outputs]
  } else {
    WARN "No 'clk' port to constrain; consider providing an SDC."
  }
}

# ---------------- pre-compile hygiene ----------------
set_fix_multiple_port_nets -all -buffer_constants
catch { set compile_seqmap_propagate_constants true }
catch { set power_enable_analysis true }

# ---------------- compile ----------------
compile_ultra -retime

# ---------------- reports ----------------
report_qor                          > [file join $RPT_DIR "qor.rpt"]
report_timing_summary               > [file join $RPT_DIR "timing_summary.rpt"]
report_timing -max_paths 10 -transition_time -capacitance -nets -input_pins > [file join $RPT_DIR "worst_setup.rpt"]
report_timing -delay_type min -max_paths 10 -nets -input_pins > [file join $RPT_DIR "worst_hold.rpt"]
check_timing                        > [file join $RPT_DIR "check_timing.rpt"]
report_area                         > [file join $RPT_DIR "area.rpt"]
report_area -hierarchy              > [file join $RPT_DIR "area_hier.rpt"]
report_power                        > [file join $RPT_DIR "power_vectorless.rpt"]
report_constraints -all_violators   > [file join $RPT_DIR "constraints_violators.rpt"]

# ---------------- outputs ----------------
write -format ddc     -hierarchy -output [file join $OUT_DIR "${TOP}.mapped.ddc"]
write -format verilog -hierarchy -output [file join $OUT_DIR "${TOP}.mapped.v"]
write_sdf                           [file join $OUT_DIR "${TOP}.mapped.sdf"]
catch { write_sdc                   [file join $OUT_DIR "${TOP}.post_compile.sdc"] }

puts "\nDONE. Top: $TOP"
exit
