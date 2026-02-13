###############################################################################
# Synopsys DC — Generic Synthesis (SystemVerilog) using .db libraries
# Inputs via environment variables:
#   TOP            : top module name (required)
#   SRC_FILE       : path to sources.txt (VCS-style +incdir/+define) (required)
#   LIB_ROOT       : root folder to recursively discover *.db (or)
#   LIB_TGT        : explicit target .db (optional)
#   SDC_FILE       : constraints file (optional)
#   BB_MODULES     : comma-separated list of modules to set as dont_touch (optional)
#   MEM_LIBS       : path/wildcard to generated memory .db files (optional)
#                    If set, BB_MODULES logic is skipped for RAMs.
#   WALL_IGNORE    : comma-separated list of wildcard paths to exempt from strict checks
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
set SRC_FILE     [getenv SRC_FILE     ""]
set SDC_FILE     [getenv SDC_FILE     ""]
set LIB_ROOT     [getenv LIB_ROOT     ""]
set LIB_TGT_HINT [getenv LIB_TGT      ""]
set BB_MODULES   [getenv BB_MODULES   ""]
set MEM_LIBS     [getenv MEM_LIBS     ""]
set WALL_IGNORE  [getenv WALL_IGNORE  ""]
set TOOL_DIR     [getenv TOOL_DIR     ""]
set OUT_DIR      [file normalize [getenv OUT_DIR "./out"]]
set RPT_DIR      [file normalize [getenv RPT_DIR "./reports"]]

# Helper lists for port discovery
set SRAM_W_PORTS {wdata rdata}
set SRAM_A_PORTS {addr waddr raddr}

# Change these to calibrate SRAM estimation
set SRAM_BIT_AREA 0.1   ; # um^2 per bit
set SRAM_OH_AREA  100.0 ; # um^2 overhead (decoders, sense amps)

# Validate environment
if {$TOP eq ""}          { DIE "TOP not set" }
if {$SRC_FILE eq ""}     { DIE "SRC_FILE not set" }
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
puts "Incdirs     : [join $incdirs "\n"]"
puts "Defines     : [join $defines " "]"
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
  if {[llength $dbs] == 0} { DIE "No .db files found under $LIB_ROOT" }

  set std_dbs [filter_stdcell $dbs]
  if {[llength $std_dbs] == 0} { set std_dbs $dbs }

  # We must use ALL files that match the target corner (TT)
  # instead of picking just one.
  set target_db [list]
  foreach f $std_dbs {
    # Check if file is typical corner (TT)
    if {[classify_corner $f] eq "TT"} {
       lappend target_db $f
    }
  }

  # Failsafe: if no TT found, grab everything standard cell related
  if {[llength $target_db] == 0} {
     puts "WARN: No TT corner files found. Using all detected DBs as target."
     set target_db $std_dbs
  }

  # Link DBs matches Target DBs for std cells
  set link_dbs $target_db
}

set use_mem_libs 0
if {$MEM_LIBS ne ""} {
  set gen_dbs [glob -nocomplain $MEM_LIBS]
  if {[llength $gen_dbs] > 0} {
    puts "INFO: Pre-loading Memory Libraries: $gen_dbs"
    lappend link_dbs {*}$gen_dbs
    set use_mem_libs 1
  }
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
catch { set_app_var hdlin_keep_signal_name user }

# ---------------- STRICT MODE & WALL_IGNORE ----------------
# 1. Force stop on error
set_app_var sh_continue_on_error false

# 2. Promote CRITICAL warnings to errors
set strict_warning_ids {
  ELAB-311 ELAB-307 ELAB-909
  VER-104 VER-130 VER-129 VER-225 VER-708 VER-936 VER-941 VER-1005
  LINK-5 LINK-19
  UID-401 UID-101
}
foreach msg_id $strict_warning_ids {
  set_message_info -id $msg_id -stop_on
}

# 3. Parse ignore list (comma separated wildcards)
set wall_ignore_patterns [list]
if {$WALL_IGNORE ne ""} {
  foreach p [split $WALL_IGNORE ","] {
    lappend wall_ignore_patterns [string trim $p]
  }
}

# ---------------- analyze (File-by-File) ----------------
puts "Starting Analysis (Strict Mode Enabled)..."

foreach f $v_files {
  if {![file exists $f]} { DIE "Source file not found: $f" }

  # Check if this file matches WALL_IGNORE
  set is_exempt 0
  foreach pattern $wall_ignore_patterns {
    if {[string match $pattern $f] || [string match $pattern [file tail $f]]} {
      set is_exempt 1
      break
    }
  }

  if {$is_exempt} {
    puts "INFO: Exempting file from strict checks: $f"
    set_app_var sh_continue_on_error true
    foreach msg_id $strict_warning_ids {
      set_message_info -id $msg_id -stop_off
    }
  }

  # Analyze single file
  if {[catch {
    if {[llength $defines]} {
      analyze -format sverilog -work WORK -define $defines $f
    } else {
      analyze -format sverilog -work WORK $f
    }
  } err_msg]} {
    if {!$is_exempt} {
      DIE "STRICT ANALYSIS FAILED on $f:\n$err_msg"
    } else {
      puts "WARN: Error in exempt file $f (Ignored due to WALL_IGNORE): $err_msg"
    }
  }

  # Immediately restore Strict Mode
  if {$is_exempt} {
    set_app_var sh_continue_on_error false
    foreach msg_id $strict_warning_ids {
      set_message_info -id $msg_id -stop_on
    }
  }
}

# ---------------- elaborate ----------------
# Elaborate is critical; typically we want this to strict fail,
# but if previous analysis passed with exemptions, this might trigger link errors.
if {[catch {elaborate $TOP -work WORK} elaberr]} {
  DIE "Elaborate failed for '$TOP': $elaberr"
}
current_design $TOP

# Uniquify BEFORE linking or setting blackboxes.
uniquify
link

# ---------------- Memory / Blackbox Logic ----------------

if {$use_mem_libs} {
  # ---------------------------------------
  # FLOW A: Real Macro Flow (Generated DBs)
  # ---------------------------------------
  puts "\nINFO: MEM_LIBS detected. Real-Macro flow active."

} else {
  # ---------------------------------------
  # FLOW B: Blackbox Flow with Estimation
  # ---------------------------------------
  puts "\nINFO: MEM_LIBS not set or empty. Defaulting to Blackbox Flow with Area Estimation."

  set _bbmods [split [string map {, " "} [string trim $BB_MODULES]]]
  set total_sram_area 0.0

  if {[llength $_bbmods] > 0} {
    foreach m $_bbmods {
      if {$m eq ""} continue

      set cells [get_cells -hier -filter "ref_name=~${m}*" -quiet]

      if {[llength $cells]} {
        puts "INFO: Processing blackbox: $m"
        set_dont_touch $cells

        set ref_designs [get_designs -quiet -filter "original_design_name==$m || name=~${m}*"]

        foreach_in_collection des $ref_designs {
          set d_name [get_object_name $des]

          set w [get_attribute $des "DATAW" -quiet]
          set d [get_attribute $des "SIZE" -quiet]

          if {$w eq "" || $d eq ""} {
            set des_ports [get_ports -quiet -of_objects $des]
            foreach p $SRAM_W_PORTS {
              set p_obj [filter_collection $des_ports "name == $p"]
              if {[sizeof_collection $p_obj] > 0} {
                set w [get_attribute $p_obj bit_width -quiet]
                if {$w ne ""} { break }
              }
            }
            foreach p $SRAM_A_PORTS {
              set p_obj [filter_collection $des_ports "name == $p"]
              if {[sizeof_collection $p_obj] > 0} {
                set aw [get_attribute $p_obj bit_width -quiet]
                if {$aw ne ""} { set d [expr {1 << $aw}]; break }
              }
            }
            if {$w eq "" || $d eq ""} {
              if {[regexp {DATAW(\d+)} $d_name match val]} { set w $val }
              if {[regexp {SIZE(\d+)}  $d_name match val]} { set d $val }
            }
          }

          if {$w ne "" && $d ne ""} {
            set total_bits [expr $w * $d]
            set est_area [expr ($total_bits * $SRAM_BIT_AREA) + $SRAM_OH_AREA]
            set total_sram_area [expr $total_sram_area + $est_area]

            if {[catch { set_attribute $des area $est_area } err]} {
              set saved_design [current_design]
              current_design $des
              catch { set_attribute [current_design] area $est_area }
              current_design $saved_design
            }
            puts "      > Estimated Area for $d_name ($w x $d): $est_area"
          } else {
            puts "      > WARN: Could not derive DATAW/SIZE for $d_name. Area will be 0."
          }
        }
      } else {
        puts "WARNING: Could not find cell instances for '$m'."
      }
    }
  } else {
    puts "INFO: BB_MODULES is empty. No blackbox attributes or estimation will be applied."
  }

  if {$total_sram_area > 0} {
    puts "------------------------------------------------"
    puts "INFO: Total Estimated SRAM Area: $total_sram_area"
    puts "------------------------------------------------"
  }
}

# ---------------- Pre-Compile Checks ----------------

check_design > [file join $RPT_DIR "check_design.rpt"]

# Optional: multi-core
catch { set_host_options -max_cores [getenv DC_CORES 8] }

# ------------------- Clock Setup --------------------

set NS 1.0
if {[catch {set_units -time ns -resistance kOhm -capacitance pF -voltage V -current mA}]} {
  WARN "Library locked to Picoseconds. Scaling constraints by 1000."
  set NS 1000.0
}

set CLOCK_FREQ [getenv CLOCK 800]
set period_ns [expr 1000.0 / $CLOCK_FREQ]
set target_period [expr $period_ns * $NS]
set target_uncertainty [expr $target_period * 0.10] ;# 10% Jitter/Skew margin
set target_io_delay    [expr $target_period * 0.15] ;# 15% External I/O Delay

puts "----------------------------------------------------------------"
puts "INFO: Target Frequency : $CLOCK_FREQ MHz"
puts "INFO: Library Scaling  : x$NS"
puts "INFO: Clock Period     : $target_period"
puts "INFO: Clock Uncertainty: $target_uncertainty"
puts "INFO: I/O Delay        : $target_io_delay"
puts "----------------------------------------------------------------"

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
if {[catch {compile_ultra -retime} compile_err]} {
  DIE "Synthesis (compile_ultra) Failed: $compile_err"
}
# If OPT-101 occurs, DC leaves generic "GTECH" cells. We must catch this.
set unmapped_cells [get_cells -hierarchical -filter "ref_name =~ *GTECH*"]
if {[sizeof_collection $unmapped_cells] > 0} {
  DIE "FATAL: Synthesis incomplete. Found [sizeof_collection $unmapped_cells] unmapped GTECH cells."
}

# ---------------- reports ----------------
report_qor                                      > [file join $RPT_DIR "qor.rpt"]
report_area -hier -nosplit                      > [file join $RPT_DIR "area.rpt"]
report_timing -delay_type max -path_type full_clock -max_paths 50  > [file join $RPT_DIR "timing_max.rpt"]
report_timing -delay_type min -path_type full_clock -max_paths 50  > [file join $RPT_DIR "timing_min.rpt"]
report_power                                    > [file join $RPT_DIR "power_vectorless.rpt"]
report_constraints -all_violators               > [file join $RPT_DIR "constraints_violators.rpt"]

# ---------------- outputs ----------------
write -format ddc     -hierarchy -output [file join $OUT_DIR "${TOP}.mapped.ddc"]
write -format verilog -hierarchy -output [file join $OUT_DIR "${TOP}.mapped.v"]
write_sdf                           [file join $OUT_DIR "${TOP}.mapped.sdf"]
catch { write_sdc                   [file join $OUT_DIR "${TOP}.post_compile.sdc"] }

puts "\nDONE. Top: $TOP"
exit 0
