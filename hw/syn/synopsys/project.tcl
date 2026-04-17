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
#   CLOCK_FREQ     : clock frequency in MHz (default: 800)
#   DELAY_UNC      : clock uncertainty in % (default: 2)
#   DELAY_IO       : clock I/O delay in % (default: 5)
#   SAIF_FILE      : path to SAIF file (optional)
#   SAIF_INST      : path to top module instance in SAIF file (optional)
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

# ------------------------- SRAM Wrapper Generation ------------------------- #
# Per (family, type) pin tables. Keys map role -> pin basename in the macro.
# Missing roles are treated as "no such pin on this macro" (not driven).
# For 2RW macros, roles are suffixed _A and _B for port-A / port-B.
set ::SRAM_PINS(SAED14_1RW) [dict create \
  CLK  CE  \
  CS_N CSB \
  WE_N WEB \
  OE_N OEB \
  ADDR A   \
  DIN  I   \
  DOUT O]
set ::SRAM_PINS(SAED14_2RW) [dict create \
  CLK_A  CE1  \
  CS_N_A CSB1 \
  WE_N_A WEB1 \
  OE_N_A OEB1 \
  ADDR_A A1   \
  DIN_A  I1   \
  DOUT_A O1   \
  CLK_B  CE2  \
  CS_N_B CSB2 \
  WE_N_B WEB2 \
  OE_N_B OEB2 \
  ADDR_B A2   \
  DIN_B  I2   \
  DOUT_B O2]
set ::SRAM_PINS(ASAP7_1RW) [dict create \
  CLK     clk     \
  WE      write   \
  RE      read    \
  ADDR    ADDRESS \
  BANKSEL banksel \
  SDEL    sdel    \
  DIN     wd      \
  DOUT    dataout]

# Classify an sram lib-cell name -> dict(family,type,depth,width,name)
proc sram_classify_cell {name} {
  # SAED14 / generic: SRAM1RW1024x32 or ram_2rw_128x64
  if {[regexp -nocase {(\w*(1rw|2rw)(\d+)x(\d+)\w*)} $name _ full kind depth width]} {
    return [dict create family SAED14 type [string toupper $kind] \
      depth $depth width $width name $full]
  }
  # ASAP7 banked: srambank_<ROWS>x<BANKS>x<WIDTH>_<tech>
  if {[regexp -nocase {srambank_(\d+)x(\d+)x(\d+)} $name _ rows banks width]} {
    return [dict create family ASAP7 type 1RW \
      depth [expr {$rows * $banks}] width $width name $name \
      rows $rows banks $banks]
  }
  return ""
}

# Introspect pins -> dict: basename -> dict(dir=<in|out|inout>, width=<N>)
# Bus bits like ADDRESS[3] are collapsed to base "ADDRESS"; width is the count.
proc sram_introspect_pins {lib_name cell_name} {
  set out [dict create]
  foreach_in_collection p [get_lib_pins -quiet "$lib_name/$cell_name/*"] {
    set pn  [get_attribute $p name]
    set dir [get_attribute $p pin_direction]
    regsub {\[\d+\]$} $pn {} base
    if {[dict exists $out $base]} {
      dict with out $base { incr width }
    } else {
      dict set out $base [dict create dir $dir width 1]
    }
  }
  return $out
}

# Verify all required roles in the (family,type) pin-table exist on the cell;
# DIE with full pin list on mismatch. Returns resolved dict(role -> pin_name).
proc sram_resolve_pins {lib_name cell_name family type pins_found} {
  set key "${family}_${type}"
  if {![info exists ::SRAM_PINS($key)]} {
    DIE "No SRAM pin table defined for '$key'"
  }
  set table $::SRAM_PINS($key)
  set resolved [dict create]
  set missing [list]
  dict for {role pin} $table {
    if {[dict exists $pins_found $pin]} {
      dict set resolved $role $pin
    } else {
      lappend missing "$role=$pin"
    }
  }
  if {[llength $missing]} {
    puts stderr "---- Actual pins on $lib_name/$cell_name ----"
    dict for {pn info} $pins_found {
      puts stderr [format "  %-16s dir=%s width=%d" \
        $pn [dict get $info dir] [dict get $info width]]
    }
    DIE "SRAM pin mismatch on $cell_name ($key): missing [join $missing {, }]. Update ::SRAM_PINS($key) in project.tcl."
  }
  return $resolved
}

# Pin-width lookup helper
proc sram_pin_width {pins_found pname} {
  return [dict get $pins_found $pname width]
}

# Emit one 1RW instance block. $R = role->pinname dict.
# Family dispatch: SAED14 uses active-low controls + single flat ADDR;
# ASAP7 srambank uses active-high controls, separate banksel, and sdel tie-off.
proc sram_emit_sp_inst {fh m R} {
  set nm     [dict get $m name]
  set fam    [dict get $m family]
  switch -- $fam {
    ASAP7 {
      # Use ACTUAL macro bus widths (not rows/banks from name-parsing)
      set row_w [dict get $m pin_w_ADDR]
      set bs_w  [dict get $m pin_w_BANKSEL]
      # Split internal_addr: upper bs_w bits -> banksel, lower row_w bits -> ADDRESS
      puts $fh "            wire \[[expr {$bs_w-1}]:0\]  i_banksel = internal_addr\[[expr {$row_w+$bs_w-1}]:$row_w\];"
      puts $fh "            wire \[[expr {$row_w-1}]:0\] i_row     = internal_addr\[[expr {$row_w-1}]:0\];"
      puts $fh "            $nm u_mem ("
      puts $fh "                .[dict get $R CLK]     ( clk ),"
      puts $fh "                .[dict get $R WE]      ( write ),"
      puts $fh "                .[dict get $R RE]      ( read ),"
      puts $fh "                .[dict get $R ADDR]    ( i_row ),"
      puts $fh "                .[dict get $R BANKSEL] ( i_banksel ),"
      puts $fh "                .[dict get $R SDEL]    ( [dict get $m pin_w_SDEL]'b0 ),"
      puts $fh "                .[dict get $R DIN]     ( wdata ),"
      puts $fh "                .[dict get $R DOUT]    ( rdata )"
      puts $fh "            );"
    }
    default {
      # SAED14 / generic 1RW
      puts $fh "            $nm u_mem ("
      puts $fh "                .[dict get $R CLK]  ( clk ),"
      if {[dict exists $R CS_N]} { puts $fh "                .[dict get $R CS_N] ( cs_n )," }
      if {[dict exists $R CE_N]} { puts $fh "                .[dict get $R CE_N] ( ce_n )," }
      puts $fh "                .[dict get $R WE_N] ( write_n ),"
      if {[dict exists $R OE_N]} { puts $fh "                .[dict get $R OE_N] ( read_n )," }
      puts $fh "                .[dict get $R ADDR] ( internal_addr ),"
      puts $fh "                .[dict get $R DIN]  ( wdata ),"
      puts $fh "                .[dict get $R DOUT] ( rdata )"
      puts $fh "            );"
    }
  }
}

# Emit one 2RW instance block. Port A = write, Port B = read.
# Currently only SAED14_2RW is supported; other families can extend here.
proc sram_emit_dp_inst {fh m R} {
  set nm  [dict get $m name]
  set fam [dict get $m family]
  set w   [dict get $m width]
  switch -- $fam {
    SAED14 {
      puts $fh "            $nm u_mem ("
      puts $fh "                .[dict get $R CLK_A]  ( clk ),"
      puts $fh "                .[dict get $R CS_N_A] ( 1'b0 ),"
      puts $fh "                .[dict get $R WE_N_A] ( we_a_n ),"
      puts $fh "                .[dict get $R OE_N_A] ( 1'b1 ),"
      puts $fh "                .[dict get $R ADDR_A] ( a1_addr ),"
      puts $fh "                .[dict get $R DIN_A]  ( wdata ),"
      puts $fh "                .[dict get $R DOUT_A] ( ),"
      puts $fh "                .[dict get $R CLK_B]  ( clk ),"
      puts $fh "                .[dict get $R CS_N_B] ( 1'b0 ),"
      puts $fh "                .[dict get $R WE_N_B] ( 1'b1 ),"
      puts $fh "                .[dict get $R OE_N_B] ( ce_b_n ),"
      puts $fh "                .[dict get $R ADDR_B] ( a2_addr ),"
      puts $fh "                .[dict get $R DIN_B]  ( ${w}'b0 ),"
      puts $fh "                .[dict get $R DOUT_B] ( rdata )"
      puts $fh "            );"
    }
    default {
      DIE "sram_emit_dp_inst: no 2RW emitter for family '$fam'"
    }
  }
}

# Generate VX_sp_ram_asic.v / VX_dp_ram_asic.v by introspecting loaded mem libs.
# Returns list of generated file paths. Only 1RW is emitted for SP; 2RW for DP.
# DP wrapper for ASAP7 is not emitted (srambank is single-port).
proc gen_sram_wrappers {out_dir} {
  file mkdir $out_dir
  set sp_mems [list]
  set dp_mems [list]
  set all_resolved [dict create]  ;# cell_name -> role dict

  foreach_in_collection lib [get_libs -quiet *] {
    set lib_name [get_attribute $lib name]
    foreach_in_collection cell [get_lib_cells -quiet "$lib_name/*"] {
      set cname [get_attribute $cell name]
      set info [sram_classify_cell $cname]
      if {$info eq ""} { continue }
      set pins_found [sram_introspect_pins $lib_name $cname]
      set resolved   [sram_resolve_pins $lib_name $cname \
                        [dict get $info family] [dict get $info type] \
                        $pins_found]
      dict set all_resolved $cname $resolved
      # Attach actual macro pin widths (for slicing internal_addr, etc.)
      dict for {role pin} $resolved {
        dict set info pin_w_$role [sram_pin_width $pins_found $pin]
      }
      if {[dict get $info type] eq "1RW"} {
        lappend sp_mems $info
      } elseif {[dict get $info type] eq "2RW"} {
        lappend dp_mems $info
      }
    }
  }

  if {[llength $sp_mems] == 0 && [llength $dp_mems] == 0} {
    DIE "gen_sram_wrappers: no SRAM lib cells found in loaded libraries"
  }

  # Sort by (width, depth) for deterministic output
  set sp_mems [lsort -command {apply {{a b} {
    set wa [dict get $a width]; set wb [dict get $b width]
    if {$wa != $wb} { return [expr {$wa - $wb}] }
    return [expr {[dict get $a depth] - [dict get $b depth]}]
  }}} $sp_mems]

  set generated [list]

  # --- SP wrapper ---
  if {[llength $sp_mems] > 0} {
    set fp [file join $out_dir VX_sp_ram_asic.v]
    set fh [open $fp w]
    puts $fh "// AUTOMATICALLY GENERATED by project.tcl (gen_sram_wrappers)"
    puts $fh "(* keep_hierarchy = \"yes\" *)"
    puts $fh "module VX_sp_ram_asic #("
    puts $fh "    parameter DATAW = 32,"
    puts $fh "    parameter SIZE  = 1024,"
    puts $fh "    parameter WRENW = 1,"
    puts $fh "    parameter ADDRW = \$clog2(SIZE)"
    puts $fh ") ("
    puts $fh "    input  wire             clk,"
    puts $fh "    input  wire             reset,"
    puts $fh "    input  wire             read,"
    puts $fh "    input  wire             write,"
    puts $fh "    input  wire \[WRENW-1:0\] wren,"
    puts $fh "    input  wire \[ADDRW-1:0\] addr,"
    puts $fh "    input  wire \[DATAW-1:0\] wdata,"
    puts $fh "    output wire \[DATAW-1:0\] rdata"
    puts $fh ");"
    puts $fh "    wire write_n = ~write;"
    puts $fh "    wire read_n  = ~read;"
    puts $fh "    wire cs_n    = 1'b0;"
    puts $fh "    wire ce_n    = ~(read | write);"
    puts $fh ""
    puts $fh "    generate"
    set first 1
    foreach m $sp_mems {
      set w [dict get $m width]; set d [dict get $m depth]
      set nm [dict get $m name]
      set prefix [expr {$first ? "if" : "else if"}]; set first 0
      # ASAP7: physical address = row_bits + bank_sel_bits (per actual macro pins).
      # Generic: ceil(log2(depth)).
      if {[dict get $m family] eq "ASAP7"} {
        set phy_aw [expr {[dict get $m pin_w_ADDR] + [dict get $m pin_w_BANKSEL]}]
      } else {
        set phy_aw [expr {max(1, int(ceil(log($d)/log(2))))}]
        if {$d == 1} { set phy_aw 1 }
      }
      puts $fh "        $prefix (DATAW == $w && SIZE <= $d) begin : g_$nm"
      puts $fh "            wire \[[expr {$phy_aw-1}]:0\] internal_addr;"
      puts $fh "            assign internal_addr = (ADDRW < $phy_aw) ?"
      puts $fh "                { {($phy_aw-ADDRW){1'b0}}, addr } : addr\[[expr {$phy_aw-1}]:0\];"
      set R [dict get $all_resolved $nm]
      sram_emit_sp_inst $fh $m $R
      puts $fh "        end"
    }
    puts $fh "    endgenerate"
    puts $fh "endmodule"
    close $fh
    puts "INFO: gen_sram_wrappers: wrote $fp ([llength $sp_mems] 1RW macros)"
    lappend generated $fp
  }

  # --- DP wrapper (SAED14-style only; ASAP7 srambank is 1RW) ---
  if {[llength $dp_mems] > 0} {
    set dp_mems [lsort -command {apply {{a b} {
      set wa [dict get $a width]; set wb [dict get $b width]
      if {$wa != $wb} { return [expr {$wa - $wb}] }
      return [expr {[dict get $a depth] - [dict get $b depth]}]
    }}} $dp_mems]
    set fp [file join $out_dir VX_dp_ram_asic.v]
    set fh [open $fp w]
    puts $fh "// AUTOMATICALLY GENERATED by project.tcl (gen_sram_wrappers)"
    puts $fh "(* keep_hierarchy = \"yes\" *)"
    puts $fh "module VX_dp_ram_asic #("
    puts $fh "    parameter DATAW = 32,"
    puts $fh "    parameter SIZE  = 1024,"
    puts $fh "    parameter WRENW = 1,"
    puts $fh "    parameter ADDRW = \$clog2(SIZE)"
    puts $fh ") ("
    puts $fh "    input  wire             clk,"
    puts $fh "    input  wire             reset,"
    puts $fh "    input  wire             read,"
    puts $fh "    input  wire             write,"
    puts $fh "    input  wire \[WRENW-1:0\] wren,"
    puts $fh "    input  wire \[ADDRW-1:0\] waddr,"
    puts $fh "    input  wire \[DATAW-1:0\] wdata,"
    puts $fh "    input  wire \[ADDRW-1:0\] raddr,"
    puts $fh "    output wire \[DATAW-1:0\] rdata"
    puts $fh ");"
    puts $fh "    wire we_a_n = ~write;"
    puts $fh "    wire ce_b_n = ~read;"
    puts $fh ""
    puts $fh "    generate"
    set first 1
    foreach m $dp_mems {
      set w [dict get $m width]; set d [dict get $m depth]
      set nm [dict get $m name]
      set prefix [expr {$first ? "if" : "else if"}]; set first 0
      set phy_aw [expr {max(1, int(ceil(log($d)/log(2))))}]
      puts $fh "        $prefix (DATAW == $w && SIZE <= $d) begin : g_$nm"
      puts $fh "            wire \[[expr {$phy_aw-1}]:0\] a1_addr;"
      puts $fh "            assign a1_addr = (ADDRW < $phy_aw) ?"
      puts $fh "                { {($phy_aw-ADDRW){1'b0}}, waddr } : waddr\[[expr {$phy_aw-1}]:0\];"
      puts $fh "            wire \[[expr {$phy_aw-1}]:0\] a2_addr;"
      puts $fh "            assign a2_addr = (ADDRW < $phy_aw) ?"
      puts $fh "                { {($phy_aw-ADDRW){1'b0}}, raddr } : raddr\[[expr {$phy_aw-1}]:0\];"
      set R [dict get $all_resolved $nm]
      sram_emit_dp_inst $fh $m $R
      puts $fh "        end"
    }
    puts $fh "    endgenerate"
    puts $fh "endmodule"
    close $fh
    puts "INFO: gen_sram_wrappers: wrote $fp ([llength $dp_mems] 2RW macros)"
    lappend generated $fp
  }

  return $generated
}

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
set CLOCK_FREQ   [getenv CLOCK_FREQ  800]
set DELAY_UNC    [getenv DELAY_UNC  0.02]
set DELAY_IO     [getenv DELAY_IO   0.05]
set SAIF_FILE    [getenv SAIF_FILE    ""]
set SAIF_INST    [getenv SAIF_INST    ""]

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

# ---------------- SRAM wrapper generation (from loaded mem .db) ----------------
if {$use_mem_libs} {
  foreach db $gen_dbs {
    if {[catch {read_db $db} err]} {
      DIE "read_db failed on $db: $err"
    }
  }
  set sram_out_dir [file normalize "./src"]
  set sram_files [gen_sram_wrappers $sram_out_dir]
  # Prepend so they're analyzed before consumers
  set v_files [concat $v_files $sram_files]
}

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

# Capture report_units output into a Tcl variable
set _ru ""
catch { redirect -variable _ru { report_units } }

# Parse ONLY the Time_unit line, extract token in parentheses at end of that line.
# Example line: "Time_unit            : 1.0e-09 Second(ns)"
set LIB_TIME_UNIT "ns"
if {[regexp -line {^Time_unit\s*:\s*.*\(([^)]+)\)\s*$} $_ru -> LIB_TIME_UNIT]} {
  set LIB_TIME_UNIT [string trim $LIB_TIME_UNIT]
} else {
  puts "WARN: Failed to parse Time_unit from report_units; assuming ns"
  set LIB_TIME_UNIT "ns"
}

# Compute scale: how many <LIB_TIME_UNIT> per 1 ns
#   ns -> 1
#   ps -> 1000
#   fs -> 1000000
set NS_TO_LIB 1.0
switch -exact -- $LIB_TIME_UNIT {
  ns { set NS_TO_LIB 1.0 }
  ps { set NS_TO_LIB 1000.0 }
  fs { set NS_TO_LIB 1000000.0 }
  default {
    # Unknown unit: be safe and assume ns, but warn loudly
    set NS_TO_LIB 1.0
    puts "WARN: Unknown library time unit '$LIB_TIME_UNIT' from report_units; assuming ns"
  }
}

# Period in ns (always correct, independent of library units)
set period_ns [expr 1000.0 / double($CLOCK_FREQ)]  ;# ns

# Convert ns to library time units for all time-based constraints
set target_period      [expr $period_ns * $NS_TO_LIB]
set target_uncertainty [expr $target_period * $DELAY_UNC]
set target_io_delay    [expr $target_period * $DELAY_IO]

puts "----------------------------------------------------------------"
puts "INFO: Target Frequency     : $CLOCK_FREQ MHz"
puts "INFO: Period (ns)          : $period_ns ns"
puts "INFO: Library time unit    : $LIB_TIME_UNIT"
puts "INFO: Period (lib units)   : $target_period $LIB_TIME_UNIT"
puts "INFO: Uncertainty (lib)    : $target_uncertainty $LIB_TIME_UNIT"
puts "INFO: I/O Delay (lib)      : $target_io_delay $LIB_TIME_UNIT"
puts "----------------------------------------------------------------"

# ---------------- constraints ----------------

if {$SDC_FILE ne "" && [file exists $SDC_FILE]} {
  puts "INFO: Loading SDC: $SDC_FILE"
  source $SDC_FILE
} else {
  DIE "No SDC file provided"
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
report_timing -delay_type max -path_type full_clock -max_paths 50 -nets -transition_time -capacitance > [file join $RPT_DIR "timing_max.rpt"]
report_timing -delay_type min -path_type full_clock -max_paths 50 -nets -transition_time -capacitance > [file join $RPT_DIR "timing_min.rpt"]
report_clock -skew                              > [file join $RPT_DIR "clock_skew.rpt"]
report_constraints -all_violators               > [file join $RPT_DIR "constraints_violators.rpt"]

# ------------ power evaluation -----------
if {$SAIF_FILE ne "" && [file exists $SAIF_FILE]} {
  if {$SAIF_INST eq ""} {
    puts "INFO: Loading SAIF activity: $SAIF_FILE"
    WARN "SAIF_INST not set. Using current_design scope (may mis-map if SAIF has tb hierarchy)."
    read_saif -input $SAIF_FILE -auto_map_names -verbose
  } else {
    puts "INFO: Loading SAIF activity: $SAIF_FILE for instance: $SAIF_INST"
    read_saif -input $SAIF_FILE -instance_name $SAIF_INST -auto_map_names -verbose
  }

  # Update the power model with the new toggle rates
  update_power

  # Generate a hierarchical power report based on the vector data
  report_power -hierarchy > [file join $RPT_DIR "power_active.rpt"]

  # Report how much of the design was successfully annotated by the SAIF file
  report_saif > [file join $RPT_DIR "saif_annotation_coverage.rpt"]
} else {
  puts "WARN: SAIF_FILE not set or not found. Falling back to vectorless power estimation."
  report_power > [file join $RPT_DIR "power_vectorless.rpt"]
}

# ---------------- outputs ----------------
write -format ddc     -hierarchy -output [file join $OUT_DIR "${TOP}.mapped.ddc"]
write -format verilog -hierarchy -output [file join $OUT_DIR "${TOP}.mapped.v"]
write_sdf [file join $OUT_DIR "${TOP}.mapped.sdf"]
catch { write_sdc [file join $OUT_DIR "${TOP}.post_compile.sdc"] }

puts "\nDONE. Top: $TOP"
exit 0
