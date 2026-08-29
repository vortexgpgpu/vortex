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

if { $::argc != 4 } {
  puts "ERROR: Program \"$::argv0\" requires 4 arguments!\n"
  puts "Usage: $::argv0 <top_module> <device_part> <vcs_file> <xdc_file>\n"
  exit
}

# Set the project name
set project_name "project_1"

set top_module [lindex $::argv 0]
set device_part [lindex $::argv 1]
set vcs_file [lindex $::argv 2]
set xdc_file [lindex $::argv 3]

set tool_dir $::env(TOOL_DIR)
set script_dir [ file dirname [ file normalize [ info script ] ] ]

puts "Using top_module=$top_module"
puts "Using device_part=$device_part"
puts "Using vcs_file=$vcs_file"
puts "Using xdc_file=$xdc_file"
puts "Using tool_dir=$tool_dir"
puts "Using script_dir=$script_dir"

# Default the target clock frequency (MHz) when CLK_FREQ_MHZ is not set in the
# environment. Done here (real Tcl) because the XDC constraint parser rejects
# 'if'; child synth/impl runs inherit this env var and project.xdc reads it.
if {![info exists ::env(CLK_FREQ_MHZ)]} {
  set ::env(CLK_FREQ_MHZ) 300
}
puts "Using CLK_FREQ_MHZ=$::env(CLK_FREQ_MHZ)"

# Set the number of jobs based on MAX_JOBS environment variable
if {[info exists ::env(MAX_JOBS)]} {
  set num_jobs $::env(MAX_JOBS)
  puts "using num_jobs=$num_jobs"
} else {
  set num_jobs 0
}

proc run_setup {} {
  global project_name
  global top_module device_part vcs_file xdc_file
  global script_dir tool_dir
  global num_jobs
  global argv argc ;# Using global system variables: argv and argc

  # create fpu ip
  if {[info exists ::env(FPU_IP)]} {
    set ip_dir $::env(FPU_IP)
    set argv [list $ip_dir $device_part]
    set argc 2
    source ${tool_dir}/xilinx_ip_gen.tcl
  }

  # Module-interface registration gate for the full-system DUTs (VX_afu_wrap /
  # Vortex): run the same elaboration DRC the XRT gen_xo flow runs, so a top or
  # vortex synthesis fails fast on an unregistered SLR-crossing boundary rather
  # than after a multi-hour implementation. It stands up a throwaway -rtl
  # elaboration (the FPU IP is generated above and no real project is open yet)
  # and aborts the batch on a violation. Interior DUTs (fpu, cache, ...)
  # instantiate none of the checked boundary blocks, so the gate is scoped to the
  # two full-system tops. Severity via MODULE_INTERFACE_CHECK (default error).
  if {$top_module eq "VX_afu_wrap" || $top_module eq "Vortex"} {
    # Source at global scope (uplevel #0): xilinx_interface_check.tcl sets
    # MODULE_INTERFACE_BLOCKS as a global its check proc reads back, so it must
    # not land in this proc's local frame. krnl_name/build_dir are the inputs the
    # gate needs that are not already global (vcs_file/tool_dir already are). The
    # FPU IP is generated above and its project closed, so the gate's throwaway
    # -rtl elaboration opens cleanly.
    set ::krnl_name $top_module
    set ::build_dir $project_name
    uplevel #0 [list source "${tool_dir}/xilinx_elab_check.tcl"]
  }

  source "${tool_dir}/parse_vcs_list.tcl"
  set vlist [parse_vcs_list "${vcs_file}"]

  set vsources_list  [lindex $vlist 0]
  set vincludes_list [lindex $vlist 1]
  set vdefines_list  [lindex $vlist 2]

  #puts $vsources_list
  #puts $vincludes_list
  #puts $vdefines_list
  # Create project
  create_project $project_name $project_name -force -part $device_part

  # Use manual compile order so the explicit top module set below is honored.
  # In automatic-hierarchy mode Vivado can fail to validate a parameterized
  # top (e.g. VX_afu_wrap) and silently substitute a different root module.
  set_property source_mgmt_mode None [current_project]

  # Synthesis optimization level (OPT_LEVEL env, set by the Makefile):
  #   0    -- Flow_RuntimeOptimized: fastest compile, minimal optimization
  #   1, 2 -- Vivado defaults (no overrides)
  #   3    -- default: aggressive performance flow (Flow_PerfOptimized_High +
  #           Performance_ExtraTimingOpt + AggressiveExplore phys_opt)
  set opt_level 3
  if {[info exists ::env(OPT_LEVEL)]} { set opt_level $::env(OPT_LEVEL) }

  if {$opt_level == 0} {
    set_property strategy "Flow_RuntimeOptimized"      [get_runs synth_1]
    set_property strategy "Performance_RuntimeOptimized" [get_runs impl_1]
  } elseif {$opt_level >= 3} {
    # 1. Synthesis: Enable Retiming
    set_property strategy "Flow_PerfOptimized_High" [get_runs synth_1]
    set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]

    # 2. Implementation: Use ExtraTimingOpt
    set_property strategy "Performance_ExtraTimingOpt" [get_runs impl_1]

    # 3. Implementation: Force Physical Optimization steps
    set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
    set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]

    set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
    set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
  }
  # OPT_LEVEL 1 or 2: Vivado defaults (no strategy overrides).

  # Optional implementation-strategy override (IMPL_STRATEGY env), applied on
  # top of the OPT_LEVEL selection. Route-dominated designs may want a
  # congestion strategy (e.g. Congestion_SpreadLogic_high) instead of the
  # OPT_LEVEL=3 performance default.
  if {[info exists ::env(IMPL_STRATEGY)]} {
    set_property strategy $::env(IMPL_STRATEGY) [get_runs impl_1]
    puts "Using IMPL_STRATEGY=$::env(IMPL_STRATEGY)"
  }

  # Add constrains file
  read_xdc $xdc_file

  # Optional single-SLR floorplan (PBLOCK_SLR env, e.g. "SLR1"): pins the whole
  # DUT into one SLR. Multi-SLR splits put super-long-line crossings on
  # intra-subsystem paths (e.g. cache BRAMs in one SLR, their clients in
  # another), which no implementation strategy recovers.
  if {[info exists ::env(PBLOCK_SLR)]} {
    set slr $::env(PBLOCK_SLR)
    set pb_xdc [file join [pwd] "pblock_slr.xdc"]
    set fh [open $pb_xdc w]
    puts $fh "create_pblock pblock_dut"
    puts $fh "resize_pblock pblock_dut -add $slr"
    puts $fh "add_cells_to_pblock pblock_dut \[get_cells -quiet * -filter {IS_PRIMITIVE == 0}\]"
    close $fh
    read_xdc $pb_xdc
    puts "Using PBLOCK_SLR=$slr"
  }

  # Clock constraint: generated here with a literal period because the XDC
  # constraint parser sandbox rejects both 'if' and $::env(...) access. The
  # frequency is resolved above (CLK_FREQ_MHZ env, default 300) and baked in.
  set clk_period [expr {1000.0 / $::env(CLK_FREQ_MHZ)}]
  set clk_xdc "${project_name}/clock.xdc"
  set fh [open $clk_xdc w]
  puts $fh "create_clock -name core_clock -period $clk_period \[get_ports clk\]"
  close $fh
  read_xdc $clk_xdc
  puts "Clock constraint: core_clock period = ${clk_period} ns ($::env(CLK_FREQ_MHZ) MHz)"

  # Add the design sources
  add_files -norecurse -verbose $vsources_list

  # process defines
  set_property verilog_define ${vdefines_list} [current_fileset]

  # add fpu ip
  if {[info exists ::env(FPU_IP)]} {
    set ip_dir $::env(FPU_IP)
    add_files -norecurse -verbose ${ip_dir}/xil_fma/xil_fma.xci
    add_files -norecurse -verbose ${ip_dir}/xil_fdiv/xil_fdiv.xci
    add_files -norecurse -verbose ${ip_dir}/xil_fsqrt/xil_fsqrt.xci
    add_files -norecurse -verbose ${ip_dir}/xil_fmul/xil_fmul.xci
    add_files -norecurse -verbose ${ip_dir}/xil_fadd/xil_fadd.xci
  }

  # Synthesis
  set_property top $top_module [current_fileset]
  set_property \
      -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} \
      -value {-mode out_of_context} \
      -objects [get_runs synth_1]

  # register compilation hooks
  #set_property STEPS.SYNTH_DESIGN.TCL.PRE  ${script_dir}/pre_synth_hook.tcl  [get_runs synth_1]
  #set_property STEPS.SYNTH_DESIGN.TCL.POST ${script_dir}/post_synth_hook.tcl [get_runs synth_1]
  set_property STEPS.OPT_DESIGN.TCL.PRE ${script_dir}/pre_opt_hook.tcl [get_runs impl_1]
  #set_property STEPS.OPT_DESIGN.TCL.POST ${script_dir}/post_opt_hook.tcl   [get_runs impl_1]
  #set_property STEPS.POWER_OPT_DESIGN.TCL.PRE  ${script_dir}/pre_power_opt_hook.tcl  [get_runs impl_1]
  #set_property STEPS.POWER_OPT_DESIGN.TCL.POST ${script_dir}/post_power_opt_hook.tcl [get_runs impl_1]
  #set_property STEPS.PLACE_DESIGN.TCL.PRE  ${script_dir}/pre_place_hook.tcl  [get_runs impl_1]
  #set_property STEPS.PLACE_DESIGN.TCL.POST ${script_dir}/post_place_hook.tcl [get_runs impl_1]
  #set_property STEPS.POST_PLACE_POWER_OPT_DESIGN.TCL.PRE ${script_dir}/pre_place_power_opt_hook.tcl  [get_runs impl_1]
  #set_property STEPS.POST_PLACE_POWER_OPT_DESIGN.TCL.POST  ${script_dir}/post_place_power_opt_hook.tcl [get_runs impl_1]
  #set_property STEPS.PHYS_OPT_DESIGN.TCL.PRE ${script_dir}/pre_phys_opt_hook.tcl  [get_runs impl_1]
  #set_property STEPS.PHYS_OPT_DESIGN.TCL.POST  ${script_dir}/post_phys_opt_hook.tcl [get_runs impl_1]
  #set_property STEPS.ROUTE_DESIGN.TCL.PRE  ${script_dir}/pre_route_hook.tcl  [get_runs impl_1]
  #set_property STEPS.ROUTE_DESIGN.TCL.POST ${script_dir}/post_route_hook.tcl [get_runs impl_1]
  #set_property STEPS.WRITE_BITSTREAM.TCL.PRE ${script_dir}/pre_bitstream_hook.tcl  [get_runs impl_1]
  #set_property STEPS.WRITE_BITSTREAM.TCL.POST  ${script_dir}/post_bitstream_hook.tcl [get_runs impl_1]

  update_compile_order -fileset sources_1
}

proc run_synthesis {} {
  global num_jobs

  if {$num_jobs != 0} {
    launch_runs synth_1 -verbose -jobs $num_jobs
  } else {
    launch_runs synth_1 -verbose
  }
  wait_on_run synth_1
  open_run synth_1
  report_utilization -file post_synth_util.rpt -hierarchical -hierarchical_percentages
  write_checkpoint -force post_synth.dcp
}

proc run_implementation {} {
  global num_jobs

  if {$num_jobs != 0} {
    launch_runs impl_1 -verbose -jobs $num_jobs
  } else {
    launch_runs impl_1 -verbose
  }
  wait_on_run impl_1
  open_run impl_1
  report_utilization -file post_impl_util.rpt -hierarchical -hierarchical_percentages
  write_checkpoint -force post_impl.dcp
}

proc run_power_report {} {
  # ------------------------------------------------------------------
  # Vectorless baseline: always generated so results are reproducible
  # without a VCD.  Uses Vivado's internal activity model with resets
  # deasserted (representative of steady-state operation).
  # ------------------------------------------------------------------
  reset_switching_activity -all
  set_switching_activity -default_toggle_rate 0.125 -default_static_probability 0.5
  set_switching_activity -deassert_resets
  report_power -file power_vectorless.rpt
  puts "INFO: Vectorless power report -> power_vectorless.rpt"

  # ------------------------------------------------------------------
  # VCD-annotated power: only when VCD_FILE env var is provided.
  #
  # Switching activity is read from the VCD; signals absent from the
  # VCD fall back to Vivado's internal defaults.
  #
  # VCD_INST is the DUT instance path in the simulation hierarchy (mirrors
  # SAIF_INST in the Synopsys flow).  It is stripped from VCD signal paths
  # so they align with the synthesized netlist.
  # For Verilator-generated VCDs the scope is typically "TOP.<module>"
  # (e.g. VCD_INST=TOP.Vortex).  Leave empty if the VCD root scope
  # already matches the top module name.
  # ------------------------------------------------------------------
  if {[info exists ::env(VCD_FILE)] && $::env(VCD_FILE) ne ""} {
    set vcd_file $::env(VCD_FILE)
    if {![file exists $vcd_file]} {
      puts "WARNING: VCD_FILE not found: $vcd_file — skipping VCD-annotated power report"
      return
    }
    puts "INFO: Annotating switching activity from VCD: $vcd_file"
    reset_switching_activity -all
    if {[info exists ::env(VCD_INST)] && $::env(VCD_INST) ne ""} {
      read_vcd -strip_path $::env(VCD_INST) $vcd_file
    } else {
      read_vcd $vcd_file
    }
    # Keep resets deasserted — avoids inflating power with reset pulse
    set_switching_activity -deassert_resets
    report_power -file power_vcd.rpt
    puts "INFO: VCD-annotated power report -> power_vcd.rpt"
  }
}

proc run_report {} {
  # Generate the synthesis report
  report_place_status -file place.rpt
  report_route_status -file route.rpt

  # Generate methodology reports to check for any issues
  report_methodology -file methodology.rpt

  # Generate timing report
  report_timing -unique_pins -nworst 100 -delay_type max -sort_by group -file timing.rpt

  # Generate a high fanout net report
  report_high_fanout_nets -fanout_greater_than 100 -max_nets 50 -file high_fanout_nets.rpt

  # Generate clock utilization report to see register usage
  report_clock_utilization -file clock_utilization.rpt

  # Generate detailed RAM report
  report_ram_utilization -detail -file ram_utilization.rpt

  # Generate power report(s) and drc
  run_power_report
  report_drc -file drc.rpt

  # Consolidated machine-readable summary in one file (Fmax + LUT/LUTRAM/FF/BRAM/
  # URAM/DSP). This Vivado lacks report_qor_summary and report_utilization -format
  # csv, so it is built from the flat utilization report plus the worst setup path.
  if {[catch {
    proc _util_used {rpt label} {
      foreach line [split $rpt "\n"] {
        if {[regexp "\\|\\s*${label}\\s*\\|\\s*(\[0-9.\]+)" $line -> v]} { return $v }
      }
      return 0
    }
    set u      [report_utilization -return_string]
    set lut    [_util_used $u "CLB LUTs"]
    set lutram [_util_used $u "LUT as Memory"]
    set ff     [_util_used $u "CLB Registers"]
    set bram   [_util_used $u "Block RAM Tile"]
    set uram   [_util_used $u "URAM"]
    set dsp    [_util_used $u "DSPs"]
    set clk    [get_clocks -quiet core_clock]
    set period [get_property -quiet PERIOD $clk]
    set wns    [get_property -quiet SLACK [lindex [get_timing_paths -quiet -max_paths 1 -setup] 0]]
    set fmax   [expr {1000.0 / ($period - $wns)}]
    set fh [open synth_summary.csv w]
    puts $fh "design,period_ns,wns_ns,fmax_mhz,lut,lutram,ff,bram,uram,dsp"
    puts $fh [format "%s,%.3f,%.3f,%.1f,%g,%g,%g,%g,%g,%g" \
      [current_design] $period $wns $fmax $lut $lutram $ff $bram $uram $dsp]
    close $fh
    puts [format "INFO: synth_summary.csv  Fmax=%.1f MHz  WNS=%.3f ns  LUT=%g FF=%g BRAM=%g DSP=%g" \
      $fmax $wns $lut $ff $bram $dsp]
  } emsg]} {
    puts "WARNING: synth summary failed: $emsg"
  }
}

###############################################################################

# Start time
set start_time [clock seconds]

set checkpoint_synth "post_synth.dcp"
set checkpoint_impl "post_impl.dcp"

if { [file exists $checkpoint_impl] } {
  puts "Resuming from post-implementation checkpoint: $checkpoint_impl"
  open_checkpoint $checkpoint_impl
  run_report
} elseif { [file exists $checkpoint_synth] } {
  puts "Resuming from post-synthesis checkpoint: $checkpoint_synth"
  open_checkpoint $checkpoint_synth
  run_implementation
  run_report
} else {
  # Execute full pipeline
  run_setup
  run_synthesis
  run_implementation
  run_report
}

# End time and calculation
set elapsed_time [expr {[clock seconds] - $start_time}]

# Display elapsed time
set hours [format "%02d" [expr {$elapsed_time / 3600}]]
set minutes [format "%02d" [expr {($elapsed_time % 3600) / 60}]]
set seconds [format "%02d" [expr {$elapsed_time % 60}]]
puts "Total elapsed time: ${hours}h ${minutes}m ${seconds}s"
