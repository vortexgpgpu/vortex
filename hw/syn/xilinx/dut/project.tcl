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

  # Add constrains file
  read_xdc $xdc_file

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

proc run_report {} {
  # Generate the synthesis report
  report_place_status -file place.rpt
  report_route_status -file route.rpt

  # Generate timing report
  report_timing -nworst 100 -delay_type max -sort_by group -file timing.rpt

  # Generate power and drc reports
  report_power -file power.rpt
  report_drc -file drc.rpt
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
