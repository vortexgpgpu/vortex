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

# Start time
set start_time [clock seconds]

if { $::argc != 5 } {
  puts "ERROR: Program \"$::argv0\" requires 5 arguments!\n"
  puts "Usage: $::argv0 <top_module> <device_part> <vcs_file> <xdc_file> <tool_dir>\n"
  exit
}

# Set the project name
set project_name "project_1"

set top_module [lindex $::argv 0]
set device_part [lindex $::argv 1]
set vcs_file [lindex $::argv 2]
set xdc_file [lindex $::argv 3]
set tool_dir [lindex $::argv 4]

puts "Using top_module=$top_module"
puts "Using device_part=$device_part"
puts "Using vcs_file=$vcs_file"
puts "Using xdc_file=$xdc_file"
puts "Using tool_dir=$tool_dir"

# Set the number of jobs based on MAX_JOBS environment variable
if {[info exists ::env(MAX_JOBS)]} {
  set num_jobs $::env(MAX_JOBS)
  puts "using num_jobs=$num_jobs"
} else {
  set num_jobs 0
}

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

update_compile_order -fileset sources_1

set_property top $top_module [current_fileset]
set_property \
    -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} \
    -value {-mode out_of_context -flatten_hierarchy "rebuilt"} \
    -objects [get_runs synth_1]

# Synthesis
if {$num_jobs != 0} {
  launch_runs synth_1 -jobs $num_jobs
} else {
  launch_runs synth_1
}
wait_on_run synth_1
open_run synth_1
write_checkpoint -force post_synth.dcp
report_utilization -file utilization.rpt -hierarchical -hierarchical_percentages

# Implementation
if {$num_jobs != 0} {
  launch_runs impl_1 -jobs $num_jobs
} else {
  launch_runs impl_1
}
wait_on_run impl_1
open_run impl_1
write_checkpoint -force post_impl.dcp

# Generate the synthesis report
report_place_status -file place.rpt
report_route_status -file route.rpt
report_timing_summary -file timing.rpt
report_power -file power.rpt
report_drc -file drc.rpt

# End time and calculation
set elapsed_time [expr {[clock seconds] - $start_time}]

# Display elapsed time
set hours [format "%02d" [expr {$elapsed_time / 3600}]]
set minutes [format "%02d" [expr {($elapsed_time % 3600) / 60}]]
set seconds [format "%02d" [expr {$elapsed_time % 60}]]
puts "Total elapsed time: ${hours}h ${minutes}m ${seconds}s"