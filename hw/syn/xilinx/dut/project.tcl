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

#puts top_module
#puts $device_part
#puts $vcs_file
#puts xdc_file
#puts $tool_dir

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
set obj [current_fileset]
foreach def $vdefines_list {
  set_property verilog_define $def $obj
}

# Synthesis
synth_design -top $top_module -include_dirs $vincludes_list -flatten_hierarchy none
write_checkpoint -force post_synth.dcp
report_utilization -file utilization.rpt -hierarchical -hierarchical_percentages

# Optimize
opt_design

# Place
place_design
write_checkpoint -force post_place.dcp
report_place_status -file place.rpt

# Route
route_design
write_checkpoint -force post_route.dcp
report_route_status -file route.rpt

# Generate the synthesis report
report_timing -file timing.rpt
report_power -file power.rpt
report_drc -file drc.rpt