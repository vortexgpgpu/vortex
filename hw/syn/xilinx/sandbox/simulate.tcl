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

if { $::argc != 2 } {
    puts "ERROR: Program \"$::argv0\" requires 2 arguments!\n"
    puts "Usage: $::argv0 <project> <simtime>\n"
    exit
}

set project_file [lindex $::argv 0]
set sim_time     [lindex $::argv 1]

set tb_name testbench       ;# Replace with actual testbench module

open_project $project_file   ;# Ensure correct project is loaded

# Ensure testbench is set as simulation top
set_property top $tb_name [get_filesets sim_1]

# Launch the simulation
launch_simulation -mode behavioral

# Run for the specified number of cycles
run $sim_time
