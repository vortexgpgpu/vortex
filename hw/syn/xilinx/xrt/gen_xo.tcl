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
    puts "Usage: $::argv0 <xoname> <krnl_name> <vcs_file> <tool_dir> <build_dir>\n"
    exit
}

set xoname    [lindex $::argv 0]
set krnl_name [lindex $::argv 1]
set vcs_file  [lindex $::argv 2]
set tool_dir  [lindex $::argv 3]
set build_dir [lindex $::argv 4]

set script_path [ file dirname [ file normalize [ info script ] ] ]

if {[file exists "${xoname}"]} {
    file delete -force "${xoname}"
}

set argv [list ${build_dir}/ip]
set argc 1
source ${tool_dir}/xilinx_ip_gen.tcl

set argv [list ${krnl_name} ${vcs_file} ${tool_dir} ${build_dir}]
set argc 4
source ${script_path}/package_kernel.tcl

package_xo -xo_path ${xoname} -kernel_name ${krnl_name} -ip_directory "${build_dir}/xo/packaged_kernel"
