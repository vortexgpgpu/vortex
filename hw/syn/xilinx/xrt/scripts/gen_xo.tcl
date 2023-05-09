if { $::argc != 5 } {
    puts "ERROR: Program \"$::argv0\" requires 4 arguments!\n"
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
source ${script_path}/gen_ip.tcl

set argv [list ${krnl_name} ${vcs_file} ${tool_dir} ${build_dir}]
set argc 4
source ${script_path}/package_kernel.tcl

package_xo -xo_path ${xoname} -kernel_name ${krnl_name} -ip_directory "${build_dir}/xo/packaged_kernel"
