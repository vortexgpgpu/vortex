if { $::argc != 4 } {
    puts "ERROR: Program \"$::argv0\" requires 4 arguments!\n"
    puts "Usage: $::argv0 <xoname> <krnl_name> <vcs_file> <build_dir>\n"
    exit
}

set xoname    [lindex $::argv 0]
set krnl_name [lindex $::argv 1]
set vcs_file  [lindex $::argv 2]
set build_dir [lindex $::argv 3]

set script_path [ file dirname [ file normalize [ info script ] ] ]

if {[file exists "${xoname}"]} {
    file delete -force "${xoname}"
}

set argv [list ${build_dir}/ip]
set argc 1
source -notrace ${script_path}/gen_ip.tcl

set argv [list ${krnl_name} ${build_dir}]
set argc 2
source -notrace ${script_path}/package_kernel.tcl

package_xo -ctrl_protocol user_managed -xo_path ${xoname} -kernel_name ${krnl_name} -ip_directory "${build_dir}/packaged_kernel"
