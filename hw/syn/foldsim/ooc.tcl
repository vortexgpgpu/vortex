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

# The foldsim kernel alone, out of context: synthesis with global retiming,
# placement and routing at the kernel clock, as the image's own run times it
# but without the shell around it. Reports land in the current directory.
#
#   vivado -mode batch -source ooc.tcl -tclargs <kernel_srcs.f> <part> <period_ns> [no-vendor-ip]
#
# The Xilinx floating-point cores a simulator instantiates are generated as
# the aved flow generates them (TOOL_DIR's xilinx_ip_gen.tcl) and synthesized
# out of context; no-vendor-ip skips them for a kernel without any.

if {$argc < 3} {
    puts "Usage: $::argv0 <kernel_srcs.f> <part> <period_ns> \[no-vendor-ip\]"
    exit 1
}
lassign $argv srcs part period options
set vendor_ip [expr {$options ne "no-vendor-ip"}]

if {$vendor_ip} {
    set tool_dir $::env(TOOL_DIR)
    set argv [list [pwd]/ip $part]
    set argc 2
    source ${tool_dir}/xilinx_ip_gen.tcl
}

set incdirs {}
set fp [open $srcs r]
foreach line [split [read $fp] "\n"] {
    set line [string trim $line]
    if {$line eq "" || [string match "#*" $line]} continue
    if {[string match "+incdir+*" $line]} { lappend incdirs [string range $line 8 end]; continue }
    read_verilog -sv $line
}
close $fp
if {$vendor_ip} {
    foreach xci [glob [pwd]/ip/*/*.xci] { read_ip $xci }
    generate_target all [get_ips]
    synth_ip [get_ips]
}

# The clock with the uncertainty the image's runs see.
set f [open clk.xdc w]
puts $f "create_clock -name ap_clk -period $period \[get_ports ap_clk\]"
puts $f "set_clock_uncertainty -setup 0.200 \[get_clocks ap_clk\]"
close $f
read_xdc clk.xdc

set t0 [clock seconds]
synth_design -top foldsim_afu -part $part -mode out_of_context -include_dirs $incdirs -global_retiming on
set t1 [clock seconds]
report_utilization -file synth_util.rpt
report_timing_summary -max_paths 20 -file synth_timing_summary.rpt
source [file join [file dirname [file normalize [info script]]] floorplan.tcl]
opt_design
place_design -directive Explore
phys_opt_design -directive AggressiveExplore
set t2 [clock seconds]
route_design -directive Explore
phys_opt_design -directive AggressiveExplore
set t3 [clock seconds]
report_utilization -file routed_util.rpt
report_utilization -hierarchical -hierarchical_depth 6 -file routed_util_hier.rpt
report_timing_summary -max_paths 50 -file routed_timing_summary.rpt
report_design_analysis -timing -max_paths 100 -file routed_design_analysis.rpt
report_design_analysis -logic_level_distribution -logic_level_dist_paths 5000 -file routed_lld.rpt
report_high_fanout_nets -max_nets 40 -timing -file routed_hfn.rpt
set f [open times.txt w]
puts $f "synth [expr {$t1-$t0}] place [expr {$t2-$t1}] route [expr {$t3-$t2}] total [expr {$t3-$t0}]"
close $f
