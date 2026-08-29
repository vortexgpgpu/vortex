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

if { $::argc != 3 } {
    puts "ERROR: Program \"$::argv0\" requires 3 arguments!\n"
    puts "Usage: $::argv0 <krnl_name> <vcs_file> <build_dir>\n"
    exit
}

set krnl_name [lindex $::argv 0]
set vcs_file  [lindex $::argv 1]
set build_dir [lindex $::argv 2]

set tool_dir $::env(TOOL_DIR)
set script_dir [ file dirname [ file normalize [ info script ] ] ]

puts "Using krnl_name=$krnl_name"
puts "Using vcs_file=$vcs_file"
puts "Using tool_dir=$tool_dir"
puts "Using build_dir=$build_dir"
puts "Using script_dir=$script_dir"

set path_to_packaged "${build_dir}/xo/packaged_kernel"
set path_to_tmp_project "${build_dir}/xo/project"

source "${tool_dir}/parse_vcs_list.tcl"
set vlist [parse_vcs_list "${vcs_file}"]

set vsources_list  [lindex $vlist 0]
set vincludes_list [lindex $vlist 1]
set vdefines_list  [lindex $vlist 2]

#puts ${vsources_list}
#puts ${vincludes_list}
#puts ${vdefines_list}

set chipscope 0
set num_banks 1
set merged_mem_if 0

# parse vdefines_list for configuration parameters
foreach def $vdefines_list {
    set fields [split $def "="]
    set name [lindex $fields 0]
    if { $name == "CHIPSCOPE" } {
        set chipscope 1
    }
    if { $name == "PLATFORM_MEMORY_NUM_BANKS" } {
        set num_banks [lindex $fields 1]
    }
    if { $name == "PLATFORM_MERGED_MEMORY_INTERFACE" } {
        set merged_mem_if 1
    }
}

if { $merged_mem_if == 1 } {
    set num_banks 1
}

create_project -force kernel_pack $path_to_tmp_project

add_files -norecurse ${vsources_list}

set obj [get_filesets sources_1]
set ip_files [list \
 [file normalize "${build_dir}/ip/xil_fdiv/xil_fdiv.xci"] \
 [file normalize "${build_dir}/ip/xil_fma/xil_fma.xci"] \
 [file normalize "${build_dir}/ip/xil_fsqrt/xil_fsqrt.xci"] \
 [file normalize "${build_dir}/ip/xil_fmul/xil_fmul.xci"] \
 [file normalize "${build_dir}/ip/xil_fadd/xil_fadd.xci"] \
]
add_files -verbose -norecurse -fileset $obj $ip_files

set_property include_dirs ${vincludes_list} [current_fileset]
set_property verilog_define ${vdefines_list} [current_fileset]

set obj [get_filesets sources_1]
set_property -verbose -name "top" -value ${krnl_name} -objects $obj

if { $chipscope == 1 } {
    # hw debugging
    create_ip -name ila -vendor xilinx.com -library ip -version 6.2 -module_name ila_afu
    set_property -dict [list CONFIG.C_ADV_TRIGGER {true} \
                             CONFIG.C_EN_STRG_QUAL {1} \
                             CONFIG.C_DATA_DEPTH {8192} \
                             CONFIG.C_NUM_OF_PROBES {2} \
                             CONFIG.C_PROBE0_WIDTH {8} \
                             CONFIG.C_PROBE1_WIDTH {64} \
                             CONFIG.ALL_PROBE_SAME_MU {false} \
                             CONFIG.ALL_PROBE_SAME_MU_CNT {2} \
                        ] [get_ips ila_afu]
    generate_target {instantiation_template} [get_files ila_afu.xci]
    set_property generate_synth_checkpoint false [get_files ila_afu.xci]

    create_ip -name ila -vendor xilinx.com -library ip -version 6.2 -module_name ila_fetch
    set_property -dict [list CONFIG.C_ADV_TRIGGER {true} \
                             CONFIG.C_EN_STRG_QUAL {1} \
                             CONFIG.C_DATA_DEPTH {8192} \
                             CONFIG.C_NUM_OF_PROBES {3} \
                             CONFIG.C_PROBE0_WIDTH {40} \
                             CONFIG.C_PROBE1_WIDTH {80} \
                             CONFIG.C_PROBE2_WIDTH {40} \
                             CONFIG.ALL_PROBE_SAME_MU {false} \
                             CONFIG.ALL_PROBE_SAME_MU_CNT {2} \
                        ] [get_ips ila_fetch]
    generate_target {instantiation_template} [get_files ila_fetch.xci]
    set_property generate_synth_checkpoint false [get_files ila_fetch.xci]

    create_ip -name ila -vendor xilinx.com -library ip -version 6.2 -module_name ila_issue
    set_property -dict [list CONFIG.C_ADV_TRIGGER {true} \
                             CONFIG.C_EN_STRG_QUAL {1} \
                             CONFIG.C_DATA_DEPTH {8192} \
                             CONFIG.C_NUM_OF_PROBES {4} \
                             CONFIG.C_PROBE0_WIDTH {112} \
                             CONFIG.C_PROBE1_WIDTH {112} \
                             CONFIG.C_PROBE2_WIDTH {280} \
                             CONFIG.C_PROBE3_WIDTH {112} \
                             CONFIG.ALL_PROBE_SAME_MU {false} \
                             CONFIG.ALL_PROBE_SAME_MU_CNT {2} \
                        ] [get_ips ila_issue]
    generate_target {instantiation_template} [get_files ila_issue.xci]
    set_property generate_synth_checkpoint false [get_files ila_issue.xci]

    create_ip -name ila -vendor xilinx.com -library ip -version 6.2 -module_name ila_lsu
    set_property -dict [list CONFIG.C_ADV_TRIGGER {true} \
                             CONFIG.C_EN_STRG_QUAL {1} \
                             CONFIG.C_DATA_DEPTH {8192} \
                             CONFIG.C_NUM_OF_PROBES {3} \
                             CONFIG.C_PROBE0_WIDTH {288} \
                             CONFIG.C_PROBE1_WIDTH {152} \
                             CONFIG.C_PROBE2_WIDTH {72} \
                             CONFIG.ALL_PROBE_SAME_MU {false} \
                             CONFIG.ALL_PROBE_SAME_MU_CNT {2} \
                        ] [get_ips ila_lsu]
    generate_target {instantiation_template} [get_files ila_lsu.xci]
    set_property generate_synth_checkpoint false [get_files ila_lsu.xci]
}

update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
ipx::package_project -root_dir $path_to_packaged -vendor xilinx.com -library RTLKernel -taxonomy /KernelIP -import_files -set_current false
ipx::unload_core $path_to_packaged/component.xml
ipx::edit_ip_in_project -upgrade true -name tmp_edit_project -directory $path_to_packaged $path_to_packaged/component.xml

set core [ipx::current_core]

set_property core_revision 2 $core
foreach up [ipx::get_user_parameters] {
  ipx::remove_user_parameter [get_property NAME $up] $core
}

ipx::associate_bus_interfaces -busif s_axi_ctrl -clock ap_clk $core

# Associate ap_clk with every AXI master interface the RTL actually
# produced — discovered from the packaged core so the list tracks the
# real port set (memory banks + the Command Processor's m_axi_host)
# regardless of which config defines reached RTL elaboration.
set axi_masters {}
foreach bif [ipx::get_bus_interfaces -of $core] {
    set bname [get_property NAME $bif]
    if { [string match "m_axi*" $bname] } {
        lappend axi_masters $bname
    }
}
set axi_masters [lsort $axi_masters]
foreach m $axi_masters {
    ipx::associate_bus_interfaces -busif $m -clock ap_clk $core
}

set mem_map [::ipx::add_memory_map -quiet "s_axi_ctrl" $core]
set addr_block [::ipx::add_address_block -quiet "reg0" $mem_map]

# VX_afu_wrap routes s_axi_ctrl by address bit 12: [0x0000,0x1000) is the
# legacy register block, [0x1000,0x2000) is the Command Processor's
# AXI-Lite register window. Declare a 64 KB address block so the packaged
# interface resolves C_S_AXI_CTRL_ADDR_WIDTH to 16 (>= 13). Without this,
# Vivado infers the width from the enumerated registers alone (~0x40) and
# the bit-12 / [11:0] CP part-selects in VX_afu_wrap.sv go out of range.
set_property range 65536 $addr_block

# User-managed kernel (ap_ctrl_none): the AFU is a CP-driven, always-on
# command processor — the host submits to the CP's host-memory ring and
# rings the doorbell via the CP AXI-Lite regfile (0x1000+), and the CP
# drives Vortex. XRT/ERT must NOT manage an ap_start/ap_done lifecycle, so
# the kernel is packaged user_managed (see -ctrl_protocol in gen_xo.tcl)
# and carries NO ap_ctrl_hs control block (CTRL/GIER/IP_IER/IP_ISR).

set reg [::ipx::add_register -quiet "DEV" $addr_block]
set_property address_offset 0x010 $reg
set_property size           [expr {8*8}]   $reg

set reg [::ipx::add_register -quiet "ISA" $addr_block]
set_property address_offset 0x018 $reg
set_property size           [expr {8*8}]   $reg

set reg [::ipx::add_register -quiet "DCR" $addr_block]
set_property address_offset 0x020 $reg
set_property size           [expr {8*8}]   $reg

set reg [::ipx::add_register -quiet "SCP" $addr_block]
set_property address_offset 0x028 $reg
set_property size           [expr {8*8}]   $reg

# One control register per AXI master interface — the XRT RTL-kernel
# flow requires every m_axi interface to carry an ASSOCIATED_BUSIF
# register. Iterates the discovered master list (memory banks + the
# CP's m_axi_host) so it stays correct for any port count.
set reg_off 0x30
foreach m $axi_masters {
set reg [::ipx::add_register -quiet $m $addr_block]
set_property address_offset $reg_off $reg
set_property size           [expr {8*8}]   $reg
set regparam [::ipx::add_register_parameter ASSOCIATED_BUSIF $reg]
set_property value $m $regparam
set reg_off [expr {$reg_off + 8}]
}

set_property slave_memory_map_ref "s_axi_ctrl" [::ipx::get_bus_interfaces -of $core "s_axi_ctrl"]

set_property xpm_libraries {XPM_CDC XPM_MEMORY XPM_FIFO} $core
set_property sdx_kernel true $core
set_property sdx_kernel_type rtl $core
set_property supported_families { } $core
set_property auto_family_support_level level_2 $core

ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::check_integrity -kernel $core
ipx::save_core $core
close_project -delete
