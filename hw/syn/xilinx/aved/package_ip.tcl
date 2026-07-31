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
#
# Package the AFU as an IP-XACT component. The V80 linker consumes the
# component.xml directly, so there is no intermediate object container.

if { $::argc != 4 } {
    puts "ERROR: Program \"$::argv0\" requires 4 arguments!\n"
    puts "Usage: $::argv0 <krnl_name> <vcs_file> <build_dir> <part>\n"
    exit 1
}

set krnl_name [lindex $::argv 0]
set vcs_file  [lindex $::argv 1]
set build_dir [lindex $::argv 2]
set part      [lindex $::argv 3]

set path_to_tmp_project "${build_dir}/${krnl_name}_ip_project"
set path_to_packaged     "${build_dir}/${krnl_name}_ip"

create_project -force kernel_pack $path_to_tmp_project -part $part

# The source list is produced by gen_sources.sh so the include order and
# defines match exactly what the simulation build elaborated.
set fp [open $vcs_file r]
set data [read $fp]
close $fp
foreach line [split $data "\n"] {
    set line [string trim $line]
    if {$line eq "" || [string match "#*" $line]} {
        continue
    }
    if {[string match "+incdir+*" $line]} {
        set_property include_dirs [concat [get_property include_dirs [current_fileset]] \
                                          [string range $line 8 end]] [current_fileset]
    } elseif {[string match "+define+*" $line]} {
        set_property verilog_define [concat [get_property verilog_define [current_fileset]] \
                                            [string range $line 8 end]] [current_fileset]
    } else {
        add_files -norecurse $line
    }
}

set_property top $krnl_name [current_fileset]
update_compile_order -fileset sources_1

ipx::package_project -root_dir $path_to_packaged -vendor vortex -library gpgpu \
    -taxonomy /Vortex -import_files -set_current false
ipx::unload_core $path_to_packaged/component.xml
ipx::edit_ip_in_project -upgrade true -name tmp_edit_project \
    -directory $path_to_packaged $path_to_packaged/component.xml

set core [ipx::current_core]
set_property core_revision 1 $core

# Interfaces are inferred from the port-name prefixes; the AFU already
# names them as the linker expects (s_axi_control, m_axi_*, interrupt).
ipx::infer_bus_interfaces xilinx.com:interface:aximm_rtl:1.0 $core
ipx::infer_bus_interfaces xilinx.com:interface:axis_rtl:1.0 $core
ipx::associate_bus_interfaces -busif s_axi_control -clock ap_clk $core

# AMD recommends advertising that the memory-mapped masters use neither
# wrap/fixed bursts nor narrow bursts. These are bus-interface parameters,
# not properties of the interface object.
foreach bif [ipx::get_bus_interfaces -of_objects $core] {
    set bifname [get_property NAME $bif]
    if {[string match "m_axi_*" $bifname]} {
        ipx::associate_bus_interfaces -busif $bifname -clock ap_clk $core
        foreach param {HAS_BURST SUPPORTS_NARROW_BURST} {
            if {[llength [ipx::get_bus_parameters $param -of_objects $bif]] == 0} {
                ipx::add_bus_parameter $param $bif
            }
            set_property value 0 [ipx::get_bus_parameters $param -of_objects $bif]
        }
    }
}

set_property sdx_kernel true $core
set_property sdx_kernel_type rtl $core
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::check_integrity -kernel $core
ipx::save_core $core
close_project -delete
