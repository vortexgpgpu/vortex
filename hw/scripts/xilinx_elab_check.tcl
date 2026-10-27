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

# Elaboration-time module-interface registration gate.
#
# Sourced from gen_xo.tcl (which already has ${krnl_name}/${vcs_file}/
# ${build_dir}/${tool_dir} in scope and has just generated the FPU IP). Stands
# up a throwaway in-memory elaboration with the same sources / IP / defines as
# package_kernel.tcl, runs
#   synth_design -rtl -flatten_hierarchy none
# (the only netlist state on which xilinx_interface_check.tcl is valid — module
# boundaries intact, register vs combinational drivers still visible) and runs
# the DRC. A violation prints the full broken-path debug and aborts the batch
# (exit 1) so the hours-long v++ implementation never starts on an unregistered
# SLR-crossing boundary. Severity via MODULE_INTERFACE_CHECK (error|warn|off,
# default error) inside xilinx_interface_check.tcl.
#
# No device part is set: the DRC inspects driver types (RTL_REG vs
# combinational), never placement, so the partless in-memory project that
# ip_gen/package_kernel already use is sufficient and keeps the FPU IP part
# context consistent.

set mic_mode "error"
if {[info exists ::env(MODULE_INTERFACE_CHECK)]} { set mic_mode $::env(MODULE_INTERFACE_CHECK) }
if {$mic_mode eq "off" || $mic_mode eq "0"} {
    puts "MODULE-IFCHECK: disabled (MODULE_INTERFACE_CHECK=off) — skipping elaboration gate."
} else {
    puts "MODULE-IFCHECK: elaboration gate on \"${krnl_name}\""

    source "${tool_dir}/parse_vcs_list.tcl"
    set mic_vlist    [parse_vcs_list "${vcs_file}"]
    set mic_sources  [lindex $mic_vlist 0]
    set mic_includes [lindex $mic_vlist 1]
    set mic_defines  [lindex $mic_vlist 2]

    create_project -in_memory

    add_files -norecurse ${mic_sources}

    # Black-box every generated vendor IP via its BLOCK_STUB so a -rtl elaboration
    # resolves the IP instances (FPU FMA/DIV/SQRT/MUL/ADD, etc.) without pulling
    # in their internals. The DRC only inspects module boundaries, never IP guts,
    # and the FPU is an interior unit — not a checked boundary block. Globbed so
    # any IP the build generates is covered without a hard-coded list.
    set mic_stubs [glob -nocomplain "${build_dir}/ip/*/*_bmstub.v" "${build_dir}/ip/*/*_stub.v"]
    if {[llength $mic_stubs] == 0} {
        puts "MODULE-IFCHECK: WARNING — no IP black-box stubs found under ${build_dir}/ip; elaboration may fail on vendor IP."
    }
    foreach mic_s $mic_stubs { add_files -norecurse $mic_s }

    set_property include_dirs   ${mic_includes} [current_fileset]
    set_property verilog_define ${mic_defines}  [current_fileset]
    set_property top ${krnl_name} [current_fileset]
    update_compile_order -fileset sources_1

    # Elaborate only (RTL run), preserving hierarchy so module boundaries survive
    # for the interface DRC. Vendor IP stays black-boxed via the stubs above.
    synth_design -rtl -flatten_hierarchy none -top ${krnl_name}

    # check_module_interfaces prints the full broken-path detail, then throws in
    # error mode; convert to a non-zero exit so the whole gen_xo batch aborts and
    # no .xo is produced (build stops before v++ implementation).
    if {[catch { source "${tool_dir}/xilinx_interface_check.tcl" } mic_msg]} {
        puts $mic_msg
        exit 1
    }

    close_project
}
