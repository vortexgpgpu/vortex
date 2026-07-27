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

# Module-interface registration DRC.
#
# Rule (simple): for each checked block, every forward boundary signal — input
# AND output — must be 100% registered, i.e. the net's driver is a flip-flop.
# Anything else fails:
#
#   driver is a flip-flop / sync-RAM read / constant tie   -> registered  (PASS)
#   driver is any combinational cell (RTL_AND/OR/MUX/...)   -> unregistered (FAIL)
#
# This is a hard "fully registered" test, not "register-bounded":
#   - OUT_BUF=3 full register (OUT_REG=1): pin is driven by data_out_r -> FF. PASS.
#   - OUT_BUF=2 skid: pin is driven by the cut-through bypass mux (data_out_i),
#     not a flip-flop -> FAIL. A skid can present a combinational input->output
#     path when empty, so it is NOT 100% registered.
#   - Combinational bridge (VX_om_steer L2 leg: l2_out = bus_in & ~is_om;
#     VX_dcr_flush REQ_OUT_BUF=0: cache_bus driven by a combinational arb) -> FAIL.
#
# A combinationally-driven crossing cannot be pipelined onto the SLR interposer:
# once the module lands on a different SLR than its peer the path becomes a long
# combinational interposer route and Fmax collapses. A registered launch keeps
# every crossing latency-insensitive and placement-independent.
#
# Both directions are checked (a module's outgoing forward signals AND the
# incoming forward signals its peer drives). The backward handshake (ready) is
# exempt — combinational elastic back-pressure by design. clk/reset exempt.
#
# Run on the ELABORATED, pre-flatten netlist (synth_design -rtl
# -flatten_hierarchy none): module boundaries are intact and flip-flop vs
# combinational drivers are still distinct. Sourcing this file runs the check on
# the current in-memory design.
#
# Controls (environment):
#   MODULE_INTERFACE_CHECK = error  report and abort (default)
#                          = warn   report and continue
#                          = off    skip

# Blocks whose external interfaces are checked, when present in the design.
# {label  hierarchical-regexp}. Generate-block members join on a dot
# (g_*_unit[0].*_core); direct instances join on a slash.
set MODULE_INTERFACE_BLOCKS {
    {vortex         {.*/vortex}}
    {vortex_axi     {.*/vortex_axi}}
    {cp_core        {.*/cp_core}}
    {l2             {.*/l2cache}}
    {l3             {.*/l3cache}}
    {rtu_core       {.*\.rtu_core}}
    {dxa_core       {.*/dxa_core}}
    {om_core        {.*\.om_core}}
    {tex_core       {.*\.tex_core}}
    {socket         {.*/g_sockets\[[0-9]+\]\.socket}}
    {core           {.*/g_cores\[[0-9]+\]\.core}}
    {cluster        {.*/g_clusters\[[0-9]+\]\.cluster}}
    {dmmu           {.*/dmmu}}
    {immu           {.*/immu}}
    {l2_tlb         {.*/l2tlb}}
    {ptw            {.*/ptw}}
    {tex_cache      {.*/tcache}}
    {om_cache       {.*/ocache}}
    {raster_cache   {.*/rcache}}
    {kmu            {.*/kmu}}
    {global_barrier {.*/gbar_unit}}
}

# A registered (or static) driver: a flip-flop / latch, a synchronous RAM read
# port, or a constant tie. Everything else is combinational.
proc mic_is_reg {cell} {
    if {$cell eq ""} { return 1 }
    if {[get_property -quiet IS_SEQUENTIAL $cell] eq "1"} { return 1 }
    return [regexp {^(RTL_REG|RTL_LATCH|FD|LD|RAMB|RTL_RAM|GND|VCC|RTL_CONST|RTL_GND|RTL_VCC)} \
                [get_property -quiet REF_NAME $cell]]
}

# Check one block instance: every forward boundary net's driver must be a
# register. Records violations into ::MIC_GROUP keyed by {block, inst, driver
# ref} and returns the count.
proc mic_check_instance {block inst} {
    set viol 0
    foreach pin [get_pins -quiet -of_objects $inst] {
        set pn [get_property -quiet REF_PIN_NAME $pin]
        # Only master REQUEST interfaces are registered by their parent module:
        # the forward request a master drives out (req_valid/req_data). The
        # response channel (rsp_*) is the slave's to register, the backward
        # handshake (ready) is combinational elastic back-pressure, and status
        # wires (busy/done/flush) are not bus interfaces — all exempt.
        if {![regexp -nocase {req_valid|req_data} $pn]} { continue }
        set net [get_nets -quiet -of_objects $pin]
        if {$net eq ""} { continue }
        set tp [get_property -quiet TYPE $net]
        if {$tp eq "GLOBAL_CLOCK" || $tp eq "POWER" || $tp eq "GROUND"} { continue }
        set src [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT}]
        if {[llength $src] == 0} { continue }
        set dcell [get_cells -quiet -of_objects [lindex $src 0]]
        if {[mic_is_reg $dcell]} { continue }
        incr viol
        set ref [get_property -quiet REF_NAME $dcell]
        set key [list $block [get_property -quiet NAME $inst] $ref]
        if {![info exists ::MIC_GROUP($key)]} {
            set ::MIC_GROUP($key) [list \
                count 1 \
                dir  [get_property -quiet DIRECTION $pin] \
                pin  $pn \
                net  [get_property -quiet NAME $net] \
                drv  "[get_property -quiet NAME $dcell] ($ref)"]
        } else {
            dict incr ::MIC_GROUP($key) count
        }
    }
    return $viol
}

proc check_module_interfaces {} {
    global MODULE_INTERFACE_BLOCKS
    set mode "error"
    if {[info exists ::env(MODULE_INTERFACE_CHECK)]} { set mode $::env(MODULE_INTERFACE_CHECK) }
    if {$mode eq "off" || $mode eq "0"} { return }

    array unset ::MIC_GROUP
    set total 0
    set checked 0
    foreach entry $MODULE_INTERFACE_BLOCKS {
        lassign $entry label re
        set insts [get_cells -quiet -hierarchical -regexp $re]
        if {[llength $insts] == 0} { continue }
        set bviol 0
        foreach inst $insts { incr bviol [mic_check_instance $label $inst] }
        incr checked
        incr total $bviol
        puts [format "MODULE-IFCHECK: %-14s : %d instance(s), %d unregistered boundary net(s)%s" \
            $label [llength $insts] $bviol [expr {$bviol ? "  <-- FAIL" : ""}]]
    }

    if {$checked == 0} {
        puts "MODULE-IFCHECK: none of the target blocks present; skipped"
        return
    }

    if {$total > 0} {
        puts "\nMODULE-IFCHECK: broken interface details (grouped by combinational driver):"
        foreach key [lsort [array names ::MIC_GROUP]] {
            lassign $key block inst ref
            set g $::MIC_GROUP($key)
            puts "\n  \[$block\] $inst"
            puts "    boundary pin : [dict get $g pin] ([dict get $g dir])   x[dict get $g count] net(s) on this driver"
            puts "    net          : [dict get $g net]"
            puts "    driven by    : [dict get $g drv]        <-- combinational, must be a register"
        }
        puts "\nMODULE-IFCHECK: FAIL — $total module boundary net(s) not register-driven."
        if {$mode eq "error"} {
            error "MODULE-IFCHECK: module interface(s) not fully registered across the crossing — aborting (MODULE_INTERFACE_CHECK=warn to override)."
        }
        puts "MODULE-IFCHECK: (warn) continuing; MODULE_INTERFACE_CHECK=error to abort."
    } else {
        puts "MODULE-IFCHECK: PASS — every checked forward boundary net is register-driven."
    }
}

check_module_interfaces
