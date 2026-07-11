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
# Every forward net crossing a module boundary must be driven by a register on
# its source side, in both directions (the module's outgoing signals AND the
# incoming signals driven by the far endpoint). A combinationally-driven
# crossing net cannot be pipelined onto the SLR interposer (Laguna): once the
# module lands on a different SLR than its peer the path degenerates into a long
# combinational interposer route and Fmax collapses. Registering every module's
# external interfaces keeps any inter-SLR crossing latency-insensitive and
# placement-independent.
#
# Run this on the ELABORATED, pre-flatten netlist — i.e. right after
#   synth_design -rtl -flatten_hierarchy none
# where the module hierarchy still has clean interface pins. It is NOT valid on
# a post-synthesis (flattened) netlist: flatten_hierarchy=rebuilt dissolves the
# boundaries and re-parents primitives, so genuine-crossing detection
# mis-classifies internal nets. Sourcing this file runs the check on the current
# in-memory design.
#
# Controls (environment):
#   MODULE_INTERFACE_CHECK = error  report and abort (default)
#                          = warn   report and continue
#                          = off    skip
#
# Excludes clock/reset and the backward handshake (ready); only forward
# valid/data are required to be registered. On a violation the offending nets
# are grouped by the driving buffer and each group is reported with the full
# broken path (net, combinational driver, backward trace to the nearest
# register, and the load) so the unregistered output boundary can be fixed.

# Blocks whose external interfaces are checked, when present in the design.
# {label  hierarchical-regexp}. Generate-block members join on a dot
# (g_*_unit[0].*_core); direct instances join on a slash.
set MODULE_INTERFACE_BLOCKS {
    {l2             {.*/l2cache}}
    {l3             {.*/l3cache}}
    {rtu_core       {.*\.rtu_core}}
    {dxa_core       {.*/dxa_core}}
    {om_core        {.*\.om_core}}
    {tex_core       {.*\.tex_core}}
    {socket         {.*/g_sockets\[[0-9]+\]\.socket}}
    {core           {.*/g_cores\[[0-9]+\]\.core}}
    {cluster        {.*/g_clusters\[[0-9]+\]\.cluster}}
    {tex_cache      {.*/tcache}}
    {om_cache       {.*/ocache}}
    {raster_cache   {.*/rcache}}
    {kmu            {.*/kmu}}
    {global_barrier {.*/gbar_unit}}
}

# Max combinational levels a boundary net may sit from its launch register and
# still count as "register-bounded". A registered output (OUT_BUF=3) drives the
# boundary at depth 0; a 2-entry skid (OUT_BUF=2 — the pervasive Vortex pattern
# the methodology rates OK) drives it through its bypass mux at depth 1. Genuine
# unregistered distribution (an arb/xbar with no output skid, or a dropped
# register) leaves a multi-level combinational cone (>= ~4 levels) back to the
# nearest register, so a small bound cleanly separates the two. Override via
# MODULE_INTERFACE_CHECK_DEPTH.
set MIC_REG_MAXDEPTH 3
if {[info exists ::env(MODULE_INTERFACE_CHECK_DEPTH)]} { set MIC_REG_MAXDEPTH $::env(MODULE_INTERFACE_CHECK_DEPTH) }

proc mic_is_seq {cell} {
    if {$cell eq ""} { return 0 }
    # RTL elaboration primitives: RTL_REG*. Mapped primitives: FD*, LD*.
    if {[get_property -quiet IS_SEQUENTIAL $cell] eq "1"} { return 1 }
    return [regexp {^(RTL_REG|FD|LD)} [get_property -quiet REF_NAME $cell]]
}

# Minimum number of combinational levels from $cell's output back to the nearest
# register on its DATA fan-in (clock/reset/enable/select excluded), bounded by
# $maxdepth. Returns 0 if $cell is itself a register, N>0 for a register N levels
# back on some data path, or -1 if none within $maxdepth. Bounded DFS over all
# data inputs takes the min over paths, so a skid's output mux (register on one
# leg, bypass data on the other) resolves to depth 1 regardless of pin order.
proc mic_data_drivers {cell} {
    set drivers {}
    foreach ip [get_pins -quiet -of_objects $cell -filter {DIRECTION == IN}] {
        set pn [get_property -quiet REF_PIN_NAME $ip]
        if {[regexp -nocase {clk|reset|rst|^ce|^[rs]$|en$} $pn]} { continue }
        set n [get_nets -quiet -of_objects $ip]
        if {$n eq ""} { continue }
        set drv [get_pins -quiet -leaf -of_objects $n -filter {DIRECTION == OUT}]
        if {![llength $drv]} { continue }
        lappend drivers [get_cells -quiet -of_objects [lindex $drv 0]]
    }
    return $drivers
}

# Minimum combinational levels from $cell's output back to the nearest register
# on its data fan-in, as an absolute per-cell property, capped at $::MIC_CAP.
# Memoized in ::MIC_DIST (a cell's distance is fixed and boundary buses share
# driver cones, so without caching a wide bus re-walks the same combinational
# arbiter/ready network thousands of times). Returns 0 for a register, 1 for a
# skid output (bypass mux one hop from data_out_r_reg), and grows for deeper
# unregistered cones. Cells are pre-seeded with $::MIC_CAP before recursion to
# break any combinational cycle conservatively.
proc mic_reg_dist {cell} {
    if {$cell eq ""} { return $::MIC_CAP }
    set nm [get_property -quiet NAME $cell]
    if {[info exists ::MIC_DIST($nm)]} { return $::MIC_DIST($nm) }
    if {[mic_is_seq $cell]} { set ::MIC_DIST($nm) 0 ; return 0 }
    set ::MIC_DIST($nm) $::MIC_CAP
    set best $::MIC_CAP
    foreach dc [mic_data_drivers $cell] {
        set cand [expr {[mic_reg_dist $dc] + 1}]
        if {$cand < $best} { set best $cand }
    }
    if {$best > $::MIC_CAP} { set best $::MIC_CAP }
    set ::MIC_DIST($nm) $best
    return $best
}

# Parent hierarchy of a cell (strip the last '/'-component). Used to group all
# bits of one bus that share the same unregistered driving buffer.
proc mic_parent {name} {
    set i [string last "/" $name]
    if {$i < 0} { return $name }
    return [string range $name 0 [expr {$i - 1}]]
}

# Follow the combinational fan-in of $cell back to the nearest register, so the
# report shows the whole unregistered cone that should end in a boundary reg.
proc mic_trace_back {cell maxdepth} {
    set chain {}
    set cur $cell
    for {set d 0} {$d < $maxdepth} {incr d} {
        if {$cur eq "" || [mic_is_seq $cur]} { break }
        lappend chain "[get_property NAME $cur] ([get_property -quiet REF_NAME $cur])"
        set next ""
        foreach ip [get_pins -quiet -of_objects $cur -filter {DIRECTION == IN}] {
            set pn [get_property -quiet REF_PIN_NAME $ip]
            if {[regexp -nocase {clk|reset|rst|^ce|^[rs]$|en$} $pn]} { continue }
            set n [get_nets -quiet -of_objects $ip]
            if {$n eq ""} { continue }
            set drv [get_pins -quiet -leaf -of_objects $n -filter {DIRECTION == OUT}]
            if {[llength $drv]} { set next [get_cells -quiet -of_objects [lindex $drv 0]]; break }
        }
        set cur $next
    }
    if {$cur ne "" && [mic_is_seq $cur]} {
        lappend chain "REG: [get_property NAME $cur] ([get_property -quiet REF_NAME $cur])"
    } else {
        lappend chain "...(no register within $maxdepth levels)"
    }
    return $chain
}

# Collect unregistered forward boundary nets of one instance into ::MIC_GROUP,
# keyed by {block, module-instance, driving-buffer}. Returns the violation count.
proc mic_check_instance {block inst} {
    set viol 0
    foreach pin [get_pins -quiet -of_objects $inst] {
        set pn [get_property -quiet REF_PIN_NAME $pin]
        set dir [get_property -quiet DIRECTION $pin]
        # A fully registered interface has valid, ready AND data all registered
        # in both directions. Post-elaboration the interface bundle is just flat
        # boundary signals, so every one (forward valid/data on outputs, backward
        # ready on inputs, and vice-versa) is checked — only true clock/reset
        # pins are exempt.
        if {[regexp -nocase {clk|clock|reset|/rst|_rst} $pn]} { continue }
        set net [get_nets -quiet -of_objects $pin]
        if {$net eq ""} { continue }
        set tp [get_property -quiet TYPE $net]
        if {$tp eq "GLOBAL_CLOCK" || $tp eq "POWER" || $tp eq "GROUND"} { continue }
        set src [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT}]
        if {[llength $src] == 0} { continue }
        set dpin [lindex $src 0]
        set dcell [get_cells -quiet -of_objects $dpin]
        # Register-bounded (direct register, or a skid whose register is a few
        # combinational levels back) -> OK. Only a deep unregistered cone fails.
        if {[mic_reg_dist $dcell] <= $::MIC_REG_MAXDEPTH} { continue }
        incr viol
        # group by the driving buffer (parent of the combinational driver)
        set drv_name [get_property NAME $dcell]
        set key [list $block [get_property NAME $inst] [mic_parent $drv_name]]
        if {![info exists ::MIC_GROUP($key)]} {
            # first example for this group: capture net, driver, load, path
            set load ""
            set lp [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == IN}]
            if {[llength $lp]} {
                set lc [get_cells -quiet -of_objects [lindex $lp 0]]
                set load "[get_property NAME $lc] . [get_property -quiet REF_PIN_NAME [lindex $lp 0]]"
            }
            set ::MIC_GROUP($key) [list \
                count 1 \
                dir  $dir \
                pin  $pn \
                net  [get_property NAME $net] \
                drv  "$drv_name ([get_property -quiet REF_NAME $dcell])" \
                load $load \
                path [mic_trace_back $dcell 12]]
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
    # register-distance memo (shared across all blocks) + cap one past the
    # acceptance threshold so "too far" collapses to a single sentinel.
    array unset ::MIC_DIST
    set ::MIC_CAP [expr {$::MIC_REG_MAXDEPTH + 1}]
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
        puts "\nMODULE-IFCHECK: broken interface details (grouped by driving buffer):"
        foreach key [lsort [array names ::MIC_GROUP]] {
            lassign $key block inst buf
            set g $::MIC_GROUP($key)
            puts "\n  \[$block\] $inst"
            puts "    boundary pin : [dict get $g pin] ([dict get $g dir])   x[dict get $g count] net(s) on this buffer"
            puts "    net          : [dict get $g net]"
            puts "    driven by    : [dict get $g drv]        <-- combinational, must be a register"
            if {[dict get $g load] ne ""} {
                puts "    load         : [dict get $g load]"
            }
            puts "    fan-in path back to nearest register:"
            set i 0
            foreach step [dict get $g path] {
                puts "        [string repeat "  " $i]-> $step"
                incr i
            }
        }
        puts "\nMODULE-IFCHECK: FAIL — $total module boundary net(s) combinationally driven (not registered)."
        if {$mode eq "error"} {
            error "MODULE-IFCHECK: module interface(s) not fully registered across the crossing — aborting (MODULE_INTERFACE_CHECK=warn to override)."
        }
        puts "MODULE-IFCHECK: (warn) continuing; MODULE_INTERFACE_CHECK=error to abort."
    } else {
        puts "MODULE-IFCHECK: PASS — all checked module boundary forward nets are register-driven."
    }
}

check_module_interfaces
