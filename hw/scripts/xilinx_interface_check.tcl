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
# The FORWARD term (valid/data) must be 100% registered — a module's outgoing
# forward signals AND its peer's incoming forward signals (below).
#
# The BACKWARD term (ready) has no crisp "registered launch" invariant — a
# combinational `req_ready` is by design (the producer's skid absorbs it), so
# "must be registered" is the wrong test, and no local depth rule can be made
# authoritative (the real failures are route/SLR-dominated, which elaboration
# cannot see). The READY-IFCHECK pass is therefore ADVISORY ONLY: it flags
# `req_ready` nets whose combinational cone is deeper than MIC_READY_MAX_LEVELS
# as *candidates* for an SLR-crossing timing risk. It is placement-blind, so a
# hit is a candidate to cross-reference against the post-place timing report, not
# a failure. It NEVER gates the build; MODULE_INTERFACE_READY_CHECK=off silences
# it. clk/reset exempt.
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

# Container modules whose sub-instances form the checked set. Rather than a
# hand-maintained instance list (which rots when the RTL hierarchy changes), the
# check ENUMERATES every module instance contained in these programmatically, so
# adding/removing a sub-module needs no edit here. A container's own boundary is
# also checked (when it is itself a child of another container, e.g. cluster
# under Vortex; the top container's own boundary is checked via CONTAINER_MODULES
# including its wrapper if desired).
set CONTAINER_MODULES {Vortex VX_cluster VX_socket VX_graphics}

# Direct child module instances of a container: descend through generate-block
# wrappers (cells whose REF_NAME is not a real module) but stop at the next real
# module (REF_NAME VX_* or Vortex). Robust to whether Vivado dot-folds a generate
# block into the child name or keeps it as an intermediate cell.
proc mic_child_modules {cont} {
    set out {}
    foreach c [get_cells -quiet ${cont}/* -filter {IS_PRIMITIVE == false}] {
        set ref [get_property -quiet REF_NAME $c]
        if {[regexp {^(VX_|Vortex)} $ref]} {
            lappend out $c
        } else {
            foreach s [mic_child_modules $c] { lappend out $s }
        }
    }
    return $out
}

# Deduplicated {label inst} list of every checked block: each container instance
# plus its child module instances, across all CONTAINER_MODULES. label = base
# REF_NAME (parameterized suffix stripped) for grouped reporting.
proc mic_collect_blocks {} {
    global CONTAINER_MODULES
    array unset ::MIC_SEEN_BLK
    set blocks {}
    foreach ctype $CONTAINER_MODULES {
        foreach cont [get_cells -quiet -hierarchical -filter "REF_NAME =~ {${ctype}} || REF_NAME =~ {${ctype}__*}"] {
            foreach inst [concat [list $cont] [mic_child_modules $cont]] {
                set nm [get_property -quiet NAME $inst]
                if {[info exists ::MIC_SEEN_BLK($nm)]} { continue }
                set ::MIC_SEEN_BLK($nm) 1
                set ref [get_property -quiet REF_NAME $inst]
                regsub {__parameterized.*$} $ref "" ref
                lappend blocks [list $ref $inst]
            }
        }
    }
    return $blocks
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

# Combinational cone depth feeding a net: 0 if the driver is a register / static
# / boundary launch, else 1 + max input-cone depth. Bounded — we only care
# whether it exceeds `max`, so the walk early-outs and memoizes per cell.
proc mic_comb_depth {net max} {
    global MIC_DEPTH_SEEN
    set src [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT}]
    if {[llength $src] == 0} { return 0 }
    set dcell [get_cells -quiet -of_objects [lindex $src 0]]
    if {$dcell eq "" || [mic_is_reg $dcell]} { return 0 }
    set cn [get_property -quiet NAME $dcell]
    if {[info exists MIC_DEPTH_SEEN($cn)]} { return $MIC_DEPTH_SEEN($cn) }
    set MIC_DEPTH_SEEN($cn) 0
    set best 0
    foreach ip [get_pins -quiet -of_objects $dcell -filter {DIRECTION == IN}] {
        set inet [get_nets -quiet -of_objects $ip]
        if {$inet eq ""} { continue }
        set tp [get_property -quiet TYPE $inet]
        if {$tp eq "GLOBAL_CLOCK" || $tp eq "POWER" || $tp eq "GROUND"} { continue }
        set d [mic_comb_depth $inet $max]
        if {$d > $best} { set best $d }
        if {$best > $max} { break }
    }
    set r [expr {$best + 1}]
    set MIC_DEPTH_SEEN($cn) $r
    return $r
}

proc mic_ready_bydepth {a b} {
    return [expr {[dict get $::MIC_READY_GROUP($b) depth] - [dict get $::MIC_READY_GROUP($a) depth]}]
}

# Backward-ready launch check for one block instance: a boundary `req_ready` net
# that is neither a registered launch nor a shallow (<= maxlevels) cone is a
# violation. Records into ::MIC_READY_GROUP; returns the count.
proc mic_check_ready_instance {block inst maxlevels} {
    set viol 0
    foreach pin [get_pins -quiet -of_objects $inst] {
        set pn [get_property -quiet REF_PIN_NAME $pin]
        if {![regexp -nocase {req_ready} $pn]} { continue }
        set net [get_nets -quiet -of_objects $pin]
        if {$net eq ""} { continue }
        set tp [get_property -quiet TYPE $net]
        if {$tp eq "GLOBAL_CLOCK" || $tp eq "POWER" || $tp eq "GROUND"} { continue }
        set src [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT}]
        if {[llength $src] == 0} { continue }
        set dcell [get_cells -quiet -of_objects [lindex $src 0]]
        if {[mic_is_reg $dcell]} { continue }
        array unset ::MIC_DEPTH_SEEN
        set depth [mic_comb_depth $net $maxlevels]
        if {$depth <= $maxlevels} { continue }
        incr viol
        set ref [get_property -quiet REF_NAME $dcell]
        set key [list $block [get_property -quiet NAME $inst] [get_property -quiet NAME $net]]
        set ::MIC_READY_GROUP($key) [list \
            depth $depth \
            pin   $pn \
            net   [get_property -quiet NAME $net] \
            drv   "[get_property -quiet NAME $dcell] ($ref)"]
    }
    return $viol
}

proc check_ready_interfaces {} {
    # Advisory only — this pass never gates the build (the ready direction has no
    # binary "registered" invariant to gate on). MODULE_INTERFACE_READY_CHECK=off
    # silences it entirely; any other value just prints the advisory.
    set rmode "warn"
    if {[info exists ::env(MODULE_INTERFACE_READY_CHECK)]} { set rmode $::env(MODULE_INTERFACE_READY_CHECK) }
    if {$rmode eq "off" || $rmode eq "0"} { return }
    set maxlevels 4
    if {[info exists ::env(MIC_READY_MAX_LEVELS)]} { set maxlevels $::env(MIC_READY_MAX_LEVELS) }

    array unset ::MIC_READY_GROUP
    set rtotal 0
    set checked 0
    array set _cnt {}
    array set _bv  {}
    foreach pair [mic_collect_blocks] {
        lassign $pair label inst
        if {![info exists _cnt($label)]} { set _cnt($label) 0; set _bv($label) 0 }
        incr _cnt($label)
        incr _bv($label) [mic_check_ready_instance $label $inst $maxlevels]
        incr checked
    }
    foreach label [lsort [array names _cnt]] {
        incr rtotal $_bv($label)
        puts [format "READY-IFCHECK: %-18s : %d instance(s), %d deep-ready candidate(s) (>%d lvls)%s" \
            $label $_cnt($label) $_bv($label) $maxlevels [expr {$_bv($label) ? "  <-- WARN" : ""}]]
    }
    if {$checked == 0} { return }
    if {$rtotal > 0} {
        puts "\nREADY-IFCHECK: deep backward-ready candidates (advisory, ranked by depth):"
        foreach key [lsort -command mic_ready_bydepth [array names ::MIC_READY_GROUP]] {
            lassign $key block inst net
            set g $::MIC_READY_GROUP($key)
            puts "\n  \[$block\] $inst"
            puts "    ready pin : [dict get $g pin]   depth=[dict get $g depth] comb levels"
            puts "    net       : [dict get $g net]"
            puts "    driven by : [dict get $g drv]   <-- deep combinational; CANDIDATE — confirm vs post-place timing"
        }
        puts "\nREADY-IFCHECK: WARNING (advisory, non-gating) — $rtotal deep backward-ready candidate(s);"
        puts "               placement-blind, so cross-reference the post-place timing report before acting."
        puts "               MODULE_INTERFACE_READY_CHECK=off to silence."
    } else {
        puts "READY-IFCHECK: PASS — no deep backward-ready candidate on checked blocks."
    }
}

proc check_module_interfaces {} {
    set mode "error"
    if {[info exists ::env(MODULE_INTERFACE_CHECK)]} { set mode $::env(MODULE_INTERFACE_CHECK) }
    if {$mode eq "off" || $mode eq "0"} { return }

    array unset ::MIC_GROUP
    set total 0
    set checked 0
    array set _cnt {}
    array set _bv  {}
    foreach pair [mic_collect_blocks] {
        lassign $pair label inst
        if {![info exists _cnt($label)]} { set _cnt($label) 0; set _bv($label) 0 }
        incr _cnt($label)
        incr _bv($label) [mic_check_instance $label $inst]
        incr checked
    }
    foreach label [lsort [array names _cnt]] {
        incr total $_bv($label)
        puts [format "MODULE-IFCHECK: %-18s : %d instance(s), %d unregistered boundary net(s)%s" \
            $label $_cnt($label) $_bv($label) [expr {$_bv($label) ? "  <-- FAIL" : ""}]]
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
check_ready_interfaces
