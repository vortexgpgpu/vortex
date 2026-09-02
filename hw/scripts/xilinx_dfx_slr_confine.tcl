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

# Keep a block of RM logic inside ONE SLR, using ranges clipped from the DFX
# partition's own pblock.
#
# Mechanism. The SLASH compute shell wraps the whole user design in a single DFX
# reconfigurable partition, pblock_slash, spanning all three SLRs of the V80
# (SLICE_X0Y236 .. SLICE_X363Y898). It says *where* the RM may go, never that
# any part of it should stay TOGETHER. This script adds a nested pblock whose
# ranges are pblock_slash's OWN ranges clipped to one SLR, so it is by
# construction a subset of the reconfigurable region and cannot fight the DFX
# boundary. NoC NMU/NSU ports and clock/reset infrastructure are deliberately
# left unconstrained, so nothing is pushed off a hard block it must reach.
#
# ---------------------------------------------------------------------------
# HISTORY -- read before changing the scope. Two theories died here.
# ---------------------------------------------------------------------------
#
# 1. "The placer smears the RM for want of guidance; pin the RM to one SLR."
#    DISPROVEN, and expensively. Same config, 3.333 ns target:
#
#      free placement           WNS -0.260 ns   (278.3 MHz)
#      confined to SLR1         WNS -0.801 ns
#      confined to SLR2         WNS -1.599 ns
#
#    The RM was never smeared. vortex_afu_0 has always been ~99.8% in SLR1;
#    the LUT-per-SLR figures that suggested otherwise were top_wrapper totals
#    dominated by the AVED static shell, not the AFU. Pinning the AFU to SLR1
#    therefore bought nothing and only added rigidity, and pinning it to SLR2
#    dragged 88,000 cells across the interposer to chase 6,500.
#
# 2. "The shell's Congestion_SSI_SpreadLogic_high strategy causes the spread."
#    ALSO DISPROVEN. Build strat300 cleared BalancedSLR and set
#    NET_DELAY_WEIGHT high, and came out at WNS -0.343 against the -0.260
#    baseline. More tellingly the partition did not move: vortex_afu_0 stayed
#    99.8% in SLR1 and both SmartConnects stayed ~87% in SLR2. A placer told
#    to prioritise net delay still left them there. The spread was never a
#    strategy decision.
#
# WHAT IS ACTUALLY TRUE (measured on the routed checkpoint):
#
#   NOC_NMU_HBM2E sites   SLR0: 0   SLR1: 0   SLR2: 64
#
# Every HBM channel on this device is in SLR2. MEM_TAG's VNOC ports are in
# SLR1, so the AFU follows them there (99.8%), while HOST_TAG=HBM1 forces the
# host path's SmartConnect to SLR2. The two masters are anchored on different
# dies by the connectivity tags, and no placement directive can change that.
#
# THE REMAINING PLACEMENT LEVER, and what this script now does:
#
# Both SmartConnects are SPLIT across the boundary -- smartconnect_0 is 11/89
# and hbm_sc_01 is 14/86 between SLR1 and SLR2. Their own worst paths are
# INTERNAL: hbm_sc_01's is its FIFO empty flag reaching its own read-address
# counter, smartconnect_0's is its FIFO full flag reaching its own AXI-Lite
# converter. Those loops are crossing the interposer inside a 579- and
# 1775-LUT block. Making each block CONTIGUOUS -- on the die its anchor is
# already on, SLR2, not the AFU's die -- removes those crossings without
# touching connectivity, RTL, or the address map. It targets 509 of the 942
# failing endpoints.
#
# Note the direction. Three builds were lost pulling logic toward the AFU;
# the anchor wins, so the soft logic goes to the anchor.
#
# 3. ...AND THAT WAS DISPROVEN TOO. Build ic300 pinned both SmartConnects
#    (1959 LUT) into SLR2, the die they were already ~87% on. The pblock
#    applied cleanly and the result was WNS -0.326 with 3216 failing
#    endpoints, against -0.260 / 942 for free placement. The gentlest
#    possible constraint, on 2% of the design, moving cells to where most of
#    them already sat, TRIPLED the failure count.
#
#    Why: the SmartConnects are split ON PURPOSE. The placer keeps the
#    AFU-facing ~13% in SLR1 beside cp_core/u_slice_host and the NoC-facing
#    ~87% in SLR2 beside the HBM anchor. When a block's two ports are on
#    different dies, spanning it is correct, and pinning it forces one port
#    into a long route.
#
# SCOREBOARD -- every placement intervention tried has LOST to free placement:
#
#    free placement                 WNS -0.260   942 failing
#    confine AFU to SLR1            WNS -0.801
#    confine AFU to SLR2            WNS -1.599
#    clear BalancedSLR strategy     WNS -0.343  4831 failing
#    confine SmartConnects to SLR2  WNS -0.326  3216 failing
#
# PLACEMENT IS EXHAUSTED ON THIS DESIGN. Do not spend another build on
# pblocks, strategies or directives. The remaining levers are RTL (register
# the AFU<->SmartConnect crossing so it is latency-insensitive) and
# connectivity (HOST_TAG, which decides which die the host port anchors on).
# This script stays OFF.
#
# Controls (environment):
#   VX_DFX_SLR_CONFINE=1    enable (default: OFF)
#   VX_DFX_SLR_SCOPE=ic     confine the interconnect blocks (default)
#                  =rm      the old whole-RM scopes -- DISPROVEN, see above
#   VX_DFX_SLR=SLR2         force a particular SLR (default: chosen by best fit)
#   VX_DFX_SLR_CEILING=f    per-resource fill cap, 0..1 (default 0.70)

# Site-name prefixes a confined soft-logic pblock needs ranges for, and the
# resource each contributes to. Clock/IO/NoC types are intentionally absent:
# those cells stay unconstrained. Weight is units of the resource per site --
# a SLICE holds 8 LUTs, a RAMB36 counts as two RAMB18 equivalents.
array set ::DFX_TYPE_RES {
    SLICE        {LUT  8}
    RAMB18       {BRAM 1}
    RAMB36       {BRAM 2}
    URAM288      {URAM 1}
    URAM_CAS_DLY {URAM 0}
    DSP          {DSP  1}
    DSP58_CPLX   {DSP  0}
}

# Hard blocks the RM is tethered to. These are NOT given pblock ranges -- the
# NoC is a fixed network and must never be constrained -- but the SLR they sit
# in decides which SLR the soft logic should be confined to.
#
# This matters more than capacity. Confining to the roomiest SLR instead of the
# one holding the NoC ports was measured at WNS -1.599 against a -0.260
# baseline: with the logic in SLR2 and the HBM NMU at NOC_NMU512_X2Y9 in SLR1,
# every memory path became a round trip (NMU -> SLR1->2 -> one LUT -> SLR2->1 ->
# NMU), 3.753 ns of route for 0.744 ns of logic.
set ::DFX_ANCHOR_TYPES {NOC_NMU512 NOC_NSU512}

# Split a site or range endpoint name "PREFIX_X<n>Y<m>" into {prefix x y}.
# Returns {} if the name does not have that shape.
proc dfx_split_site {name} {
    if {[regexp {^(.+)_X(\d+)Y(\d+)$} $name -> prefix x y]} {
        return [list $prefix $x $y]
    }
    return {}
}

# Y bounds of each whitelisted site type within one SLR. SLRs on this device are
# horizontal bands, so a site type's Y range fully determines SLR membership for
# sites of that type. Queried per type by name glob so this never walks the
# device's full site list, and site objects stringify to their names, so no
# get_property call is needed per site.
proc dfx_slr_bounds {slr bounds_var} {
    upvar 1 $bounds_var B
    foreach t [concat [array names ::DFX_TYPE_RES] $::DFX_ANCHOR_TYPES] {
        set lo -1 ; set hi -1
        foreach name [get_sites -quiet -of_objects $slr "${t}_X*Y*"] {
            if {![regexp {Y(\d+)$} $name -> y]} { continue }
            if {$lo < 0 || $y < $lo} { set lo $y }
            if {$y > $hi} { set hi $y }
        }
        if {$lo >= 0} {
            set B($t,min) $lo
            set B($t,max) $hi
        }
    }
}

# Clip pblock_slash's ranges to one SLR's rows. Ranges whose site type is not in
# the whitelist, or that fall entirely outside the SLR, are dropped.
proc dfx_clip_ranges {ranges bounds_var} {
    upvar 1 $bounds_var B
    set out {}
    foreach r $ranges {
        set ep [split $r ":"]
        if {[llength $ep] != 2} { continue }
        set lo [dfx_split_site [lindex $ep 0]]
        set hi [dfx_split_site [lindex $ep 1]]
        if {[llength $lo] == 0 || [llength $hi] == 0} { continue }
        lassign $lo prefix xlo ylo
        lassign $hi prefix2 xhi yhi
        if {$prefix ne $prefix2} { continue }
        # Anchor types get Y bounds so SLR membership can be tested, but must
        # never enter the pblock: constraining the NoC is not ours to do.
        if {![info exists ::DFX_TYPE_RES($prefix)]} { continue }
        if {![info exists B($prefix,min)]} { continue }
        set ylo [expr {$ylo > $B($prefix,min) ? $ylo : $B($prefix,min)}]
        set yhi [expr {$yhi < $B($prefix,max) ? $yhi : $B($prefix,max)}]
        if {$ylo > $yhi} { continue }
        lappend out "${prefix}_X${xlo}Y${ylo}:${prefix}_X${xhi}Y${yhi}"
    }
    return $out
}

# Exact capacity of the dynamic region per SLR: walk the sites pblock_slash
# actually covers once and bucket each into an SLR by its Y band. Counting real
# sites (rather than multiplying range extents) keeps the numbers honest across
# the holes the region's ranges are carved around.
proc dfx_region_capacity {pb slrs bounds_arr cap_var} {
    upvar 1 $bounds_arr BB
    upvar 1 $cap_var CAP
    foreach slr $slrs {
        foreach r {LUT BRAM DSP URAM} { set CAP([get_property NAME $slr],$r) 0 }
    }
    foreach name [get_sites -quiet -of_objects $pb] {
        set parts [dfx_split_site $name]
        if {[llength $parts] == 0} { continue }
        lassign $parts prefix x y
        if {![info exists ::DFX_TYPE_RES($prefix)]} { continue }
        lassign $::DFX_TYPE_RES($prefix) res weight
        if {$weight == 0} { continue }
        foreach slr $slrs {
            set nm [get_property NAME $slr]
            if {![info exists BB($nm,$prefix,min)]} { continue }
            if {$y >= $BB($nm,$prefix,min) && $y <= $BB($nm,$prefix,max)} {
                incr CAP($nm,$res) $weight
                break
            }
        }
    }
}

# Combined footprint of one or more hierarchical cells as {LUT BRAM DSP URAM}.
proc dfx_footprint {cells foot_var} {
    upvar 1 $foot_var F
    array set F {LUT 0 BRAM 0 DSP 0 URAM 0}
    foreach cell $cells {
        set p [get_property NAME $cell]
        incr F(LUT)  [llength [get_cells -quiet -hierarchical \
            -filter "PRIMITIVE_SUBGROUP == LUT && NAME =~ {${p}/*}"]]
        set b36 [llength [get_cells -quiet -hierarchical \
            -filter "(REF_NAME =~ RAMB36* || REF_NAME =~ FIFO36*) && NAME =~ {${p}/*}"]]
        set b18 [llength [get_cells -quiet -hierarchical \
            -filter "(REF_NAME =~ RAMB18* || REF_NAME =~ FIFO18*) && NAME =~ {${p}/*}"]]
        incr F(BRAM) [expr {2 * $b36 + $b18}]
        incr F(DSP)  [llength [get_cells -quiet -hierarchical \
            -filter "REF_NAME =~ DSP* && NAME =~ {${p}/*}"]]
        incr F(URAM) [llength [get_cells -quiet -hierarchical \
            -filter "REF_NAME =~ URAM* && NAME =~ {${p}/*}"]]
    }
}

# How much NoC the dynamic region offers in each SLR.
#
# This counts SITES, not placed cells. The RM's NoC ports are implemented as
# part of the RM and have no LOC until placement, so a census of cell locations
# reads back zero at pre-opt -- which is exactly what it did, silently handing
# the tie-break to raw capacity again. Site inventory is available up front and
# is the honest proxy: the placer can only put an NMU where NMU sites exist, so
# the SLR holding most of the region's NoC sites is where the memory interface
# will land.
proc dfx_noc_sites {pb slrs bounds_arr anchor_var} {
    upvar 1 $bounds_arr BB
    upvar 1 $anchor_var AN
    foreach slr $slrs { set AN([get_property NAME $slr]) 0 }
    foreach t $::DFX_ANCHOR_TYPES {
        foreach name [get_sites -quiet -of_objects $pb "${t}_X*Y*"] {
            set parts [dfx_split_site $name]
            if {[llength $parts] == 0} { continue }
            lassign $parts prefix x y
            foreach slr $slrs {
                set nm [get_property NAME $slr]
                if {![info exists BB($nm,$prefix,min)]} { continue }
                if {$y >= $BB($nm,$prefix,min) && $y <= $BB($nm,$prefix,max)} {
                    incr AN($nm)
                    break
                }
            }
        }
    }
}

proc dfx_fmt {arr_var} {
    upvar 1 $arr_var A
    set out {}
    foreach r {LUT BRAM DSP URAM} { lappend out "$r $A($r)" }
    return [join $out ", "]
}

proc dfx_slr_confine_run {} {
    # OFF by default: measured HARMFUL on this shell. See the header note.
    if {!([info exists ::env(VX_DFX_SLR_CONFINE)] && $::env(VX_DFX_SLR_CONFINE) eq "1")} {
        puts "DFX-SLR: not applied (set VX_DFX_SLR_CONFINE=1 to enable; default scope 'ic' confines the interconnect blocks, see header)"
        return
    }
    set ceil 0.70
    if {[info exists ::env(VX_DFX_SLR_CEILING)]} { set ceil $::env(VX_DFX_SLR_CEILING) }

    set slrs [lsort [get_slrs -quiet]]
    if {[llength $slrs] < 2} {
        puts "DFX-SLR: device has [llength $slrs] SLR(s); nothing to confine"
        return
    }

    set pb [get_pblocks -quiet pblock_slash]
    if {$pb eq ""} {
        puts "DFX-SLR: pblock_slash not found; this is not the SLASH DFX flow. Skipping."
        return
    }
    set ranges [get_property -quiet GRID_RANGES $pb]
    if {[llength $ranges] == 0} {
        set ranges [get_property -quiet DERIVED_RANGES $pb]
    }
    if {[llength $ranges] == 0} {
        puts "DFX-SLR: pblock_slash has no ranges to intersect; skipping."
        return
    }

    # -- candidate scopes, largest first --
    #
    # The widest scope is the RM's two soft-logic blocks: the Vortex AFU and the
    # shell's AXI SmartConnect. Confining the AFU alone is NOT enough --
    # smartconnect_0 is a SIBLING of vortex_afu_0 under top_i/slash, not a child
    # of afu_wrap, so an afu_wrap-only pblock leaves it free to straddle. It was
    # measured owning ALL TWENTY endpoints of the ten worst paths at -0.260 ns,
    # every one of them crossing SLR1->SLR2 for ~0.45 ns of inter-SLR
    # compensation on 0.18-0.34 ns of actual logic.
    #
    # Everything else under top_i/slash is deliberately left alone: the NoC
    # wrappers (hbm_vnoc_*) hold hard blocks that must stay at their fixed sites,
    # and the axi_register_slice_*term_* tie-offs terminate unused shell ports
    # whose placement should follow the ports, not the user logic.
    # Query as one call: expanding object lists with {*} flattens them to bare
    # name strings, and get_property then rejects them with
    # "Invalid option value ... specified for 'object'".
    set scope "ic"
    if {[info exists ::env(VX_DFX_SLR_SCOPE)]} { set scope $::env(VX_DFX_SLR_SCOPE) }

    set candidates {}
    if {$scope eq "ic"} {
        # The interconnect blocks, as ONE pblock.
        #
        # They were first given a pblock each. Vivado rejects that: two pblocks
        # created in the same DFX parent context are treated as nested, and the
        # second fails DRC HDPR-23 ("child Pblock is not contained by the parent
        # Pblock") even though both carry identical ranges. Since both blocks
        # target the same SLR -- their anchors are both there -- a single pblock
        # is exactly equivalent and sidesteps the nesting entirely.
        set ic [get_cells -quiet [list top_i/slash/hbm_sc_01 \
                                       top_i/slash/smartconnect_0]]
        if {[llength $ic] > 0} { lappend candidates [list shell_ic $ic] }
    } else {
        # DISPROVEN scopes, kept only for re-measurement. See the header.
        set soft [get_cells -quiet [list top_i/slash/vortex_afu_0 \
                                         top_i/slash/smartconnect_0]]
        if {[llength $soft] > 0} { lappend candidates [list rm_soft_logic $soft] }

        set afu [get_cells -quiet -hierarchical -filter {NAME =~ */afu_wrap && IS_PRIMITIVE == 0}]
        set cp  [get_cells -quiet -hierarchical -filter {NAME =~ */afu_wrap/cp_core && IS_PRIMITIVE == 0}]
        if {[llength $afu] == 1} { lappend candidates [list afu_wrap $afu] }
        if {[llength $cp]  == 1} { lappend candidates [list cp_core  $cp]  }
    }
    if {[llength $candidates] == 0} {
        puts "DFX-SLR: no confinable scope matched; skipping."
        return
    }

    # -- per-SLR clipped ranges, then one exact capacity pass over the region --
    array set RANGES {}
    array set BB {}
    foreach slr $slrs {
        set nm [get_property NAME $slr]
        array unset B ; array set B {}
        dfx_slr_bounds $slr B
        foreach k [array names B] { set BB($nm,$k) $B($k) }
        set RANGES($nm) [dfx_clip_ranges $ranges B]
    }
    array set CAPS {}
    dfx_region_capacity $pb $slrs BB CAPS

    array set ANCHORS {}
    dfx_noc_sites $pb $slrs BB ANCHORS

    foreach slr $slrs {
        set nm [get_property NAME $slr]
        array unset C ; array set C {}
        foreach r {LUT BRAM DSP URAM} { set C($r) $CAPS($nm,$r) }
        puts "DFX-SLR:   $nm capacity = [dfx_fmt C] ([llength $RANGES($nm)] range(s)), NoC sites = $ANCHORS($nm)"
    }

    # -- pick the largest scope that fits, on the SLR that fits it best --
    set forced ""
    if {[info exists ::env(VX_DFX_SLR)]} { set forced $::env(VX_DFX_SLR) }

    foreach cand $candidates {
        lassign $cand label cells
        array unset F ; array set F {}
        dfx_footprint $cells F
        puts "DFX-SLR: $label footprint = [dfx_fmt F] ([llength $cells] cell(s))"

        # Rank legal SLRs by NoC anchors first, fill second. Capacity only breaks
        # ties: an SLR with 6% more room but none of the design's memory ports
        # turns every HBM access into a boundary round trip.
        set best "" ; set best_peak 2.0 ; set best_anchors -1
        foreach slr $slrs {
            set nm [get_property NAME $slr]
            if {$forced ne "" && $nm ne $forced} { continue }
            if {[llength $RANGES($nm)] == 0} { continue }
            set fits 1 ; set peak 0.0
            foreach r {LUT BRAM DSP URAM} {
                if {$F($r) == 0} { continue }
                set budget [expr {int($ceil * $CAPS($nm,$r))}]
                if {$F($r) > $budget} { set fits 0 ; break }
                if {$CAPS($nm,$r) > 0} {
                    set f [expr {double($F($r)) / $CAPS($nm,$r)}]
                    if {$f > $peak} { set peak $f }
                }
            }
            if {!$fits} { continue }
            if {$ANCHORS($nm) > $best_anchors ||
                ($ANCHORS($nm) == $best_anchors && $peak < $best_peak)} {
                set best $nm ; set best_peak $peak ; set best_anchors $ANCHORS($nm)
            }
        }

        if {$best eq ""} {
            puts "DFX-SLR: $label does not fit one SLR under a [expr {$ceil*100}]% ceiling; trying a smaller scope."
            continue
        }

        # Exactly one pblock is ever created -- see the ic-scope note above for
        # why a second one fails DRC HDPR-23 as a nested pblock. Candidates are
        # alternatives, largest-first; the first that fits wins and returns.
        set pbname "pblock_slrfit_$label"
        create_pblock $pbname
        resize_pblock [get_pblocks $pbname] -add $RANGES($best)
        add_cells_to_pblock [get_pblocks $pbname] $cells
        foreach c $cells { puts "DFX-SLR:   + [get_property NAME $c]" }
        set_property IS_SOFT FALSE [get_pblocks $pbname]
        puts [format "DFX-SLR: confined %s to %s (peak fill %.1f%%, %d NoC anchor(s) there)" \
            $label $best [expr {100*$best_peak}] $best_anchors]
        puts "DFX-SLR: interface/NoC/clocking cells left unconstrained by design."
        return
    }

    puts "DFX-SLR: no scope fits a single SLR; leaving placement to the automatic partitioner."
}

dfx_slr_confine_run
