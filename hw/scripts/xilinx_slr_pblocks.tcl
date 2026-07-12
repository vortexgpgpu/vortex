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

# Per-SLR floorplan for Vortex on SSI (multi-SLR) Alveo devices.
#
# Purpose. Vivado's automatic SSI partitioner minimizes inter-SLR wire count
# (SLLs). It does NOT spread logic to relieve routing congestion; a wire-minimal
# partition can still pack one SLR dense enough to reach congestion level 7 and
# fail to route. This floorplan exists for the one case the automatic flow will
# not handle: deliberately distributing large, tightly-connected blocks across
# SLRs to stay under a per-SLR fill ceiling, trading a few extra SLLs for
# routability.
#
# Safety contract. This script never applies a floorplan that would fail
# placement. It projects per-SLR fill before committing any pblock and, if the
# projection is infeasible OR the atom model does not reconcile against the real
# netlist, it applies NOTHING and lets the automatic partitioner run. A bad
# floorplan is strictly worse than none (it fails after ~10 min of placement
# instead of routing sub-optimally), so the default on any doubt is to stand
# down.
#
# Controls (environment):
#   USE_SLR_PBLOCKS=0            disable entirely (default: enabled)
#   SLR_PBLOCKS_FILL_CEILING=f   per-SLR LUT fill cap, 0..1 (default 0.65)
#   SLR_PBLOCKS_FORCE=1          apply even if the guardrail rejects (debug only)
#
# Binning model:
#   anchor SLR (SLR0, HBM side) : globally-shared blocks — l2cache, l3cache, and
#                                 the shared control units (kmu, gbar) that fan
#                                 out register-bounded to every core.
#   compute atoms (FFD-packed)  : one atom per socket (its cores + private L1s,
#                                 kept whole so the unregistered core<->L1 path
#                                 never crosses an SLR); and each cluster
#                                 extension core BUNDLED WITH ITS PRIVATE CACHE
#                                 (tex+tcache, raster+rcache, om+ocache,
#                                 rtu+rtcache, dxa) so a wide core<->cache bus is
#                                 never split across a boundary.
# Atoms are packed largest-first into the SLR with the most remaining headroom
# under the ceiling, starting away from the cache-loaded anchor.

proc slr_pblocks_run {} {
    # -- gate --
    if {[info exists ::env(USE_SLR_PBLOCKS)] && $::env(USE_SLR_PBLOCKS) eq "0"} {
        return
    }
    set ceil 0.65
    if {[info exists ::env(SLR_PBLOCKS_FILL_CEILING)]} {
        set ceil $::env(SLR_PBLOCKS_FILL_CEILING)
    }
    set force [expr {[info exists ::env(SLR_PBLOCKS_FORCE)] && $::env(SLR_PBLOCKS_FORCE) eq "1"}]

    set slrs [lsort [get_slrs]]
    set nslr [llength $slrs]
    if {$nslr < 2} {
        puts "SLR-PBLOCKS: device has $nslr SLR(s); floorplan skipped"
        return
    }

    # ---- capacity model: real RM-usable LUT sites per SLR ----
    # Use the platform's own dynamic-region pblock, not raw SLR site counts. The
    # static shell occupies part of each SLR; raw counts over-estimate by ~14%.
    array set CAP {}
    foreach slr $slrs {
        set nm [get_property NAME $slr]
        set pb [get_pblocks -quiet "pblock_dynamic_${nm}"]
        set sites {}
        if {$pb ne ""} {
            set sites [get_sites -quiet -of_objects $pb -filter {SITE_TYPE =~ SLICE*}]
        }
        if {[llength $sites] == 0} {
            set sites [get_sites -quiet -of_objects $slr -filter {SITE_TYPE =~ SLICE*}]
        }
        set CAP($nm) [expr {[llength $sites] * 8}]
    }

    # ---- LUT footprint of a hierarchical cell (primitive count) ----
    proc lut_of {cell} {
        if {$cell eq ""} { return 0 }
        set p [get_property NAME $cell]
        return [llength [get_cells -quiet -hierarchical \
            -filter "PRIMITIVE_SUBGROUP == LUT && NAME =~ {${p}/*}"]]
    }
    proc cells_re {re} { return [get_cells -quiet -hierarchical -regexp $re] }

    # ---- collect atoms ----
    # Compute atoms: {label lut {cell ...}}.  Anchor: shared caches/controls.
    set atoms  {}
    set anchor_cells {}
    set anchor_lut 0

    # Globally-shared blocks -> anchor.
    foreach re {{.*/l3cache} {.*/l2cache} {.*/kmu} {.*/gbar_unit}} {
        foreach c [cells_re $re] {
            lappend anchor_cells $c
            incr anchor_lut [lut_of $c]
        }
    }

    # Per-cluster reconciliation: everything sizeable inside a cluster must be
    # captured by an atom or a cluster cache. If it is not, the atom model is
    # stale (RTL changed) and we must NOT floorplan on a partial view.
    set clusters [cells_re {.*/g_clusters\[[0-9]+\]\.cluster}]
    if {[llength $clusters] == 0} {
        # Non-clustered top (single cluster inlined): treat vortex as one region.
        set clusters [cells_re {.*/(Vortex|vortex)$}]
    }

    set reconcile_ok 1
    foreach cl $clusters {
        set clp [get_property NAME $cl]
        set cl_total [lut_of $cl]
        set acc 0

        # sockets (cores + private L1s), whole.
        foreach s [cells_re "[slr_re_escape $clp]/g_sockets\\\[\[0-9\]+\\\]\\.socket"] {
            set g {}
            foreach c [cells_re "[slr_re_escape [get_property NAME $s]]/g_cores\\\[\[0-9\]+\\\]\\.core"] { lappend g $c }
            foreach c [cells_re "[slr_re_escape [get_property NAME $s]]/(icache|dcache)"]                { lappend g $c }
            if {[llength $g] == 0} { set g [list $s] }
            set l 0; foreach c $g { incr l [lut_of $c] }
            lappend atoms [list "socket:[get_property NAME $s]" $l $g]
            incr acc $l
        }

        # extension core BUNDLED with its private cache.
        foreach {corere cachere} {
            {.*\.tex_core}    {.*/tcache}
            {.*\.raster_core} {.*/rcache}
            {.*\.om_core}     {.*/ocache}
            {.*\.rtu_core}    {.*/rtcache}
            {.*/dxa_core}     {}
        } {
            foreach core [get_cells -quiet -hierarchical -regexp $corere] {
                # Literal prefix match (not a glob -filter: hierarchical names
                # contain [0] which a glob treats as a char class, not literal).
                if {[string first "${clp}/" [get_property NAME $core]] != 0} { continue }
                set g [list $core]
                set l [lut_of $core]
                if {$cachere ne ""} {
                    foreach ch [get_cells -quiet -hierarchical -regexp $cachere] {
                        if {[string first "${clp}/" [get_property NAME $ch]] != 0} { continue }
                        lappend g $ch; incr l [lut_of $ch]
                    }
                }
                lappend atoms [list "[get_property NAME $core]+cache" $l $g]
                incr acc $l
            }
        }

        # cluster-shared caches not bundled above (e.g. an l2 inside the cluster).
        foreach ch [get_cells -quiet -hierarchical -regexp {.*/(l2cache)}] {
            if {[string first "${clp}/" [get_property NAME $ch]] != 0} { continue }
            lappend anchor_cells $ch; incr anchor_lut [lut_of $ch]; incr acc [lut_of $ch]
        }

        if {$cl_total > 0} {
            set frac [expr {double($acc)/$cl_total}]
            puts [format "SLR-PBLOCKS: reconcile %s : accounted %d / %d LUT = %.1f%%" $clp $acc $cl_total [expr {100*$frac}]]
            # Reject on either side: <85% means an atom is missing (patterns stale);
            # >115% means cells are double-counted. Both mean the model is unsafe.
            if {$frac < 0.85 || $frac > 1.15} { set reconcile_ok 0 }
        }
    }

    set n_compute [llength $atoms]
    puts "SLR-PBLOCKS: $nslr SLRs; anchor caches=$anchor_lut LUT; $n_compute compute atom(s); ceiling [expr {$ceil*100}]%"
    foreach slr $slrs { puts [format "SLR-PBLOCKS:   cap %s = %d LUT" [get_property NAME $slr] $CAP([get_property NAME $slr])] }

    if {$n_compute == 0 && [llength $anchor_cells] == 0} {
        puts "SLR-PBLOCKS: no target instances matched; floorplan not applied"
        return
    }

    # ---- guardrail 1: atom model must reconcile against the netlist ----
    if {!$reconcile_ok && !$force} {
        puts "SLR-PBLOCKS: REJECT — atom model covers <85% of a cluster's LUTs (stale patterns?)."
        puts "SLR-PBLOCKS: applying NO floorplan; automatic SSI partitioner will run."
        return
    }

    # ---- FFD bin-pack: anchor pre-loaded with caches; pack atoms largest-first
    #      into the SLR with most headroom under the ceiling. ----
    array set FILL {}
    foreach slr $slrs { set FILL([get_property NAME $slr]) 0 }
    set anchor_nm [get_property NAME [lindex $slrs 0]]
    incr FILL($anchor_nm) $anchor_lut

    # largest-first
    set order [lsort -integer -index 1 -decreasing $atoms]

    set assign {}
    set feasible 1
    foreach a $order {
        lassign $a label lut cells
        # pick SLR with the most absolute headroom that still honors the ceiling
        set best ""; set best_head -1
        foreach slr $slrs {
            set nm [get_property NAME $slr]
            set cap_ceil [expr {int($ceil * $CAP($nm))}]
            set head [expr {$cap_ceil - $FILL($nm)}]
            if {$lut <= $head && $head > $best_head} { set best $nm; set best_head $head }
        }
        if {$best eq ""} {
            puts [format "SLR-PBLOCKS: INFEASIBLE — atom %s (%d LUT) exceeds ceiling headroom on every SLR." $label $lut]
            set feasible 0
            break
        }
        incr FILL($best) $lut
        lappend assign [list $label $cells $best]
    }

    # ---- report projection ----
    puts "SLR-PBLOCKS: projected per-SLR fill —"
    foreach slr $slrs {
        set nm [get_property NAME $slr]
        puts [format "SLR-PBLOCKS:   %s : %d / %d LUT = %.1f%%" $nm $FILL($nm) $CAP($nm) [expr {100.0*$FILL($nm)/$CAP($nm)}]]
    }

    # ---- guardrail 2: feasibility ----
    if {!$feasible && !$force} {
        puts "SLR-PBLOCKS: REJECT — no ceiling-legal assignment exists (design too dense to floorplan safely)."
        puts "SLR-PBLOCKS: applying NO floorplan; automatic SSI partitioner will run."
        return
    }

    # ---- apply ----
    create_pblock pb_slr_anchor
    resize_pblock pb_slr_anchor -add [lindex $slrs 0]
    if {[llength $anchor_cells]} {
        add_cells_to_pblock pb_slr_anchor $anchor_cells
        puts "SLR-PBLOCKS: anchor $anchor_nm <- [llength $anchor_cells] shared cache/control cell(s), $anchor_lut LUT"
    }
    set idx 0
    foreach as $assign {
        lassign $as label cells slr
        set pb "pb_atom${idx}"
        create_pblock $pb
        resize_pblock $pb -add $slr
        add_cells_to_pblock $pb $cells
        puts "SLR-PBLOCKS: $label -> $slr"
        incr idx
    }
    puts "SLR-PBLOCKS: applied [expr {$idx+1}] pblock(s)."
}

# Escape regex metacharacters in a literal hierarchical name so it can anchor a
# child get_cells -regexp query.
proc slr_re_escape {s} {
    return [string map {\\ \\\\ . \\. \[ \\\[ \] \\\] ( \\( ) \\) + \\+ * \\* ? \\? ^ \\^ $ \\$ | \\| \{ \\\{ \} \\\}} $s]
}

slr_pblocks_run
