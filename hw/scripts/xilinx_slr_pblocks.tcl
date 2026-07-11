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
# Capacity model. Every resource an SLR can run out of is budgeted, not just
# LUTs: a Vortex socket is DSP-dense (FPU + tensor cores) and cache-dense
# (BRAM), so a LUT-balanced partition can still overflow an SLR's DSP or BRAM
# columns and fail placement. An atom fits an SLR only if it has headroom in
# ALL of LUT, BRAM, DSP and URAM.
#   LUT  : LUT primitives            vs SLICE sites x 8
#   BRAM : RAMB18 equivalents        vs RAMB18 sites (a RAMB36/FIFO36 = 2)
#   DSP  : DSP primitives            vs DSP sites
#   URAM : URAM primitives           vs URAM sites
# Soft logic gets the lower ceiling: LUT fill is what drives routing congestion,
# whereas hard blocks sit in fixed columns and only need to be placeable, so
# they are allowed to run closer to full.
#
# Controls (environment):
#   USE_SLR_PBLOCKS=1            apply the floorplan (default: off — Vivado's own
#                                 SSI partitioner runs unless a design is known to
#                                 need spreading, and a floorplan can only hurt a
#                                 design that already places and routes)
#   SLR_PBLOCKS_FILL_CEILING=f   per-SLR LUT fill cap, 0..1 (default 0.85 — the
#                                 remaining ~15% is left for the Xilinx static
#                                 shell / platform IP that also lands in the SLR)
#   SLR_PBLOCKS_HARD_CEILING=f   per-SLR BRAM/DSP/URAM fill cap, 0..1 (default 0.95)
#   SLR_PBLOCKS_ANCHOR=i         index of the anchor SLR (default: the middle one)
#   SLR_PBLOCKS_FORCE=1          apply even if the guardrail rejects (debug only)
#
# Binning model:
#   anchor SLR (the middle one) : globally-shared blocks — l2cache, l3cache, and
#                                 the shared control units (kmu, gbar) that fan
#                                 out register-bounded to every core. Every
#                                 compute atom talks to these, and only the
#                                 middle SLR is adjacent to every other SLR.
#   compute atoms (bin-packed)  : one atom per socket (its cores + private L1s,
#                                 kept whole so the unregistered core<->L1 path
#                                 never crosses an SLR); and each cluster
#                                 extension core BUNDLED WITH ITS PRIVATE CACHE
#                                 (tex+tcache, raster+rcache, om+ocache,
#                                 rtu+rtcache, dxa) so a wide core<->cache bus is
#                                 never split across a boundary.
# Atoms are packed worst-resource-first, best-fit: each goes to the legal SLR
# whose worst resource ends up lowest. "Worst resource" is the largest fraction
# an atom would consume of the tightest SLR's supply of any ONE resource —
# ordering on LUTs alone defers a DSP-dense atom until the DSP columns are gone,
# and first-fit packs the design into a corner once the anchor has claimed one
# SLR's BRAM.

# Resources budgeted, in report order. LUT is soft logic; the rest are hard
# blocks and share the hard-block ceiling.
set ::SLR_RES {LUT BRAM DSP URAM}

# Escape regex metacharacters in a literal hierarchical name so it can anchor a
# child get_cells -regexp query.
proc slr_re_escape {s} {
    return [string map {\\ \\\\ . \\. \[ \\\[ \] \\\] ( \\( ) \\) + \\+ * \\* ? \\? ^ \\^ $ \\$ | \\| \{ \\\{ \} \\\}} $s]
}

# Number of names in a pre-collected list that lie under a hierarchical prefix.
proc slr_under {names prefix} {
    set p "${prefix}/"
    set n 0
    foreach c $names {
        if {[string first $p $c] == 0} { incr n }
    }
    return $n
}

# Footprint of a hierarchical cell as {LUT BRAM DSP URAM}.
# LUTs are counted with a hierarchical query (there are too many to pre-collect);
# hard blocks are attributed from lists gathered once, since there are only a few
# thousand of them and a query per cell per resource would dominate runtime.
proc slr_foot {cell} {
    if {$cell eq ""} { return {0 0 0 0} }
    set p [get_property NAME $cell]
    set lut [llength [get_cells -quiet -hierarchical \
        -filter "PRIMITIVE_SUBGROUP == LUT && NAME =~ {${p}/*}"]]
    set bram [expr {2 * [slr_under $::SLR_B36 $p] + [slr_under $::SLR_B18 $p]}]
    return [list $lut $bram [slr_under $::SLR_DSP $p] [slr_under $::SLR_URAM $p]]
}

proc slr_foot_add {a b} {
    set r {}
    foreach x $a y $b { lappend r [expr {$x + $y}] }
    return $r
}

proc slr_pblocks_run {} {
    # -- gate --
    if {!([info exists ::env(USE_SLR_PBLOCKS)] && $::env(USE_SLR_PBLOCKS) eq "1")} {
        return
    }
    set ceil 0.85
    if {[info exists ::env(SLR_PBLOCKS_FILL_CEILING)]} {
        set ceil $::env(SLR_PBLOCKS_FILL_CEILING)
    }
    set hceil 0.95
    if {[info exists ::env(SLR_PBLOCKS_HARD_CEILING)]} {
        set hceil $::env(SLR_PBLOCKS_HARD_CEILING)
    }
    array set CEIL {}
    foreach r $::SLR_RES {
        set CEIL($r) [expr {$r eq "LUT" ? $ceil : $hceil}]
    }
    set force [expr {[info exists ::env(SLR_PBLOCKS_FORCE)] && $::env(SLR_PBLOCKS_FORCE) eq "1"}]

    set slrs [lsort [get_slrs]]
    set nslr [llength $slrs]
    if {$nslr < 2} {
        puts "SLR-PBLOCKS: device has $nslr SLR(s); floorplan skipped"
        return
    }

    # ---- capacity model: real RM-usable sites per SLR, per resource ----
    # Use the platform's own dynamic-region pblock, not raw SLR site counts. The
    # static shell occupies part of each SLR; raw counts over-estimate by ~14%.
    array set CAP {}
    foreach slr $slrs {
        set nm [get_property NAME $slr]
        set src [get_pblocks -quiet "pblock_dynamic_${nm}"]
        if {$src eq "" || [llength [get_sites -quiet -of_objects $src -filter {SITE_TYPE =~ SLICE*}]] == 0} {
            set src $slr
        }
        set CAP($nm,LUT)  [expr {[llength [get_sites -quiet -of_objects $src -filter {SITE_TYPE =~ SLICE*}]] * 8}]
        set CAP($nm,BRAM) [llength [get_sites -quiet -of_objects $src -filter {SITE_TYPE =~ RAMB18*}]]
        set CAP($nm,DSP)  [llength [get_sites -quiet -of_objects $src -filter {SITE_TYPE =~ DSP*}]]
        set CAP($nm,URAM) [llength [get_sites -quiet -of_objects $src -filter {SITE_TYPE =~ URAM*}]]
    }

    # ---- hard-block inventory, gathered once ----
    set ::SLR_B36  [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB36* || REF_NAME =~ FIFO36*}]
    set ::SLR_B18  [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB18* || REF_NAME =~ FIFO18*}]
    set ::SLR_DSP  [get_cells -quiet -hierarchical -filter {REF_NAME =~ DSP*}]
    set ::SLR_URAM [get_cells -quiet -hierarchical -filter {REF_NAME =~ URAM*}]

    proc cells_re {re} { return [get_cells -quiet -hierarchical -regexp $re] }

    # ---- collect atoms ----
    # Compute atoms: {label foot {cell ...}}.  Anchor: shared caches/controls.
    set atoms  {}
    set anchor_cells {}
    set anchor_foot {0 0 0 0}
    # An l2 is matched both here and by the per-cluster sweep below; a cell may
    # only be counted into the anchor once or its footprint is doubled.
    array set anchor_seen {}

    # Globally-shared blocks -> anchor.
    foreach re {{.*/l3cache} {.*/l2cache} {.*/kmu} {.*/gbar_unit}} {
        foreach c [cells_re $re] {
            set n [get_property NAME $c]
            if {[info exists anchor_seen($n)]} { continue }
            set anchor_seen($n) 1
            lappend anchor_cells $c
            set anchor_foot [slr_foot_add $anchor_foot [slr_foot $c]]
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
        set cl_total [lindex [slr_foot $cl] 0]
        set acc 0

        # sockets (cores + private L1s), whole.
        foreach s [cells_re "[slr_re_escape $clp]/g_sockets\\\[\[0-9\]+\\\]\\.socket"] {
            set g {}
            foreach c [cells_re "[slr_re_escape [get_property NAME $s]]/g_cores\\\[\[0-9\]+\\\]\\.core"] { lappend g $c }
            foreach c [cells_re "[slr_re_escape [get_property NAME $s]]/(icache|dcache)"]                { lappend g $c }
            if {[llength $g] == 0} { set g [list $s] }
            set f {0 0 0 0}
            foreach c $g { set f [slr_foot_add $f [slr_foot $c]] }
            lappend atoms [list "socket:[get_property NAME $s]" $f $g]
            incr acc [lindex $f 0]
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
                set f [slr_foot $core]
                if {$cachere ne ""} {
                    foreach ch [get_cells -quiet -hierarchical -regexp $cachere] {
                        if {[string first "${clp}/" [get_property NAME $ch]] != 0} { continue }
                        lappend g $ch
                        set f [slr_foot_add $f [slr_foot $ch]]
                    }
                }
                lappend atoms [list "[get_property NAME $core]+cache" $f $g]
                incr acc [lindex $f 0]
            }
        }

        # cluster-shared caches not bundled above (e.g. an l2 inside the cluster).
        # These count toward this cluster's coverage whether or not the anchor
        # sweep already claimed them.
        foreach ch [get_cells -quiet -hierarchical -regexp {.*/(l2cache)}] {
            set n [get_property NAME $ch]
            if {[string first "${clp}/" $n] != 0} { continue }
            set f [slr_foot $ch]
            incr acc [lindex $f 0]
            if {[info exists anchor_seen($n)]} { continue }
            set anchor_seen($n) 1
            lappend anchor_cells $ch
            set anchor_foot [slr_foot_add $anchor_foot $f]
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
    puts "SLR-PBLOCKS: $nslr SLRs; $n_compute compute atom(s); LUT ceiling [expr {$ceil*100}]%, hard-block ceiling [expr {$hceil*100}]%"
    puts "SLR-PBLOCKS: anchor shared caches = [slr_fmt $anchor_foot]"
    foreach slr $slrs {
        set nm [get_property NAME $slr]
        set c {}
        foreach r $::SLR_RES { lappend c $CAP($nm,$r) }
        puts "SLR-PBLOCKS:   cap $nm = [slr_fmt $c]"
    }

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

    # ---- guardrail 2: a demanded resource with no site inventory means the site
    #      naming is not what this model assumes; its budget would be a no-op. ----
    set total $anchor_foot
    foreach a $atoms { set total [slr_foot_add $total [lindex $a 1]] }
    set ri 0
    foreach r $::SLR_RES {
        if {[lindex $total $ri] > 0} {
            foreach slr $slrs {
                if {$CAP([get_property NAME $slr],$r) == 0} {
                    puts "SLR-PBLOCKS: REJECT — design uses $r but no $r sites found on [get_property NAME $slr]."
                    puts "SLR-PBLOCKS: applying NO floorplan; automatic SSI partitioner will run."
                    if {!$force} { return }
                }
            }
        }
        incr ri
    }

    # ---- bin-pack: anchor pinned; atoms packed worst-resource-first, BEST-FIT. ----
    # The anchor holds the shared L2/L3, which every compute atom talks to, so it
    # is pinned to the MIDDLE SLR: the SLR list is the physical adjacency chain
    # (SLR0-SLR1-SLR2), and only the middle is adjacent to every other SLR. An
    # anchor on an end SLR would leave the far SLR two hops from the shared cache
    # — and the end-to-end pair (SLR0<->SLR2) has NO direct SLLs, so that path
    # cannot be Laguna-pipelined at all. The one interface this pushes off-SLR is
    # the anchor's memory port, which crosses a single adjacent boundary.
    array set FILL {}
    foreach slr $slrs {
        foreach r $::SLR_RES { set FILL([get_property NAME $slr],$r) 0 }
    }
    set aidx [expr {$nslr / 2}]
    if {[info exists ::env(SLR_PBLOCKS_ANCHOR)]} {
        set aidx $::env(SLR_PBLOCKS_ANCHOR)
    }
    set anchor_nm [get_property NAME [lindex $slrs $aidx]]
    set ri 0
    foreach r $::SLR_RES {
        incr FILL($anchor_nm,$r) [lindex $anchor_foot $ri]
        incr ri
    }

    # The anchor is pinned, not packed, so nothing else would ever test it: it
    # has to fit the anchor SLR on its own before any atom is placed beside it.
    set feasible 1
    foreach r $::SLR_RES {
        if {$FILL($anchor_nm,$r) > int($CEIL($r) * $CAP($anchor_nm,$r))} {
            puts "SLR-PBLOCKS: INFEASIBLE — anchor caches use $FILL($anchor_nm,$r) $r, over the ceiling on $anchor_nm ($CAP($anchor_nm,$r) $r)."
            set feasible 0
        }
    }

    # Order by the atom's worst resource: the largest fraction of any one
    # resource it would take from the SLR with the least of it. Ordering on LUT
    # alone would defer a DSP-dense atom until the DSP columns are gone.
    set scored {}
    foreach a $atoms {
        lassign $a label foot cells
        set worst 0.0
        set ri 0
        foreach r $::SLR_RES {
            set mincap 0
            foreach slr $slrs {
                set c $CAP([get_property NAME $slr],$r)
                if {$mincap == 0 || $c < $mincap} { set mincap $c }
            }
            if {$mincap > 0} {
                set f [expr {double([lindex $foot $ri]) / $mincap}]
                if {$f > $worst} { set worst $f }
            }
            incr ri
        }
        lappend scored [list $worst $label $foot $cells]
    }
    set order [lsort -real -index 0 -decreasing $scored]

    set assign {}
    foreach a $order {
        if {!$feasible} { break }
        lassign $a worst label foot cells
        # Best-fit: of the SLRs with room under every ceiling, take the one whose
        # WORST resource ends up lowest. First-fit would cram each atom into the
        # first legal SLR and leave the last atoms with nowhere balanced to go —
        # with one SLR's BRAM already spent on the anchor, that packs the design
        # into a corner and fails on the tightest resource (here, DSP).
        set best ""
        set best_peak 2.0
        foreach slr $slrs {
            set nm [get_property NAME $slr]
            set fits 1
            set peak 0.0
            set ri 0
            foreach r $::SLR_RES {
                set head [expr {int($CEIL($r) * $CAP($nm,$r)) - $FILL($nm,$r)}]
                if {[lindex $foot $ri] > $head} { set fits 0 ; break }
                if {$CAP($nm,$r) > 0} {
                    set f [expr {double($FILL($nm,$r) + [lindex $foot $ri]) / ($CEIL($r) * $CAP($nm,$r))}]
                    if {$f > $peak} { set peak $f }
                }
                incr ri
            }
            if {$fits && $peak < $best_peak} { set best $nm ; set best_peak $peak }
        }
        if {$best eq ""} {
            puts "SLR-PBLOCKS: INFEASIBLE — atom $label ([slr_fmt $foot]) exceeds ceiling headroom on every SLR."
            set feasible 0
            break
        }
        set ri 0
        foreach r $::SLR_RES {
            incr FILL($best,$r) [lindex $foot $ri]
            incr ri
        }
        lappend assign [list $label $cells $best]
    }

    # ---- report projection ----
    puts "SLR-PBLOCKS: projected per-SLR fill —"
    foreach slr $slrs {
        set nm [get_property NAME $slr]
        set line ""
        foreach r $::SLR_RES {
            set pct [expr {$CAP($nm,$r) > 0 ? 100.0*$FILL($nm,$r)/$CAP($nm,$r) : 0.0}]
            append line [format "  %s %d/%d=%.1f%%" $r $FILL($nm,$r) $CAP($nm,$r) $pct]
        }
        puts "SLR-PBLOCKS:   $nm :$line"
    }

    # ---- guardrail 3: feasibility ----
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
        puts "SLR-PBLOCKS: anchor $anchor_nm <- [llength $anchor_cells] shared cache/control cell(s)"
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

# Render a {LUT BRAM DSP URAM} footprint for the log.
proc slr_fmt {foot} {
    set out {}
    set i 0
    foreach r $::SLR_RES {
        lappend out "$r [lindex $foot $i]"
        incr i
    }
    return [join $out ", "]
}

slr_pblocks_run
