# Steer the RM's memory-path NoC ingress units onto the SLR that holds the HBM,
# so the whole user clock domain can live on one die.
#
# WHY. Every failing path in the 300 MHz domain is charged inter-SLR clock
# compensation, (DCD - CCD) * 0.1. On the fast300 worst path that is 0.456 ns
# out of a 3.333 ns budget, against a deficit of 0.260 ns. It is that large
# because the two endpoints share almost none of the clock tree: the clock root
# is in SLR0 and they sit on different dies. Put both endpoints on one die and
# CCD rises toward DCD; at CCD ~5.5 the charge falls to ~0.10 ns. That is ~0.35
# ns recovered on EVERY path at once -- more than the whole gap, and the only
# lever found that helps all five failing structures simultaneously.
#
# WHY THE DOMAIN IS SPLIT. Measured on the routed checkpoint:
#
#   NOC_NMU_HBM2E sites (HBM ingress)   SLR0: 0   SLR1: 0   SLR2: 64
#
# All HBM is in SLR2, so HOST_TAG=HBM1 pins the host path and both
# SmartConnects there (~87%). But MEM_TAG=MEM maps to eight vNOC ports
# (HBM_VNOC_INI_00..07 -> hbm_vnoc_00..07), and those are NOC_NMU512 -- general
# NoC ingress, NOT fixed silicon. SLR1 and SLR2 have 24 such sites each. The
# NoC compiler put 7 of the 8 in SLR1, and the AFU followed its widest master
# there (99.8% SLR1). One placeable choice is holding the design off the die
# where all of its memory physically lives.
#
# This is the U55C topology question. There, HBM and host both anchor to SLR0,
# the kernel follows, and there is no crossing to pay for. Here the anchors are
# split, and nothing has ever overridden that.
#
# WHAT THIS DOES. Relocates the hbm_vnoc_* NMUs onto free NOC_NMU512 sites in
# the target SLR. It does NOT touch the HBM2E NMUs (fixed silicon) or the
# static shell's NoC.
#
# RESULT: THIS DOES NOT WORK. IT CANNOT WORK FROM THE VORTEX BUILD.
#
# The cells look steerable and are not. At the pre-opt hook every vNOC NMU
# reports LOC="" and IS_LOC_FIXED=0, and set_property LOC succeeds on all
# eight. opt_design then rejects them:
#
#   ERROR: [DRC HDPR-122] DFX NoC location validity check: NoC instance
#   location conflict. NoC logical instance '.../hbm_vnoc_00/.../NOC_NMU512_INST'
#   is placed at site 'NOC_NMU512_X0Y17' which is breaking the NoC compiler
#   rules for locked paths. The allowed location for this NoC instance is site
#   'NOC_NMU512_X2Y9'.
#
# Each RM NoC port has EXACTLY ONE allowed site, fixed by the prebuilt static
# shell's locked NoC routes, and it is the site it already had:
#
#   hbm_vnoc_00 -> NOC_NMU512_X2Y9   (SLR1)
#   hbm_vnoc_01 -> NOC_NMU512_X2Y15  (SLR2)
#   hbm_vnoc_02 -> NOC_NMU512_X0Y12  (SLR1)
#   hbm_vnoc_03 -> NOC_NMU512_X1Y8   (SLR1)
#
# So the design's SLR topology is decided by static_shell_slash.dcp, not by
# anything in this repo: MEM anchors in SLR1, all HBM (and therefore HOST_TAG
# and both SmartConnects) in SLR2, and the ~0.456 ns of inter-SLR clock
# compensation that follows is structural. Moving it requires rebuilding and
# reflashing the AVED static shell with a different NoC solution.
#
# Kept as documentation of the negative result and of how to interrogate the
# NoC placement rules. Do not re-enable expecting a different answer.
#
# Controls (environment):
#   VX_NOC_STEER=1          enable (default: OFF)
#   VX_NOC_STEER_SLR=SLR2   target SLR (default SLR2, where the HBM is)
#   VX_NOC_STEER_ABORT=1    abort the build if the NMUs cannot be moved
#                           (default 1: a negative result should be cheap)

proc noc_steer_run {} {
    if {!([info exists ::env(VX_NOC_STEER)] && $::env(VX_NOC_STEER) eq "1")} {
        puts "NOC-STEER: not applied (set VX_NOC_STEER=1 to enable)"
        return
    }
    set target "SLR2"
    if {[info exists ::env(VX_NOC_STEER_SLR)]} { set target $::env(VX_NOC_STEER_SLR) }
    set do_abort 1
    if {[info exists ::env(VX_NOC_STEER_ABORT)]} {
        set do_abort [expr {$::env(VX_NOC_STEER_ABORT) eq "1"}]
    }

    set slr [get_slrs -quiet $target]
    if {[llength $slr] != 1} {
        puts "NOC-STEER: SLR '$target' not found; skipping."
        return
    }

    # ---- the cells to move: the RM's vNOC ingress units ----
    set nmus [get_cells -quiet -hier \
                -filter {NAME =~ "top_i/slash/hbm_vnoc_*" && PRIMITIVE_TYPE =~ *NOC_NMU512*}]
    if {[llength $nmus] == 0} {
        puts "NOC-STEER: no hbm_vnoc NMU cells found; skipping."
        return
    }
    puts "NOC-STEER: found [llength $nmus] vNOC ingress unit(s)"

    # ---- current placement, and whether it is already nailed down ----
    set to_move {}
    foreach c $nmus {
        set loc   [get_property -quiet LOC $c]
        set fixed [get_property -quiet IS_LOC_FIXED $c]
        set where "unplaced"
        if {$loc ne ""} {
            set s [get_slrs -quiet -of_objects [get_sites -quiet $loc]]
            if {$s ne ""} { set where [get_property NAME $s] }
        }
        puts [format "NOC-STEER:   %-58s LOC=%-20s %-6s fixed=%s" \
                [get_property NAME $c] $loc $where $fixed]
        if {$where ne $target} { lappend to_move $c }
    }
    if {[llength $to_move] == 0} {
        puts "NOC-STEER: all vNOC ingress already in $target; nothing to do."
        return
    }

    # ---- candidate sites: target SLR *intersected with* the DFX partition ----
    #
    # Both halves matter. A first attempt used the SLR's sites directly and put
    # four NMUs at NOC_NMU512_*Y18, which is in SLR2 but OUTSIDE pblock_slash,
    # and opt_design refused them:
    #   [DRC HDPR-29] Reconfigurable logic illegally placed ... outside
    #   reconfigurable Pblock 'pblock_slash'
    # RM cells must land inside the reconfigurable region; pblock_slash's
    # NOC_NMU512 ranges stop at Y17.
    #
    # Note also that "free" cannot be tested by occupancy here: at pre-opt the
    # whole RM is unplaced, so every site looks free. Capacity is judged against
    # the intersection instead, and the sites this does NOT take stay available
    # for the RM's other NoC ports (the ddr_noc_* group).
    set slr_sites [get_sites -quiet -of_objects $slr -filter {SITE_TYPE =~ *NOC_NMU512*}]
    set pb [get_pblocks -quiet pblock_slash]
    if {[llength $pb] != 1} {
        puts "NOC-STEER: pblock_slash not found; refusing to place blind."
        if {$do_abort} { error "NOC-STEER: pblock_slash not found" }
        return
    }
    set pb_sites [get_sites -quiet -of_objects $pb -filter {SITE_TYPE =~ *NOC_NMU512*}]
    array set in_pb {}
    foreach s $pb_sites { set in_pb([get_property NAME $s]) 1 }

    set cand {}
    foreach s $slr_sites {
        if {[info exists in_pb([get_property NAME $s])]} { lappend cand $s }
    }
    puts "NOC-STEER: $target has [llength $slr_sites] NOC_NMU512 site(s),\
[llength $cand] of them inside pblock_slash"
    puts "NOC-STEER: need [llength $to_move]"
    if {[llength $cand] < [llength $to_move]} {
        puts "NOC-STEER: FAILED -- not enough in-partition NMU sites in $target."
        if {$do_abort} { error "NOC-STEER: insufficient in-partition NMU capacity in $target" }
        return
    }
    set free $cand

    # ---- try to relocate; the first failure is the answer we came for ----
    set moved 0
    foreach c $to_move {
        set site [lindex $free $moved]
        if {[catch {
            set_property LOC [get_property NAME $site] $c
        } msg]} {
            puts "NOC-STEER: FAILED to place [get_property NAME $c] at [get_property NAME $site]"
            puts "NOC-STEER:   $msg"
            puts "NOC-STEER: the NoC solution is locked at this point in the flow."
            if {$do_abort} {
                error "NOC-STEER: NoC placement is not malleable in the pre-opt hook"
            }
            return
        }
        puts "NOC-STEER:   moved [get_property NAME $c] -> [get_property NAME $site]"
        incr moved
    }
    puts "NOC-STEER: relocated $moved vNOC ingress unit(s) into $target"
}

noc_steer_run
