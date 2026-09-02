# Sourced by slashkit via --pre-synth-tcls, after the linker has created the
# project and its runs but before it launches them. That is the only point in
# the linker flow where per-step implementation properties can be set, so it is
# where the pre-opt hook gets attached.

set hook_dir [file dirname [file normalize [info script]]]
set pre_opt  [file join $hook_dir pre_opt_hook.tcl]

if {![file exists $pre_opt]} {
    error "pre_opt_hook.tcl not found next to pre_synth_hook.tcl: $pre_opt"
}

# TOOL_DIR is read by the hook itself; fail here rather than mid-implementation.
if {![info exists ::env(TOOL_DIR)]} {
    error "TOOL_DIR is not set -- the pre-opt hook needs it to locate hw/scripts"
}

set_property STEPS.OPT_DESIGN.TCL.PRE $pre_opt [get_runs impl_1]
puts "INFO: attached pre-opt hook to impl_1: $pre_opt"

# ---------------------------------------------------------------------------
# DO NOT re-add an implementation-strategy override here. It was tried and
# measured HARMFUL.
#
# The theory was that the shell's `Congestion_SSI_SpreadLogic_high` (set by
# slash_project_build.tcl:190) was smearing a 3.5%-utilization design across
# SLRs, via `Floorplan.BalancedSLR.high` (balance logic across SLRs) and
# `NET_DELAY_WEIGHT low` (deprioritize net delay).
#
# Build `strat300` cleared both, set NET_DELAY_WEIGHT high, and added
# AggressiveExplore place/route/phys_opt plus post-route phys_opt:
#
#     fast300  (shell strategy as-is)   WNS -0.260   942 failing endpoints
#     strat300 (strategy overridden)    WNS -0.343  4831 failing endpoints
#
# Worse by 0.083 ns, and 5x the failing endpoints. More importantly it
# DISPROVED the theory: the SLR partition barely moved.
#
#     block             balancing ON      balancing OFF
#     vortex_afu_0      99.8% SLR1        99.8% SLR1
#     smartconnect_0    86.4% SLR2        89.3% SLR2
#     hbm_sc_01         86.7% SLR2        85.7% SLR2
#
# A placer explicitly told to prioritize net delay still left both
# SmartConnects on the far die from the AFU. They are held there by their NoC
# anchors, not by a strategy setting: every HBM channel on this device is in
# SLR2 (NOC_NMU_HBM2E sites -- SLR0: 0, SLR1: 0, SLR2: 64), while MEM_TAG's
# VNOC ports are in SLR1. The crossing is anchored, not chosen, and the real
# fix is connectivity-side. See docs/proposals/v80_slr_efficiency_plan.md.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# UPDATE (vnoc2): the premise above has changed, but the conclusion still holds
# for PLACEMENT strategy, so the block above stays in force.
#
# The connectivity-side fix it calls for was done: the static shell was rebuilt
# with the 8 hbm_vnoc_* NoC ingress units pinned to SLR2 NMU512 sites
# (slash_base.tcl CONFIG.PHYSICAL_LOC). MEM now anchors on the same die as HBM.
#
#     fast300 (split shell)     WNS -0.260   942 failing   278.3 MHz
#     vnoc2   (co-located)      WNS -0.034   265 failing   297.0 MHz
#
# Both SmartConnects left the critical path entirely. What remains is 265
# endpoints averaging -13 ps, all inside vortex_afu_0, concentrated in
# cp_core/g_cpe[0].u_fetch -- and the worst path is a REPLICA-TO-REPLICA hop
# on offset_r, i.e. a high-fanout artifact the placer already tried to fix by
# replication.
#
# That profile -- tiny, distributed, fanout-driven -- is what post-route
# physical optimization exists for. It is NOT a placement-strategy problem, so
# this deliberately does not touch the shell's placer settings; it only adds an
# optimization pass after routing. Off by default.
# ---------------------------------------------------------------------------
if {[info exists ::env(VX_POST_PHYSOPT)] && $::env(VX_POST_PHYSOPT) eq "1"} {
    set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
    set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore \
        [get_runs impl_1]
    puts "INFO: enabled post-route phys_opt (AggressiveExplore) on impl_1"
} else {
    puts "INFO: post-route phys_opt not enabled (set VX_POST_PHYSOPT=1)"
}
