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

# Post-route physical optimization. Required to close 300 MHz.
#
# With the static shell's vNOC memory ingress pinned to SLR2, the RM routes
# with a small, uniform residual -- ~265 endpoints averaging -13 ps, the worst
# being a replica-to-replica hop on a high-fanout register the placer had
# already tried to fix by replication. That profile is fanout-driven rather
# than logic depth, which is what post-route phys_opt targets:
#
#     without   WNS -0.034   297.0 MHz
#     with      WNS  0.000   300.0 MHz
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore \
    [get_runs impl_1]

# ---------------------------------------------------------------------------
# DO NOT add an implementation-strategy override here, and do not pblock the
# AFU. Both were tried against the shell's Congestion_SSI_SpreadLogic_high and
# both measured worse than leaving placement alone:
#
#     free placement                 WNS -0.260   942 failing
#     confine AFU to SLR1            WNS -0.801
#     confine AFU to SLR2            WNS -1.599
#     clear BalancedSLR strategy     WNS -0.343  4831 failing
#     confine SmartConnects to SLR2  WNS -0.326  3216 failing
#
# The SLR split those were fighting was never a placement decision. It came
# from the static shell leaving CONFIG.PHYSICAL_LOC empty on its eight
# hbm_vnoc_* NoC ingress units, so the NoC compiler scattered them while every
# HBM NMU sits in SLR2. An RM cannot move them -- DRC HDPR-122 allows each RM
# NoC port exactly one site. It is fixed shell-side in slash_base.tcl.
# ---------------------------------------------------------------------------
