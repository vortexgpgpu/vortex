# Sourced by slashkit via --pre-synth-tcls, after the linker has created the
# project and its runs but before it launches them.

# The native flow's hook: the pre-opt utilization report and post-route
# phys_opt_design -directive AggressiveExplore.
set fs_hook_dir [file dirname [file normalize [info script]]]
source [file join $fs_hook_dir .. xilinx aved pre_synth_hook.tcl]

# An image with a floorplan (foldsim --slr-groups) applies it before
# opt_design, after the native pre-opt hook.
if {[info exists ::env(FOLDSIM_FLOORPLAN)] && $::env(FOLDSIM_FLOORPLAN) ne ""} {
    set_property STEPS.OPT_DESIGN.TCL.PRE [file join $fs_hook_dir pre_opt_hook.tcl] [get_runs impl_1]
    puts "INFO: foldsim floorplan $::env(FOLDSIM_FLOORPLAN) attached to impl_1"
}

# The kernel's out-of-context synthesis sees its clock (the IP's OOC XDC) and
# retimes against it: a threaded model's registered state is retimable.
foreach run [get_runs -quiet -filter {IS_SYNTHESIS} *foldsim_afu*] {
    set_property STEPS.SYNTH_DESIGN.ARGS.GLOBAL_RETIMING on $run
    puts "INFO: global retiming on for $run"
}
