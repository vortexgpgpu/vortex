# Sourced by slashkit via --pre-synth-tcls, after the linker has built the
# project and its block design but before it launches any run.

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
# retimes against it: a threaded model's registered state is retimable. The
# block design's IP runs do not exist yet here (launching the reconfigurable
# module's synthesis would create them), so the hook generates the design's
# targets and creates them first; launch_runs then takes the runs as they are.
# No run found is an error: the image would silently synthesize without it.
set fs_bd [get_files -quiet -all slash_base.bd]
if {[llength $fs_bd] != 1} {
    error "foldsim pre_synth_hook: expected one slash_base.bd, found [llength $fs_bd]"
}
generate_target all $fs_bd
create_ip_run $fs_bd
set fs_runs [get_runs -quiet -filter {IS_SYNTHESIS} *foldsim_afu*]
if {[llength $fs_runs] == 0} {
    error "foldsim pre_synth_hook: no out-of-context synthesis run of the foldsim kernel"
}
foreach run $fs_runs {
    set_property STEPS.SYNTH_DESIGN.ARGS.GLOBAL_RETIMING on $run
    puts "INFO: global retiming on for $run"
}
