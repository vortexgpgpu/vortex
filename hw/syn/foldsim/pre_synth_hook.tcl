# Sourced by slashkit via --pre-synth-tcls, after the linker has created the
# project and its runs but before it launches them.

# The native flow's hook: the pre-opt utilization report and post-route
# phys_opt_design -directive AggressiveExplore.
source [file join [file dirname [file normalize [info script]]] .. xilinx aved pre_synth_hook.tcl]

# The kernel's out-of-context synthesis sees its clock (the IP's OOC XDC) and
# retimes against it: a threaded model's registered state is retimable.
foreach run [get_runs -quiet -filter {IS_SYNTHESIS} *foldsim_afu*] {
    set_property STEPS.SYNTH_DESIGN.ARGS.GLOBAL_RETIMING on $run
    puts "INFO: global retiming on for $run"
}
