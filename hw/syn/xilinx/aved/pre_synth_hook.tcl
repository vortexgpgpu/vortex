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
