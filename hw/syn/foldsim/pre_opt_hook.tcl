# Runs on impl_1 immediately before opt_design, attached by pre_synth_hook.tcl
# when the image has a floorplan: the native flow's pre-opt hook, then the
# floorplan.

set fs_hook_dir [file dirname [file normalize [info script]]]
source [file join $fs_hook_dir .. xilinx aved pre_opt_hook.tcl]
# The variable reaches the run from the linker's environment; a run without
# it would silently place without the floorplan the image was built for.
if {![info exists ::env(FOLDSIM_FLOORPLAN)] || $::env(FOLDSIM_FLOORPLAN) eq ""} {
    error "FOLDSIM_FLOORPLAN is not set in the implementation run"
}
source [file join $fs_hook_dir floorplan.tcl]
