load_package flow
package require cmdline

set options { 
    { "project.arg" "" "Project name" } 
    { "family.arg" "" "Device family name" } 
    { "device.arg" "" "Device name" } 
    { "top.arg" "" "Top level module" }     
    { "src.arg" "" "Verilog source file" } 
    { "inc.arg" "" "Include path (optional)" } 
    { "sdc.arg" "" "Timing Design Constraints file (optional)" } 
    { "set.arg" "" "Macro value (optional)" } 
}

set q_args_orig $quartus(args)

array set opts [::cmdline::getoptions quartus(args) $options]

# Verify required parameters
set requiredParameters {project family device top src}
foreach p $requiredParameters {
    if {$opts($p) == ""} {
        puts stderr "Missing required parameter: -$p"
        exit 1
    }
}

project_new $opts(project) -overwrite

set_global_assignment -name FAMILY $opts(family)
set_global_assignment -name DEVICE $opts(device)
set_global_assignment -name TOP_LEVEL_ENTITY $opts(top)
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY bin

set_global_assignment -name NUM_PARALLEL_PROCESSORS ALL
set_global_assignment -name VERILOG_INPUT_VERSION SYSTEMVERILOG_2009
set_global_assignment -name ADD_PASS_THROUGH_LOGIC_TO_INFERRED_RAMS ON
set_global_assignment -name VERILOG_MACRO QUARTUS
set_global_assignment -name VERILOG_MACRO SYNTHESIS
set_global_assignment -name VERILOG_MACRO NDEBUG
set_global_assignment -name MESSAGE_DISABLE 16818
set_global_assignment -name TIMEQUEST_DO_REPORT_TIMING ON

#set_global_assignment -name OPTIMIZATION_TECHNIQUE SPEED
#set_global_assignment -name OPTIMIZATION_MODE "AGGRESSIVE PERFORMANCE"
#set_global_assignment -name FINAL_PLACEMENT_OPTIMIZATION ALWAYS
#set_global_assignment -name PLACEMENT_EFFORT_MULTIPLIER 2.0
#set_global_assignment -name FITTER_EFFORT "STANDARD FIT"
#set_global_assignment -name OPTIMIZE_HOLD_TIMING "ALL PATHS"
#set_global_assignment -name OPTIMIZE_MULTI_CORNER_TIMING ON
#set_global_assignment -name ROUTER_TIMING_OPTIMIZATION_LEVEL MAXIMUM
#set_global_assignment -name ROUTER_CLOCKING_TOPOLOGY_ANALYSIS ON
#set_global_assignment -name ROUTER_LCELL_INSERTION_AND_LOGIC_DUPLICATION ON
#set_global_assignment -name SYNTH_TIMING_DRIVEN_SYNTHESIS ON
#set_global_assignment -name TIMEQUEST_MULTICORNER_ANALYSIS ON
#set_global_assignment -name MIN_CORE_JUNCTION_TEMP 0
#set_global_assignment -name MAX_CORE_JUNCTION_TEMP 100
#set_global_assignment -name SEED 1

switch $opts(family) {
    "Arria 10" {
        set_global_assignment -name VERILOG_MACRO ALTERA_A10    
    }
    "Stratix 10" {
        set_global_assignment -name VERILOG_MACRO ALTERA_S10    
    }
    default {
        puts stderr "Invalid device family"
        exit 1
    }
}

set idx 0
foreach arg $q_args_orig {
    incr idx
    if [string match "-src" $arg] {
        set_global_assignment -name VERILOG_FILE [lindex $q_args_orig $idx]
    }
    if [string match "-inc" $arg] {
        set_global_assignment -name SEARCH_PATH [lindex $q_args_orig $idx]
    }
    if [string match "-sdc" $arg] {
        set_global_assignment -name SDC_FILE [lindex $q_args_orig $idx]
    }
    if [string match "-set" $arg] {
        set_global_assignment -name VERILOG_MACRO [lindex $q_args_orig $idx]
    }
}

proc make_all_pins_virtual {} {
    execute_module -tool map
    set name_ids [get_names -filter * -node_type pin]
    foreach_in_collection name_id $name_ids {        
        set pin_name [get_name_info -info full_path $name_id]
        post_message "Making VIRTUAL_PIN assignment to $pin_name"
        set_instance_assignment -to $pin_name -name VIRTUAL_PIN ON
    }
    export_assignments
}

make_all_pins_virtual

project_close