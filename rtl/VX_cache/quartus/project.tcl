load_package flow
package require cmdline

set options { \
    { "project.arg" "" "Project name" } \
    { "family.arg" "" "Device family name" } \
    { "device.arg" "" "Device name" } \
    { "top.arg" "" "Top level module" } \
    { "sdc.arg" "" "Timing Design Constraints file" } \
    { "src.arg" "" "Verilog source file" } \
    { "inc.arg" "." "Include path" } \
}

array set opts [::cmdline::getoptions quartus(args) $options]

project_new $opts(project) -overwrite

set_global_assignment -name FAMILY $opts(family)
set_global_assignment -name DEVICE $opts(device)
set_global_assignment -name TOP_LEVEL_ENTITY $opts(top)
set_global_assignment -name VERILOG_FILE $opts(src)
set_global_assignment -name SEARCH_PATH $opts(inc)
set_global_assignment -name SDC_FILE $opts(sdc)
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY bin
set_global_assignment -name NUM_PARALLEL_PROCESSORS ALL
set_global_assignment -name VERILOG_INPUT_VERSION SYSTEMVERILOG_2009

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