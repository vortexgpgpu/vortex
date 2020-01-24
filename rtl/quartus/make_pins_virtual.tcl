load_package flow

package require cmdline

proc make_all_pins_virtual { args } {

    remove_all_instance_assignments -name VIRTUAL_PIN
    execute_module -tool map
    set name_ids [get_names -filter * -node_type pin]

    foreach_in_collection name_id $name_ids {
        set pin_name [get_name_info -info full_path $name_id]

        if { -1 == [lsearch -exact { clk, reset } $pin_name] } {
            post_message "Making VIRTUAL_PIN assignment to $pin_name"
            set_instance_assignment -to $pin_name -name VIRTUAL_PIN ON
        } else {
            post_message "Skipping VIRTUAL_PIN assignment to $pin_name"
        }
    }
    export_assignments
}


make_all_pins_virtual


