######################################################################
# Automatically inserts ILA instances in a batch flow, and calls "implement_debug_core".   Can also be used in a GUI flow
# This should ONLY be invoked after synthesis, and before opt_design.   If opt_design is called first, marked nets may be missing and not found
# Warning: Currently will skip a net if it has no obvious clock domain on the driver.  Nets connected to input buffers will be dropped unless "mark_debug_clock" is attached to the net.
# Nets attached to VIO cores have the "mark_debug" attribute, and will be filtered out unless the "mark_debug_valid" attribute is attached.
# Supports the following additional attributes beyond "mark_debug"
# attribute mark_debug_valid of X : signal is "true";   -- Marks a net for ILA capture, even if net is also attached to a VIO core
# attribute mark_debug_clock of X : signal is "inst1_bufg/clock";  -- Specifies clock net to use for capturing this net.  May create a new ILA core for that clock domain
# attribute mark_debug_depth of X : signal is "4096";              -- overrides default depth for this ILA core. valid values: 1024, 2048, ... 132072.   Last attribute that is scanned will win.
# attribute mark_debug_adv_trigger of X : signal is "true";        -- specifies that advanced trigger capability will be added to ILA core
# Engineer:  J. McCluskey
proc insert_ila { depth } {
    # sequence through debug nets and organize them by clock in the
    # clock_list array. Also create max and min array for bus indices
    set dbgs [get_nets -hierarchical -filter {MARK_DEBUG}]
    if {[llength $dbgs] == 0} {
        puts "No debug net found. No ILA cores created"
        return
    }

    # process list of nets to find and reject nets that are attached to VIO cores.
    # This has a side effect that VIO nets can't be monitored with an ILA
    # This can be overridden by using the attribute "mark_debug_valid" = "true" on a net like this.
    set net_list {}
    foreach net $dbgs {
        if { [get_property -quiet MARK_DEBUG_VALID $net] != "true" } {
            set pin_list [get_pins -of_objects [get_nets -segments $net]]
            set not_vio_net 1
            foreach pin $pin_list {
                if { [get_property IS_DEBUG_CORE [get_cells -of_object $pin]] == 1 } {
                    # It seems this net is attached to a debug core (i.e. VIO core) already, so we should skip adding it to the netlist
                    set not_vio_net 0
                    break
                }
            }
            if { $not_vio_net == 1 } { lappend net_list $net; }
        } else {
            lappend net_list $net
        }
    }

    # check again to see if we have any nets left now
    if {[llength $net_list] == 0} {
        puts "All nets with MARK_DEBUG are already connected to VIO cores. No ILA cores created"
        return
    }

    # Now that the netlist has been filtered, determine bus names and clock domains
    foreach d $net_list {
        # name is root name of a bus, index is the bit index in the bus
        set name [regsub {\[[[:digit:]]+\]$} $d {}]
        set index [regsub {^.*\[([[:digit:]]+)\]$} $d {\1}]
        if {[string is integer -strict $index]} {
            if {![info exists max($name)]} {
                set max($name) $index
                set min($name) $index
            } elseif {$index > $max($name)} {
                set max($name) $index
            } elseif {$index < $min($name)} {
                set min($name) $index
            }
        } else {
            set max($name) -1
        }
        # Now we search for the local clock net associated with the target net.
        # There may be ambiguities or no answer in some cases
        if {![info exists clocks($name)]} {
            # does MARK_DEBUG_CLOCK decorate this net? If not, then search backwards to the driver cell
            set clk_name [get_property -quiet MARK_DEBUG_CLOCK $d]
            if {  [llength $clk_name] == 0 } {
                # trace to the clock net, tracing backwards via the driver pin.
                set driver_pin [get_pins -filter {DIRECTION == "OUT" && IS_LEAF == TRUE } -of_objects [ get_nets -segments $d ]]
                set driver_cell [get_cells -of_objects $driver_pin]
                if { [get_property IS_SEQUENTIAL $driver_cell] == 1 } {
                    set timing_arc [get_timing_arcs -to $driver_pin]
                    set cell_clock_pin [get_pins -filter {IS_CLOCK} [get_property FROM_PIN $timing_arc]]
                    if { [llength $cell_clock_pin] > 1 } {
                        puts "Error: in insert_ila. Found more than 1 clock pin in driver cell $driver_cell with timing arc $timing_arc for net $d"
                        continue
                    }
                } else {
                    # our driver cell is a LUT or LUTMEM in combinatorial mode, we need to trace further.
                    set paths [get_timing_paths -quiet -through $driver_pin ]
                    if { [llength $paths] > 0 } {
                        # note that here we arbitrarily select the start point of the FIRST timing path... there might be multiple clocks with timing paths for this net.
                        # use MARK_DEBUG_CLOCK to specify another clock in this case.
                        set cell_clock_pin [get_pins [get_property STARTPOINT_PIN [lindex $paths 0]]]
                    } else {
                        # Can't find any timing path, so skip the net, and warn the user.
                        puts "Critical Warning: from insert_ila.tcl Can't trace any clock domain on driver of net $d"
                        puts "Please attach the attribute MARK_DEBUG_CLOCK with a string containing the net name of the desired sampling clock, .i.e."
                        puts "attribute mark_debug_clock of $d : signal is \"inst_bufg/clk\";"
                        continue
                    }
                }
                # clk_net will usually be a list of net segments, which needs filtering to determine the net connected to the driver pin
                set clk_net [get_nets -segments -of_objects $cell_clock_pin]
            } else {
                set clk_net [get_nets -segments $clk_name]
                if { [llength $clk_net] == 0 } { puts "MARK_DEBUG_CLOCK attribute on net $d does not match any known net. Please fix."; continue; }
            }
            # trace forward to net actually connected to clock buffer output, not any of the lower level segment names
            set clocks($name) [get_nets -of_objects [get_pins -filter {DIRECTION == "OUT" && IS_LEAF == TRUE } -of_objects $clk_net]]
            if { [llength $clocks($name)] == 0 } {
                puts "Critical Warning: from insert_ila.tcl Can't trace any clock domain on driver of net $d"
                puts "Please attach the attribute MARK_DEBUG_CLOCK with a string containing the net name of the desired sampling clock, .i.e."
                puts "attribute mark_debug_clock of $d : signal is \"inst_bufg/clk\";"
                continue
            }
            if {![info exists clock_list($clocks($name))]} {
              # found a new clock
              puts "New clock found is $clocks($name)"
              set clock_list($clocks($name)) [list $name]
              set ila_depth($clocks($name)) $depth
              set ila_adv_trigger($clocks($name)) false
            } else {
              lappend clock_list($clocks($name)) $name
            }
            # Does this net have a "MARK_DEBUG_DEPTH" attribute attached?
            set clk_depth [get_property -quiet MARK_DEBUG_DEPTH $d]
            if { [llength $clk_depth] != 0 } {
                set ila_depth($clocks($name)) $clk_depth
            }
            # Does this net have a "MARK_DEBUG_ADV_TRIGGER" attribute attached?
            set trigger [get_property -quiet MARK_DEBUG_ADV_TRIGGER $d]
            if { $trigger == "true" } {
                set ila_adv_trigger($clocks($name)) true
            }
        }
    }

    set ila_count 0
    set trig_out ""
    set trig_out_ack ""

    if { [llength [array names clock_list]] > 1 } {
        set enable_trigger true
    } else {
        set enable_trigger false
    }

    foreach c [array names clock_list] {
        # Now build and connect an ILA core for each clock domain
        [incr ila_count ]
        set ila_inst "ila_$ila_count"
        # first verify if depth is a member of the set, 1024, 2048, 4096, 8192, ... 131072
        if { $ila_depth($c) < 1024 || [expr $ila_depth($c) & ($ila_depth($c) - 1)] || $ila_depth($c) > 131072 } {
            # Depth is not right...  lets fix it, and continue
            if { $ila_depth($c) < 1024 } {
                set new_depth 1024
            } elseif { $ila_depth($c) > 131072 } {
                set new_depth 131072
            } else {
                # round value to next highest power of 2, (in log space)
                set new_depth [expr 1 << int( log($ila_depth($c))/log(2) + .9999 )]
            }
            puts "Can't create ILA core $ila_inst with depth of $ila_depth($c)! Changed capture depth to $new_depth"
            set ila_depth($c) $new_depth
        }
        # create ILA and connect its clock
        puts "Creating ILA $ila_inst with clock $c, capture depth $ila_depth($c) and advanced trigger = $ila_adv_trigger($c)"
        create_debug_core $ila_inst ila
        if { $ila_adv_trigger($c) } { set mu_cnt 4; } else { set mu_cnt 2; }
        set_property    C_DATA_DEPTH   $ila_depth($c) [get_debug_cores $ila_inst]
        set_property    C_TRIGIN_EN    $enable_trigger [get_debug_cores $ila_inst]
        set_property    C_TRIGOUT_EN   $enable_trigger [get_debug_cores $ila_inst]
        set_property    C_ADV_TRIGGER  $ila_adv_trigger($c) [get_debug_cores $ila_inst]
        set_property    C_INPUT_PIPE_STAGES 1 [get_debug_cores $ila_inst]
        set_property    C_EN_STRG_QUAL true [get_debug_cores $ila_inst]
        set_property    ALL_PROBE_SAME_MU true [get_debug_cores $ila_inst]
        set_property    ALL_PROBE_SAME_MU_CNT $mu_cnt [get_debug_cores $ila_inst]
        set_property    port_width 1 [get_debug_ports $ila_inst/clk]
        connect_debug_port $ila_inst/clk $c
        # hookup trigger ports in a circle if more than one ILA is created
        if { $enable_trigger == true } {
            create_debug_port $ila_inst trig_in
            create_debug_port $ila_inst trig_in_ack
            create_debug_port $ila_inst trig_out
            create_debug_port $ila_inst trig_out_ack
            if { $trig_out != "" } {
                connect_debug_port $ila_inst/trig_in [get_nets $trig_out]
            }
            if { $trig_out_ack != "" } {
                connect_debug_port $ila_inst/trig_in_ack [get_nets $trig_out_ack]
            }
            set trig_out ${ila_inst}_trig_out_$ila_count
            create_net $trig_out
            connect_debug_port  $ila_inst/trig_out [get_nets $trig_out]
            set trig_out_ack ${ila_inst}_trig_out_ack_$ila_count
            create_net $trig_out_ack
            connect_debug_port  $ila_inst/trig_out_ack [get_nets $trig_out_ack]
        }
        # add probes
        set nprobes 0
        foreach n [lsort $clock_list($c)] {
            set nets {}
            if {$max($n) < 0} {
                lappend nets [get_nets $n]
            } else {
                # n is a bus name
                for {set i $min($n)} {$i <= $max($n)} {incr i} {
                    lappend nets [get_nets $n[$i]]
                }
            }
            set prb probe$nprobes
            if {$nprobes > 0} {
                create_debug_port $ila_inst probe
            }
            set_property port_width [llength $nets] [get_debug_ports $ila_inst/$prb]
            connect_debug_port $ila_inst/$prb $nets
            incr nprobes
        }
    }

    # at this point, we need to complete the circular connection of trigger outputs and acks
    if { $enable_trigger == true } {
        connect_debug_port ila_1/trig_in [get_nets $trig_out]
        connect_debug_port ila_1/trig_in_ack [get_nets $trig_out_ack]
    }
    set project_found [get_projects -quiet]
    if { $project_found != "New Project" } {
        puts "Saving constraints now in project [current_project -quiet]"
        save_constraints_as debug_constraints.xdc
    }

    # run ILA cores implementation
    implement_debug_core

    # write out probe info file
    write_debug_probes -force debug_nets.ltx
}