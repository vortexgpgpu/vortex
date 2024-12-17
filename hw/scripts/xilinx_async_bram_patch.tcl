# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

namespace eval vortex {

variable info 0
variable debug 0

proc print_error {msg {do_exit 1}} {
  if {$do_exit} {
    puts "ERROR: $msg"
    exit -1
  } else {
    puts "WARNING: $msg"
  }
}

proc str_replace {str match repl} {
  set result ""
  regsub $match $str $repl result
  return $result
}

proc regex_escape {str} {
  return [string map {
    \\ \\\\
    ^ \\^
    . \\.
    \[ \\\[
    \] \\\]
    \$ \\\$
    \( \\\(
    \) \\\)
    | \\|
    * \\*
    + \\+
    ? \\?
    \{ \\\{
    \} \\\}
  } $str]
}

proc unique_cell_name {name} {
  if {[get_cells -quiet $name] == {}} { return $name }
  set index 0
  while {[get_cells -quiet ${name}_${index}] != {}} { incr index }
  return ${name}_${index}
}

proc unique_net_name {name} {
  if {[get_nets -quiet $name] == {}} { return $name }
  set index 0
  while {[get_nets -quiet ${name}_${index}] != {}} { incr index }
  return ${name}_${index}
}

proc build_parent_child_map {all_cells} {
  set parent_child_map {}
  foreach cell $all_cells {
    set parent [get_property PARENT $cell]
    if {$parent ne ""} {
      if {[dict exists $parent_child_map $parent]} {
        dict lappend parent_child_map $parent $cell
      } else {
        dict set parent_child_map $parent [list $cell]
      }
    }
  }
  return $parent_child_map
}

proc find_cell_descendants_recursive {parent_cell parent_child_map} {
  set descendants {}
  if {[dict exists $parent_child_map $parent_cell]} {
    set children [dict get $parent_child_map $parent_cell]
    foreach child $children {
      # Add the child to the list
      lappend descendants $child
      # Recursively add its descendants
      set sub_descendants [find_cell_descendants_recursive $child $parent_child_map]
      lappend descendants {*}$sub_descendants
    }
  }
  return $descendants
}

proc find_cell_descendants {parent_cell} {
  set all_cells [get_cells -hierarchical]
  set parent_child_map [build_parent_child_map $all_cells]
  return [find_cell_descendants_recursive $parent_cell $parent_child_map]
}

proc find_nested_cells {parent_cell name_match {should_exist 1}} {
  set hier_sep [get_hierarchy_separator]
  set matching_cells {}
  foreach cell [find_cell_descendants $parent_cell] {
    set parent_name [get_property PARENT $cell]
    set cell_name [get_property NAME $cell]
    set name_prefix [regex_escape "${parent_name}${hier_sep}"]
    set pattern "${name_prefix}${name_match}"
    if {[regexp $pattern $cell_name]} {
      lappend matching_cells $cell
    }
  }
  if {[llength $matching_cells] == 0} {
    print_error "No matching cell found for '$parent_cell' matching '$name_match'." $should_exist
  }
  return $matching_cells
}

proc find_cell_nets {cell name_match {should_exist 1}} {
  set matching_nets {}
  foreach net [get_nets -hierarchical -filter "PARENT_CELL == $cell"] {
    set name [get_property NAME $net]
    if {[regexp $name_match $name]} {
      lappend matching_nets $net
    }
  }
  if {[llength $matching_nets] == 0} {
    print_error "No matching net found for '$cell' matching '$name_match'." $should_exist
  }
  return $matching_nets
}

proc get_cell_net {cell name} {
  set net [get_nets -hierarchical -filter "PARENT_CELL == $cell && NAME == $name"]
  if {[llength $net] == 0} {
    puts "ERROR: No matching net found for '$cell' matching '$name'."
    exit -1
  }
  return $net;
}

proc find_cell_pins {cell name_match {should_exist 1}} {
  set hier_sep [get_hierarchy_separator]
  set matching_pins {}
  foreach pin [get_pins -of_objects $cell] {
    set name [get_property NAME $pin]
    set name_prefix [regex_escape "${cell}${hier_sep}"]
    set pattern "${name_prefix}${name_match}"
    if {[regexp $pattern $name]} {
      lappend matching_pins $pin
    }
  }
  if {[llength $matching_pins] == 0} {
    print_error "No matching pin found for '$cell' matching '$name_match'." $should_exist
  }
  return $matching_pins
}

proc get_cell_pin {cell name} {
  set pin [get_pins -of_objects $cell -filter "NAME == $name"]
  if {[llength $pin] == 0} {
    puts "ERROR: No matching pin found for '$cell' matching '$name'."
    exit -1
  }
  return $pin
}

proc remove_cell_from_netlist {cell} {
  variable info

  # Disconnect all pins of the cell
  foreach pin [get_pins -quiet -of_objects $cell] {
    foreach net [get_nets -quiet -of_objects $pin] {
      disconnect_net -net $net -objects $pin
      if {$info} {puts "INFO: Disconnected net '$net' from pin '$pin'."}
    }
  }

  # Remove the cell
  remove_cell $cell
  if {$info} {puts "INFO: Cell '$cell' was removed successfully."}
}

proc replace_pin_source {pin source_pin} {
  variable debug

  # Disconnect existing net from pin
  set net [get_nets -of_objects $pin]
  if {[llength $net] == 1} {
    disconnect_net -net $net -objects $pin
    if {$debug} {puts "DEBUG: Disconnected net '$net' from pin '$pin'."}
  } elseif {[llength $net] > 1} {
    puts "ERROR: Multiple nets connected to pin '$pin'."
    exit -1
  } else {
    puts "WARNING: No net connected to pin '$pin'."
  }

  set source_net [get_nets -quiet -of_objects $source_pin]
  if {[llength $source_net] == 0} {
    # Create a new net if none exists
    set source_cell [get_cells -of_objects $source_pin]
    set net_name [unique_net_name "${source_cell}_net"]
    set source_net [create_net $net_name]
    if {$debug} {puts "DEBUG: Created source_net: '$source_net'"}
    # Connect the source pin to the new net
    connect_net -net $source_net -objects $source_pin -hierarchical
    if {$debug} {puts "DEBUG: Connected net '$source_net' to pin '$source_pin'."}
  } elseif {[llength $source_net] > 1} {
    puts "ERROR: Multiple nets connected to pin '$source_pin'."
    exit -1
  }

  # Connect pin to the new source net
  connect_net -net $source_net -objects $pin -hierarchical
  if {$debug} {puts "DEBUG: Connected net '$source_net' to pin '$pin'."}
}

proc find_net_driver {input_net {should_exist 1}} {
  set driverPins [get_pins -quiet -leaf -of_objects $input_net -filter {DIRECTION == "OUT"}]
  if {[llength $driverPins] == 0} {
    set driverPorts [get_ports -quiet -of_objects $input_net -filter {DIRECTION == "IN"}]
    if {[llength $driverPorts] == 0} {
      print_error "No driver found for '$input_net'." $should_exist
    } elseif {[llength $driverPorts] > 1} {
      puts "WARNING: Multiple driver ports found for '$input_net'."
      return [lindex $driverPorts 0]
    }
    return $driverPorts
  } elseif {[llength $driverPins] > 1} {
    puts "WARNING: Multiple driver pins found for '$input_net'."
    return [lindex $driverPins 0]
  }
  return $driverPins
}

proc find_pin_driver {input_pin {should_exist 1}} {
  set net [get_nets -quiet -of_objects $input_pin]
  if {[llength $net] == 0} {
    print_error "No net connected to pin '$input_pin'." $should_exist
    return ""
  } elseif {[llength $net] > 1} {
    puts "ERROR: Multiple nets connected to pin '$input_pin'."
    exit -1
  }
  return [find_net_driver $net]
}

proc create_register_next {parent reg_cell} {
  variable info
  variable debug

  set hier_sep [get_hierarchy_separator]

  set reg_d_pin [get_pins "${reg_cell}${hier_sep}D"]
  if {[llength $reg_d_pin] == 0} {
    puts "ERROR: No D pin found on register cell '$reg_cell'."
    exit -1
  } elseif {[llength $reg_d_pin] > 1} {
    puts "ERROR: Multiple D pins found on register cell '$reg_cell'."
    exit -1
  }

  if {$debug} {puts "DEBUG: reg_d_pin: '$reg_d_pin'"}

  set reg_d_src_pin [find_pin_driver $reg_d_pin]
  if {$reg_d_src_pin == ""} {
    puts "ERROR: No source pin found connected to '$reg_d_pin'."
    exit -1
  }

  if {$debug} {puts "DEBUG: reg_d_src_pin: '$reg_d_src_pin'"}

  set reg_r_src_pin ""

  set register_type [get_property REF_NAME $reg_cell]
  if {$register_type == "FDRE"} {
    set reg_r_pin [get_pins "${reg_cell}${hier_sep}R"]
    if {[llength $reg_r_pin] == 0} {
      puts "ERROR: No R pin found on FDRE cell '$reg_cell'."
      exit -1
    } elseif {[llength $reg_r_pin] > 1} {
      puts "ERROR: Multiple R pins found on FDRE cell '$reg_cell'."
      exit -1
    }

    if {$debug} {puts "DEBUG: reg_r_pin: '$reg_r_pin'"}

    set reg_r_src_pin [find_pin_driver $reg_r_pin]
    if {$reg_r_src_pin == ""} {
      puts "ERROR: No source pin found connected to '$reg_r_pin'."
      exit -1
    }
  } elseif {$register_type == "FDSE"} {
    set reg_s_pin [get_pins "${reg_cell}${hier_sep}S"]
    if {[llength $reg_s_pin] == 0} {
      puts "ERROR: No S pin found on FDSE cell '$reg_cell'."
      exit -1
    } elseif {[llength $reg_s_pin] > 1} {
      puts "ERROR: Multiple S pins found on FDSE cell '$reg_cell'."
      exit -1
    }

    if {$debug} {puts "DEBUG: reg_s_pin: '$reg_s_pin'"}

    set reg_r_src_pin [find_pin_driver $reg_s_pin]
    if {$reg_r_src_pin == ""} {
      puts "ERROR: No source pin found connected to '$reg_s_pin'."
      exit -1
    }
  } else {
    puts "ERROR: Unsupported register type: '$register_type'."
    exit 1
  }

  if {$debug} {puts "DEBUG: reg_r_src_pin: '$reg_r_src_pin'"}

  set reg_d_src_net [get_nets -of_objects $reg_d_src_pin]
  if {[llength $reg_d_src_net] == 0} {
    puts "ERROR: Unable to get source nets for pins."
    exit -1
  } elseif {[llength $reg_d_src_net] > 1} {
    puts "ERROR: Multiple source nets found for pins."
    exit -1
  }

  set reg_r_src_net [get_nets -of_objects $reg_r_src_pin]
  if {[llength $reg_r_src_net] == 0} {
    puts "ERROR: Unable to get source nets for pins."
    exit -1
  } elseif {[llength $reg_r_src_net] > 1} {
    puts "ERROR: Multiple source nets found for pins."
    exit -1
  }

  # Create a MUX cell to implement register next value
  # Use a 2x1 LUT to describe the logic:
  # FDRE: O = I1 ? 0 : I0; where I0=D, I1=R
  # FDSE: O = I1 ? 1 : I0; where I0=D, I1=S
  set lut_name [unique_cell_name "${parent}${hier_sep}raddr_next"]
  set lut_cell [create_cell -reference LUT2 $lut_name]
  if {$info} {puts "INFO: Created lut cell: '$lut_cell'"}

  if {$register_type == "FDRE"} {
    set_property INIT 4'b0010 $lut_cell
  } elseif {$register_type == "FDSE"} {
    set_property INIT 4'b1110 $lut_cell
  } else {
    puts "ERROR: Unsupported register type: '$register_type'."
    exit 1
  }

  set lut_i0_pin [get_pins "${lut_cell}${hier_sep}I0"]
  if {[llength $lut_i0_pin] == 0} {
    puts "ERROR: No I0 pin found on FDSE cell '$lut_cell'."
    exit -1
  } elseif {[llength $lut_i0_pin] > 1} {
    puts "ERROR: Multiple I0 pins found on FDSE cell '$lut_cell'."
    exit -1
  }

  set lut_i1_pin [get_pins "${lut_cell}${hier_sep}I1"]
  if {[llength $lut_i1_pin] == 0} {
    puts "ERROR: No I1 pin found on FDSE cell '$lut_cell'."
    exit -1
  } elseif {[llength $lut_i1_pin] > 1} {
    puts "ERROR: Multiple I1 pins found on FDSE cell '$lut_cell'."
    exit -1
  }

  set lut_o_pin [get_pins "${lut_cell}${hier_sep}O"]
  if {[llength $lut_o_pin] == 0} {
    puts "ERROR: No O pin found on FDSE cell '$lut_cell'."
    exit -1
  } elseif {[llength $lut_o_pin] > 1} {
    puts "ERROR: Multiple O pins found on FDSE cell '$lut_cell'."
    exit -1
  }

  connect_net -net $reg_d_src_net -objects $lut_i0_pin -hierarchical
  if {$debug} {puts "DEBUG: Connected net '$reg_d_src_net' to pin '$lut_i0_pin'."}

  connect_net -net $reg_r_src_net -objects $lut_i1_pin -hierarchical
  if {$debug} {puts "DEBUG: Connected net '$reg_r_src_net' to pin '$lut_i1_pin'."}

  return $lut_o_pin
}

proc getOrCreateVCCPin {parent} {
  variable info
  variable debug

  set hier_sep [get_hierarchy_separator]
  set cell_name "${parent}${hier_sep}VCC"

  set vcc_cell [get_cells -quiet $cell_name]
  if {[llength $vcc_cell] == 0} {
    set vcc_cell [create_cell -reference VCC $cell_name]
    if {$info} {puts "INFO: Created VCC cell: '$vcc_cell'"}
  } elseif {[llength $vcc_cell] > 1} {
    puts "ERROR: Multiple VCC cells found with name '$cell_name'."
    exit -1
  }

  set vcc_pin [get_pins "${vcc_cell}${hier_sep}P"]
  if {[llength $vcc_pin] == 0} {
    puts "ERROR: No VCC pin found on VCC cell '$vcc_cell'."
    exit -1
  } elseif {[llength $vcc_pin] > 1} {
    puts "ERROR: Multiple VCC pins found on VCC cell '$vcc_cell'."
    exit -1
  }

  return $vcc_pin
}

proc getOrCreateGNDPin {parent} {
  variable info
  variable debug

  set hier_sep [get_hierarchy_separator]
  set cell_name "${parent}${hier_sep}GND"

  set gnd_cell [get_cells -quiet $cell_name]
  if {[llength $gnd_cell] == 0} {
    set gnd_cell [create_cell -reference GND $cell_name]
    if {$info} {puts "INFO: Created GND cell: '$gnd_cell'"}
  } elseif {[llength $gnd_cell] > 1} {
    puts "ERROR: Multiple GND cells found with name '$cell_name'."
    exit -1
  }

  set gnd_pin [get_pins "${gnd_cell}${hier_sep}G"]
  if {[llength $gnd_pin] == 0} {
    puts "ERROR: No GND pin found on GND cell '$gnd_cell'."
    exit -1
  } elseif {[llength $gnd_pin] > 1} {
    puts "ERROR: Multiple GND pins found on GND cell '$gnd_cell'."
    exit -1
  }

  return $gnd_pin
}

proc find_net_sinks {input_net {should_exist 1}} {
  set sink_pins {}
  foreach pin [get_pins -quiet -leaf -of_objects $input_net -filter {DIRECTION == "IN"}] {
    lappend sink_pins $pin
  }
  foreach port [get_ports -quiet -of_objects $input_net -filter {DIRECTION == "OUT"}] {
    lappend sink_pins $port
  }
  if {[llength $sink_pins] == 0} {
    print_error "No sink found for '$input_net'." $should_exist
  }
  return $sink_pins
}

proc find_matching_nets {cell nets match repl} {
  set matching_nets {}
  foreach net $nets {
    set net_name [str_replace $net $match $repl]
    set matching_net [get_cell_net $cell $net_name]
    if {$matching_net != ""} {
      lappend matching_nets $matching_net
    }
  }
  if {[llength $matching_nets] == 0} {
    puts "ERROR: No matching nets found for '$nets'."
    exit -1
  } elseif {[llength $matching_nets] != [llength $nets]} {
    puts "ERROR: Mismatch in number of matching nets."
    exit -1
  }
  return $matching_nets
}

proc find_matching_pins {cell pins match repl} {
  set matching_pins {}
  foreach pin $pins {
    set pin_name [str_replace $pin $match $repl]
    set matching_pin [get_cell_pin $cell $pin_name]
    if {$matching_pin != ""} {
      lappend matching_pins $matching_pin
    }
  }
  if {[llength $matching_pins] == 0} {
    puts "ERROR: No matching pins found for '$pins'."
    exit -1
  } elseif {[llength $matching_pins] != [llength $pins]} {
    puts "ERROR: Mismatch in number of matching pins."
    exit -1
  }
  return $matching_pins
}

proc replace_net_source {net source_pin} {
  foreach pin [find_net_sinks $net 0] {
    replace_pin_source $pin $source_pin
  }
}

proc resolve_async_bram {inst} {
  variable info
  variable debug

  puts "INFO: Resolving asynchronous BRAM patch: '$inst'."

  set hier_sep [get_hierarchy_separator]

  set raddr_w_nets [find_cell_nets $inst "raddr_w(\\\[\\d+\\\])?$"]
  set read_s_net [find_cell_nets $inst "read_s$"]
  set is_raddr_reg_net [find_cell_nets $inst "is_raddr_reg$"]

  set raddr_s_nets [find_matching_nets $inst $raddr_w_nets "raddr_w(\\\[\\d+\\\])?$" "raddr_s\\1"]

  set reg_next_pins {}
  set reg_ce_src_pin ""

  foreach raddr_w_net $raddr_w_nets {
    if {$debug} {puts "DEBUG: Processing raddr_w net: '$raddr_w_net'"}

    # Find raddr_w_net's driver pin
    set raddr_src_pin [find_net_driver $raddr_w_net]
    if {$debug} {puts "DEBUG: raddr_src_pin: '$raddr_src_pin'"}

    # Get the driver cell
    set raddr_src_cell [get_cells -of_objects $raddr_src_pin]
    if {[llength $raddr_src_cell] == 0} {
      puts "ERROR: No source cell found connected to pin '$raddr_src_pin'."
      exit -1
    } elseif {[llength $raddr_src_cell] > 1} {
      puts "ERROR: Multiple source cells found connected to pin '$raddr_src_pin'."
      exit -1
    }

    # Check driver type
    set driver_type [get_property REF_NAME $raddr_src_cell]
    if {$driver_type == "FDRE" || $driver_type == "FDSE"} {
      if {$debug} {puts "DEBUG: Net '$raddr_w_net' is registered, driver_type='$driver_type'"}
    } else {
      puts "WARNING: Net '$raddr_w_net' is not be registered, driver_type='$driver_type'"
      break
    }

    # Create register next cell and return output pin
    set reg_next_pin [create_register_next $inst $raddr_src_cell]
    if {$reg_next_pin == ""} {
      puts "ERROR: failed to create register next value for '$raddr_src_cell'."
      exit -1
    }
    if {$debug} {puts "DEBUG: reg_next_pin: '$reg_next_pin'"}

    lappend reg_next_pins $reg_next_pin

    # Find the CE pin on raddr_src_cell
    if {$reg_ce_src_pin == ""} {
      set reg_ce_pin [get_pins "${raddr_src_cell}${hier_sep}CE"]
      if {[llength $reg_ce_pin] == 0} {
        puts "ERROR: No CE pin found on register cell '$raddr_src_cell'."
        exit -1
      } elseif {[llength $reg_ce_pin] > 1} {
        puts "ERROR: Multiple CE pins found on register cell '$raddr_src_cell'."
        exit -1
      }
      if {$debug} {puts "DEBUG: reg_ce_pin: '$reg_ce_pin'"}

      set reg_ce_src_pin [find_pin_driver $reg_ce_pin]
      if {$reg_ce_src_pin == ""} {
        puts "ERROR: No source pin found connected to '$reg_ce_pin'."
        exit -1
      }
      if {$debug} {puts "DEBUG: reg_ce_src_pin: '$reg_ce_src_pin'"}
    }
  }

  # do we have a fully registered read address?
  if {[llength $reg_next_pins] == [llength $raddr_w_nets]} {
    if {$info} {puts "INFO: Fully registered read address detected."}

    # Connect all reg_next_pins to all input pins attached to raddr_s_nets
    set addr_width [llength $raddr_w_nets]
    for {set addr_idx 0} {$addr_idx < $addr_width} {incr addr_idx} {
      set raddr_s_net [lindex $raddr_s_nets $addr_idx]
      set reg_next_pin [lindex $reg_next_pins $addr_idx]
      if {$info} {puts "INFO: Connecting pin '$reg_next_pin' to '$raddr_s_net's pins."}
      # Connect reg_next_pin to all input pins attached to raddr_s_net
      replace_net_source $raddr_s_net $reg_next_pin
    }

    # Connect reg_ce_src_pin to all input pins attached to read_s_net
    if {$info} {puts "INFO: Connecting pin '$reg_ce_src_pin' to '$read_s_net's pins."}
    replace_net_source $read_s_net $reg_ce_src_pin

    # Create Const<1>'s pin
    set vcc_pin [getOrCreateVCCPin $inst]

    # Connect vcc_pin to all input pins attached to is_raddr_reg_net
    if {$info} {puts "INFO: Connecting pin '$vcc_pin' to '$is_raddr_reg_net's pins."}
    replace_net_source $is_raddr_reg_net $vcc_pin

    # Remove all async_ram cells
    foreach cell [find_nested_cells $inst "g_async_ram.*" 0] {
      remove_cell_from_netlist $cell
    }
  } else {
    puts "WARNING: Not all read addresses are registered!"

    # Create  Const<0>'s pin
    set gnd_pin [getOrCreateGNDPin $inst]

    # Connect gnd_pin to all input pins attached to is_raddr_reg_net
    if {$info} {puts "INFO: Connecting pin '$gnd_pin' to '$is_raddr_reg_net's pins."}
    replace_net_source $is_raddr_reg_net $gnd_pin

    # Remove all sync_ram cells
    foreach cell [find_nested_cells $inst "g_sync_ram.*" 0] {
      remove_cell_from_netlist $cell
    }
  }

  # Remove placeholder cell
  foreach cell [find_nested_cells $inst "placeholder$"] {
    remove_cell_from_netlist $cell
  }
}

proc resolve_async_brams {} {
  set bram_patch_cells {}
  foreach cell [get_cells -hierarchical -filter {REF_NAME =~ "*VX_async_ram_patch*"}] {
    puts "INFO: Found async BRAM patch cell: '$cell'."
    lappend bram_patch_cells $cell
  }
  if {[llength $bram_patch_cells] != 0} {
    foreach cell $bram_patch_cells {
      resolve_async_bram $cell
    }
  } else {
    puts "INFO: No async BRAM patch cells found in the design."
  }
}

proc dump_async_bram_cells {} {
  set bram_patch_cells [get_cells -hierarchical -filter {REF_NAME =~ "*VX_async_ram_patch*"}]
  if {[llength $bram_patch_cells] != 0} {
    foreach cell $bram_patch_cells {
      puts "INFO: Found async BRAM patch cell: '$cell'."
      set child_cells [find_cell_descendants $cell]
      foreach child $child_cells {
        set type [get_property REF_NAME $child]
        puts "INFO:   child cell: '$child', type: '$type'"
      }
    }
  } else {
    puts "INFO: No async BRAM patch cells found in the design."
  }
}

}

# Invoke the procedure to resolve async BRAM
vortex::resolve_async_brams

# dump async bram cells
#vortex::dump_async_bram_cells
