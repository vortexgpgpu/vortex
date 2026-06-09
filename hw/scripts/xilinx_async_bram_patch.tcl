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

variable debug 0
variable start_time [clock seconds]
variable cpu_start_time [clock clicks -milliseconds]

# --- Standardized Logging Wrappers ---

proc print_msg {level msg} {
  variable debug
  set prefix "\[AsyncBRAM Patch\]"
  switch $level {
    "ERROR"    { puts "ERROR: $prefix $msg" }
    "CRITICAL" { puts "CRITICAL WARNING: $prefix $msg" }
    "WARNING"  { puts "WARNING: $prefix $msg" }
    "INFO"     { puts "INFO: $prefix $msg" }
    "DEBUG"    { if {$debug} { puts "DEBUG: $prefix $msg" } }
  }
}

proc log_error {msg {exit_code -1}} {
  print_msg "ERROR" $msg
  if {$exit_code != 0} { exit $exit_code }
}

proc log_critical {msg} {
  print_msg "CRITICAL" $msg
}

proc log_warning {msg} {
  print_msg "WARNING" $msg
}

proc log_info {msg} {
  print_msg "INFO" $msg
}

proc log_debug {msg} {
  print_msg "DEBUG" $msg
}

# --- Time Formatting Utils ---

proc format_time_val {total_seconds} {
  set s [expr {int($total_seconds)}]
  set ms [expr {int(($total_seconds - $s) * 100)}]
  set h [expr {$s / 3600}]
  set m [expr {($s % 3600) / 60}]
  set s [expr {$s % 60}]

  if {$ms > 0} {
      return [format "%02d:%02d:%02d.%02d" $h $m $s $ms]
  } else {
      return [format "%02d:%02d:%02d" $h $m $s]
  }
}

proc log_time {} {
  variable start_time
  variable cpu_start_time

  set now [clock seconds]
  set cpu_now [clock clicks -milliseconds]

  set elapsed_s [expr {$now - $start_time}]
  set cpu_s [expr {($cpu_now - $cpu_start_time) / 1000.0}]

  set cpu_fmt [format_time_val $cpu_s]
  set elapsed_fmt [format_time_val $elapsed_s]

  puts "INFO: \[AsyncBRAM Patch\] Time (s): cpu = $cpu_fmt ; elapsed = $elapsed_fmt ."
}

# --- String & Regex Utilities ---

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

# --- Net Finding Utilities ---

proc find_cell_nets {cell name_match {required 1}} {
  set matching_nets {}
  foreach net [get_nets -hierarchical -filter "PARENT_CELL == $cell"] {
    set name [get_property NAME $net]
    if {[regexp $name_match $name]} {
      lappend matching_nets $net
    }
  }

  if {[llength $matching_nets] == 0 && $required} {
    log_error "No matching net found for '$cell' matching '$name_match'."
  }
  return $matching_nets
}

proc find_cell_net {cell name_match {required 1}} {
  set nets [find_cell_nets $cell $name_match $required]

  if {[llength $nets] == 0} {
    return ""
  } elseif {[llength $nets] > 1} {
    log_error "Multiple matching nets found for '$cell' matching '$name_match'."
  }
  return [lindex $nets 0]
}

proc get_cell_net {cell name} {
  set net [get_nets -hierarchical -filter "PARENT_CELL == $cell && NAME == $name"]
  if {[llength $net] == 0} {
    log_error "No matching net found for '$cell' matching '$name'."
  }
  return $net
}

proc get_cell_pin {cell name} {
  set pin [get_pins -of_objects $cell -filter "NAME == $name"]
  if {[llength $pin] == 0} {
    log_error "No matching pin found for '$cell' matching '$name'."
  }
  return $pin
}

# --- Modification Utilities ---

proc remove_cell_from_netlist {cell} {
  foreach pin [get_pins -quiet -of_objects $cell] {
    foreach net [get_nets -quiet -of_objects $pin] {
      disconnect_net -net $net -objects $pin
      log_debug "Disconnected net '$net' from pin '$pin'."
    }
  }
  remove_cell $cell
  log_debug "Cell '$cell' was removed successfully."
}

proc find_net_driver {target_net {required 1}} {
  set driverPins [get_pins -quiet -leaf -of_objects $target_net -filter {DIRECTION == "OUT"}]

  if {[llength $driverPins] == 0} {
    set driverPorts [get_ports -quiet -of_objects $target_net -filter {DIRECTION == "IN"}]

    if {[llength $driverPorts] == 0} {
      if {$required} { log_error "No driver found for '$target_net'." }
      return ""
    } elseif {[llength $driverPorts] > 1} {
      log_warning "Multiple driver ports found for '$target_net'. Using first."
      return [lindex $driverPorts 0]
    }
    return $driverPorts
  } elseif {[llength $driverPins] > 1} {
    log_warning "Multiple driver pins found for '$target_net'. Using first."
    return [lindex $driverPins 0]
  }
  return $driverPins
}

proc find_pin_driver {target_pin {required 1}} {
  set net [get_nets -quiet -of_objects $target_pin]

  if {[llength $net] == 0} {
    if {$required} { log_error "No net connected to pin '$target_pin'." }
    return ""
  } elseif {[llength $net] > 1} {
    log_error "Multiple nets connected to pin '$target_pin'."
  }
  return [find_net_driver $net]
}

# --- Core Logic ---

proc create_register_next {parent reg_cell raddr_reset} {
  set hier_sep [get_hierarchy_separator]

  set reg_d_pin [get_pins "${reg_cell}${hier_sep}D"]
  if {[llength $reg_d_pin] != 1} {
    log_error "Found [llength $reg_d_pin] D pins on register cell '$reg_cell'. Expected 1."
  }

  log_debug "reg_d_pin: '$reg_d_pin'"

  set reg_d_src_pin [find_pin_driver $reg_d_pin]
  log_debug "reg_d_src_pin: '$reg_d_src_pin'"

  # [Optimization] 1. Check if reset is requested by the patch instance
  if {$raddr_reset == ""} {
    log_debug "No reset requested by patch. Skipping LUT."
    return $reg_d_src_pin
  }

  set register_type [get_property REF_NAME $reg_cell]
  set reg_r_pin ""
  set lut_init ""

  if {$register_type == "FDRE"} {
    set reg_r_pin [get_pins -quiet "${reg_cell}${hier_sep}R"]
    set lut_init "4'b0010" ;# Reset dominates
  } elseif {$register_type == "FDSE"} {
    set reg_r_pin [get_pins -quiet "${reg_cell}${hier_sep}S"]
    set lut_init "4'b1110" ;# Set dominates
  } else {
    log_error "Unsupported register type: '$register_type'."
  }

  if {[llength $reg_r_pin] == 0} {
    log_error "No Reset/Set pin found on $register_type cell '$reg_cell'."
  }

  set reg_r_src_pin [find_pin_driver $reg_r_pin 0]

  # [Optimization] 2. Check if the register's reset pin is unconnected or grounded
  if {$reg_r_src_pin == ""} {
    log_debug "Reset pin unconnected. Optimization: Skipping LUT."
    return $reg_d_src_pin
  } else {
    set driver_cell [get_cells -quiet -of_objects $reg_r_src_pin]
    if {[llength $driver_cell] > 0} {
        set driver_ref [get_property REF_NAME $driver_cell]
        if {$driver_ref == "GND"} {
            log_debug "Reset driven by GND. Optimization: Skipping LUT."
            return $reg_d_src_pin
        }
    }
  }

  log_debug "reg_r_src_pin: '$reg_r_src_pin'"

  set reg_d_src_net [get_nets -of_objects $reg_d_src_pin]
  set reg_r_src_net [get_nets -of_objects $reg_r_src_pin]

  # Create LUT2 MUX
  set lut_name [unique_cell_name "${parent}${hier_sep}raddr_next"]
  set lut_cell [create_cell -reference LUT2 $lut_name]
  set_property INIT $lut_init $lut_cell

  log_debug "Created LUT cell: '$lut_cell' (Init: $lut_init)"

  set lut_i0_pin [get_pins "${lut_cell}${hier_sep}I0"]
  set lut_i1_pin [get_pins "${lut_cell}${hier_sep}I1"]
  set lut_o_pin  [get_pins "${lut_cell}${hier_sep}O"]

  connect_net -net $reg_d_src_net -objects $lut_i0_pin -hierarchical
  connect_net -net $reg_r_src_net -objects $lut_i1_pin -hierarchical

  return $lut_o_pin
}

proc getOrCreateVCCPin {parent} {
  set hier_sep [get_hierarchy_separator]
  set cell_name "${parent}${hier_sep}VCC"

  set vcc_cell [get_cells -quiet $cell_name]
  if {[llength $vcc_cell] == 0} {
    set vcc_cell [create_cell -reference VCC $cell_name]
    log_debug "Created VCC cell: '$vcc_cell'"
  }

  set vcc_pin [get_pins "${vcc_cell}${hier_sep}P"]
  if {[llength $vcc_pin] == 0} {
    log_error "No VCC pin found on VCC cell '$vcc_cell'."
  }
  return $vcc_pin
}

proc getOrCreateGNDPin {parent} {
  set hier_sep [get_hierarchy_separator]
  set cell_name "${parent}${hier_sep}GND"

  set gnd_cell [get_cells -quiet $cell_name]
  if {[llength $gnd_cell] == 0} {
    set gnd_cell [create_cell -reference GND $cell_name]
    log_debug "Created GND cell: '$gnd_cell'"
  }

  set gnd_pin [get_pins "${gnd_cell}${hier_sep}G"]
  if {[llength $gnd_pin] == 0} {
    log_error "No GND pin found on GND cell '$gnd_cell'."
  }
  return $gnd_pin
}

proc find_net_sinks {source_net {required 1}} {
  set sink_pins {}

  foreach pin [get_pins -quiet -of_objects $source_net] {
    set direction [get_property DIRECTION $pin]
    # Input pins of nested cells
    if {$direction == "IN"} {
      lappend sink_pins $pin
    }
    # Output pins of the parent cell
    set pin_cell [get_cells -of_objects $pin]
    set is_primitive [get_property IS_PRIMITIVE $pin_cell]
    if {$direction == "OUT" && !$is_primitive} {
      lappend sink_pins $pin
    }
  }

  foreach port [get_ports -quiet -of_objects $source_net -filter {DIRECTION == "OUT"}] {
    lappend sink_pins $port
  }

  if {[llength $sink_pins] == 0 && $required} {
    log_error "No sink found for '$source_net'."
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

  if {[llength $matching_nets] != [llength $nets]} {
    log_error "Mismatch in number of matching nets. Found [llength $matching_nets], expected [llength $nets]."
  }
  return $matching_nets
}

proc replace_net_source {net source_pin} {
  foreach pin [find_net_sinks $net 0] {
    # 1. Disconnect
    disconnect_net -net $net -objects $pin
    log_debug "Disconnected net '$net' from pin '$pin'."

    # 2. Find/Create Source Net
    set source_net [get_nets -quiet -of_objects $source_pin]

    if {[llength $source_net] == 0} {
      set source_cell [get_cells -of_objects $source_pin]
      set net_name [unique_net_name "${source_cell}_tmp_net"]
      set source_net [create_net $net_name]
      connect_net -net $source_net -objects $source_pin -hierarchical
      log_debug "Created new source_net '$source_net' and connected to '$source_pin'."
    } elseif {[llength $source_net] > 1} {
      log_error "Multiple nets connected to pin '$source_pin'."
    }

    # 3. Connect Sink
    set external_net [get_nets -quiet -of_objects $pin]

    if {[llength $external_net] == 0} {
      connect_net -net $source_net -objects $pin -hierarchical
      log_debug "Connected net '$source_net' to pin '$pin'."
    } elseif {[llength $external_net] == 1} {
      foreach external_pin [get_pins -of_objects $external_net] {
        disconnect_net -net $external_net -objects $pin
        replace_net_source $external_net $source_pin
      }
    } else {
      log_error "Multiple nets connected to pin '$pin'."
    }
  }
}

# --- Main Patching Routine ---

proc resolve_async_bram {inst} {
  log_info "Resolving asynchronous BRAM patch: '$inst'."

  set hier_sep [get_hierarchy_separator]

  # 1. Find Nets
  set raddr_w_nets [find_cell_nets $inst "raddr_w(\\\[\\d+\\\])?$"]
  set raddr_s_nets [find_matching_nets $inst $raddr_w_nets "raddr_w(\\\[\\d+\\\])?$" "raddr_s\\1"]

  set read_s_net [find_cell_net $inst "read_s$"]
  log_debug "read_s_net: '$read_s_net'"

  set is_raddr_reg_net [find_cell_net $inst "g_async_ram.is_raddr_reg$" 0]
  set raddr_reset_net [find_cell_net $inst "raddr_reset$" 0]
  log_debug "raddr_reset: '$raddr_reset_net'"

  set reg_next_pins {}
  set reg_ce_src_pin ""

  # 2. Analyze Drivers & Create Logic
  foreach raddr_w_net $raddr_w_nets {
    log_debug "Processing raddr_w net: '$raddr_w_net'"

    set raddr_src_pin [find_net_driver $raddr_w_net]

    if {[get_ports -quiet $raddr_src_pin] ne ""} {
      log_warning "Net '$raddr_w_net' driven by port. Skipping patch for this instance."
      break
    }

    set raddr_src_cell [get_cells -of_objects $raddr_src_pin]
    set driver_type [get_property REF_NAME $raddr_src_cell]

    if {$driver_type != "FDRE" && $driver_type != "FDSE"} {
      log_warning "Net '$raddr_w_net' driven by '$driver_type' (not FDRE/FDSE). Skipping."
      break
    }

    # Create Next Logic (with optimization)
    set reg_next_pin [create_register_next $inst $raddr_src_cell $raddr_reset_net]
    if {$reg_next_pin == ""} {
      log_error "Failed to create register next value for '$raddr_src_cell'."
    }

    lappend reg_next_pins $reg_next_pin

    # Capture CE Pin (once)
    if {$reg_ce_src_pin == ""} {
      set reg_ce_pin [get_pins "${raddr_src_cell}${hier_sep}CE"]
      if {[llength $reg_ce_pin] != 1} {
        log_error "Expected 1 CE pin on '$raddr_src_cell', found [llength $reg_ce_pin]."
      }
      set reg_ce_src_pin [find_pin_driver $reg_ce_pin]
      log_debug "reg_ce_src_pin: '$reg_ce_src_pin'"
    }
  }

  set addr_width [llength $raddr_w_nets]

  # 3. Apply Patch or Fallback
  if {[llength $reg_next_pins] == $addr_width} {
    log_debug "Fully registered read address detected. Patching..."

    # Connect Address
    for {set i 0} {$i < $addr_width} {incr i} {
      replace_net_source [lindex $raddr_s_nets $i] [lindex $reg_next_pins $i]
    }

    # Connect CE
    replace_net_source $read_s_net $reg_ce_src_pin

    # Connect Const 1 (Valid Patch)
    if {$is_raddr_reg_net != ""} {
      set vcc_pin [getOrCreateVCCPin $inst]
      replace_net_source $is_raddr_reg_net $vcc_pin
    }
  } else {
    log_critical "Read address not fully registered. Falling back to Asynchronous (Grounding)."

    set gnd_pin [getOrCreateGNDPin $inst]

    # Ground Address & Control
    foreach net $raddr_s_nets { replace_net_source $net $gnd_pin }
    replace_net_source $read_s_net $gnd_pin

    # Connect Const 0 (Invalid Patch)
    if {$is_raddr_reg_net != ""} {
      replace_net_source $is_raddr_reg_net $gnd_pin
    }
  }

  # 4. Cleanup (Optimized for Speed)
  remove_cell_from_netlist [get_cells -quiet "${inst}${hier_sep}placeholder1"]
  if {$is_raddr_reg_net != ""} {
    remove_cell_from_netlist [get_cells -quiet "${inst}${hier_sep}*placeholder2*"]
  }
}

proc resolve_async_brams {} {
  variable debug
  set bram_patch_cells [get_cells -hierarchical -filter {REF_NAME =~ "*VX_async_ram_patch*"}]
  if {[llength $bram_patch_cells] != 0} {
    foreach cell $bram_patch_cells {
      resolve_async_bram $cell
    }
  } else {
    log_info "No async BRAM patch cells found in the design."
  }
}

proc dump_cell_hierarchy {parent_cell indent} {
  set children [get_cells -quiet -hierarchical -filter "PARENT_CELL == $parent_cell"]
  foreach child $children {
    set type [get_property REF_NAME $child]
    log_info "${indent}child cell: '$child', type: '$type'"
    dump_cell_hierarchy $child "${indent}  "
  }
}

proc dump_async_bram_cells {} {
  set bram_patch_cells [get_cells -hierarchical -filter {REF_NAME =~ "*VX_async_ram_patch*"}]
  if {[llength $bram_patch_cells] != 0} {
    foreach cell $bram_patch_cells {
      log_info "Found async BRAM patch cell: '$cell'."
      dump_cell_hierarchy $cell "  "
    }
  } else {
    log_info "No async BRAM patch cells found in the design."
  }
}

}

# Run
vortex::resolve_async_brams
# vortex::dump_async_bram_cells
vortex::log_time