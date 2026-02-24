set clk_names [list \
  clk clock clk_i clock_i clk_in clock_in \
]

set reset_names [list \
  rst rst_i rst_in rst_n_i rst_ni rst_n_in \
  reset reset_i reset_in reset_n_i reset_ni reset_n_in \
]

set clk_port ""
foreach n $clk_names {
  set p [get_ports -quiet $n]
  if {[sizeof_collection $p] > 0} {
    set clk_port $p
    set clk_port_name $n
    break
  }
}

if {$clk_port eq ""} {
  puts "ERROR: No clock port found. Tried: $clk_names"
  exit 1
}

puts "INFO: Using clock port: $clk_port_name"

create_clock -name clk -period $target_period $clk_port
set_clock_uncertainty $target_uncertainty [get_clocks clk]

# Virtual Clock (for I/O)
create_clock -name ext_clk -period $target_period

# I/O Delays
set in_ports [remove_from_collection [all_inputs] [get_ports $clk_port]]
set_input_delay  $target_io_delay -clock ext_clk $in_ports
set_output_delay $target_io_delay -clock ext_clk [all_outputs]

# Prevent synthesis from trying to buffer the global reset tree
set rst_ports [get_ports -quiet $reset_names]
if {[sizeof_collection $rst_ports] > 0} {
  puts "INFO: Found reset ports: [get_object_name $rst_ports]"
  set_ideal_network $rst_ports
} else {
  puts "WARNING: No reset ports found matching your list."
}