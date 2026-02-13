create_clock -name clk -period $target_period [get_ports clk]
set_clock_uncertainty $target_uncertainty [get_clocks clk]

# Virtual Clock (for I/O)
create_clock -name ext_clk -period $target_period

# 2. I/O Delays
set in_ports [remove_from_collection [all_inputs] [get_ports clk]]
set_input_delay  $target_io_delay -clock ext_clk $in_ports
set_output_delay $target_io_delay -clock ext_clk [all_outputs]

# Prevent synthesis from trying to buffer the global reset tree
set rst_ports [get_ports -quiet "reset reset_n reset_i reset_n_i reset_ni rst rst_i rst_n_i rst_ni"]
if {[sizeof_collection $rst_ports] > 0} {
    set_ideal_network $rst_ports
}