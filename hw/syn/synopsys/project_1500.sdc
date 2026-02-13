set_units -time ns

# On-chip clock
create_clock -name clk -period 0.667 [get_ports clk]  ;# 1.5 GHz
set_clock_uncertainty 0.05 [get_clocks clk]

# Virtual external clock for I/O timing
create_clock -name ext_clk -period 0.667

# I/O delays: exclude the clock port from input delays
set in_ports [filter_collection [all_inputs] "name != clk"]
set_input_delay  0.1 -clock ext_clk $in_ports
set_output_delay 0.1 -clock ext_clk [all_outputs]

# Prevent synthesis from trying to buffer the global reset tree
set_ideal_network [get_ports reset_n]