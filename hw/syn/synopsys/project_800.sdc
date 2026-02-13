# On-chip clock
create_clock -name clk -period 1.25 [get_ports clk]  ;# 800 MHz
set_clock_uncertainty 0.05 [get_clocks clk]

# Virtual external clock for I/O timing
create_clock -name ext_clk -period 1.25

# I/O delays: exclude the clock port from input delays
set in_ports [filter_collection [all_inputs] "name != clk"]
set_input_delay  0.15 -clock ext_clk $in_ports
set_output_delay 0.15 -clock ext_clk [all_outputs]

# Optional: treat async reset as false path (uncomment if desired)
# set_false_path -from [get_ports reset_n]
