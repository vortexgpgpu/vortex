# On-chip clock
create_clock -name clk -period 6.0 [get_ports clk]
set_clock_uncertainty 0.05 [get_clocks clk]

# Virtual external clock for I/O timing
create_clock -name ext_clk -period 6.0

# I/O delays: exclude the clock port from input delays
set _all_in  [all_inputs]
set _in_wo_clk {}
foreach p $_all_in {
  if {$p ne "clk"} { lappend _in_wo_clk $p }
}
set_input_delay  0.40 -clock ext_clk $_in_wo_clk
set_output_delay 0.40 -clock ext_clk [all_outputs]

# Optional: treat async reset as false path (uncomment if desired)
# set_false_path -from [get_ports reset_n]
