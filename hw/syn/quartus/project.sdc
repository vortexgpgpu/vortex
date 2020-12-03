set_time_format -unit ns -decimal_places 3

create_clock -name {clk} -period "220 MHz" -waveform { 0.0 1.0 } [get_ports {clk}]

derive_pll_clocks -create_base_clocks
derive_clock_uncertainty



