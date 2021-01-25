create_clock -name {clk} -period "220 MHz" [get_ports {clk}]

derive_pll_clocks -create_base_clocks
derive_clock_uncertainty



