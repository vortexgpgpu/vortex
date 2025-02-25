set CLK_FREQ_MHZ 300
set clk_port_name clk
set clk_port [get_ports $clk_port_name]
create_clock -name core_clock -period [expr 1000.0 / $CLK_FREQ_MHZ] $clk_port