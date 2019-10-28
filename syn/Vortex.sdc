###################################################################

# Created by write_sdc on Mon Oct 28 17:09:02 2019

###################################################################
set sdc_version 1.9

set_units -time ns -resistance kOhm -capacitance pF -voltage V -current mA
set_max_fanout 20 [get_ports clk]
set_max_fanout 20 [get_ports reset]
set_propagated_clock [get_ports clk]
create_clock [get_ports clk]  -period 10  -waveform {0 5}
set_false_path   -from [get_ports reset]
