#!/bin/bash

# Generate Power Report
# first argument is the project name

quartus_pow --input_vcd=trace.vcd --vcd_filter_glitches=on --default_input_io_toggle_rate=10000transitions/s $1