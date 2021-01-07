`ifndef UTIL_DPI
`define UTIL_DPI

import "DPI-C" context function int dpi_register();
import "DPI-C" context function void dpi_assert(int inst, input logic cond, input int delay);

`endif