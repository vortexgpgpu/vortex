`ifndef UTIL_DPI
`define UTIL_DPI

import "DPI-C" function void dpi_imul(input logic enable, input int a, input int b, input logic is_signed_a, input logic is_signed_b, output int resultl, output int resulth);
import "DPI-C" function void dpi_idiv(input logic enable, input int a, input int b, input logic is_signed, output int quotient, output int remainder);

import "DPI-C" function int dpi_register();
import "DPI-C" function void dpi_assert(int inst, input logic cond, input int delay);

import "DPI-C" function void dpi_trace(input string format /*verilator sformat*/);
import "DPI-C" function void dpi_trace_start();
import "DPI-C" function void dpi_trace_stop();

`endif
