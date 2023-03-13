`ifndef UTIL_DPI
`define UTIL_DPI

import "DPI-C" function void dpi_imul(input logic enable, input longint a, input longint b, input logic is_signed_a, input logic is_signed_b, output longint resultl, output longint resulth);
import "DPI-C" function void dpi_idiv(input logic enable, input longint a, input longint b, input logic is_signed, output longint quotient, output longint remainder);

import "DPI-C" function int dpi_register();
import "DPI-C" function void dpi_assert(int inst, input logic cond, input int delay);

import "DPI-C" function void dpi_trace(input int level, input string format /*verilator sformat*/);
import "DPI-C" function void dpi_trace_start();
import "DPI-C" function void dpi_trace_stop();

`endif
