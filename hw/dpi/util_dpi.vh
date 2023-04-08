`include "VX_config.vh"
`ifndef UTIL_DPI
`define UTIL_DPI

`ifdef MODE_32_BIT
`define INT_LEN int
`else
`define INT_LEN longint
`endif

import "DPI-C" function void dpi_imul(input logic enable, input `INT_LEN a, input `INT_LEN b, input logic is_signed_a, input logic is_signed_b, output `INT_LEN resultl, output `INT_LEN resulth);
import "DPI-C" function void dpi_idiv(input logic enable, input `INT_LEN a, input `INT_LEN b, input logic is_signed, output `INT_LEN quotient, output `INT_LEN remainder);

import "DPI-C" function int dpi_register();
import "DPI-C" function void dpi_assert(int inst, input logic cond, input int delay);

import "DPI-C" function void dpi_trace(input int level, input string format /*verilator sformat*/);
import "DPI-C" function void dpi_trace_start();
import "DPI-C" function void dpi_trace_stop();

`endif
