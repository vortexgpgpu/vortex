`ifndef UTIL_DPI_VH
`define UTIL_DPI_VH

`include "VX_config.vh"

`ifdef XLEN_64
`define INT_TYPE longint
`else
`define INT_TYPE int
`endif

import "DPI-C" function void dpi_imul(input logic enable, input logic is_signed_a, input logic is_signed_b, input `INT_TYPE a, input `INT_TYPE b, output `INT_TYPE resultl, output `INT_TYPE resulth);
import "DPI-C" function void dpi_idiv(input logic enable, input logic is_signed, input `INT_TYPE a, input `INT_TYPE b, output `INT_TYPE quotient, output `INT_TYPE remainder);

import "DPI-C" function int dpi_register();
import "DPI-C" function void dpi_assert(int inst, input logic cond, input int delay);

import "DPI-C" function void dpi_trace(input int level, input string format /*verilator sformat*/);
import "DPI-C" function void dpi_trace_start();
import "DPI-C" function void dpi_trace_stop();

`endif
