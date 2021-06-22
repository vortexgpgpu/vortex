`ifndef UTIL_DPI
`define UTIL_DPI

import "DPI-C" context function void dpi_imul(input int a, input int b, input logic is_signed_a, input logic is_signed_b, output int resultl, output int resulth);
import "DPI-C" context function void dpi_idiv(input int a, input int b, input logic is_signed, output int quotient, output int remainder);

import "DPI-C" context function int dpi_register();
import "DPI-C" context function void dpi_assert(int inst, input logic cond, input int delay);

`endif