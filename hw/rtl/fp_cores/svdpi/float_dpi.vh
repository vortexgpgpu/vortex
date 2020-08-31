`ifndef FLOAT_DPI
`define FLOAT_DPI

import "DPI-C" context function void dpi_fadd(input logic clk, input logic enable, input int a, input int b, output int result);
import "DPI-C" context function void dpi_fsub(input logic clk, input logic enable, input int a, input int b, output int result);
import "DPI-C" context function void dpi_fmul(input logic clk, input logic enable, input int a, input int b, output int result);
import "DPI-C" context function void dpi_fmadd(input logic clk, input logic enable, input int a, input int b, input int c, output int result);
import "DPI-C" context function void dpi_fmsub(input logic clk, input logic enable, input int a, input int b, input int c, output int result);
import "DPI-C" context function void dpi_fdiv(input logic clk, input logic enable, input int a, input int b, output int result);
import "DPI-C" context function void dpi_fsqrt(input logic clk, input logic enable, input int a, output int result);
import "DPI-C" context function void dpi_ftoi(input logic clk, input logic enable, input int a, output int result);
import "DPI-C" context function void dpi_ftou(input logic clk, input logic enable, input int a, output int result);
import "DPI-C" context function void dpi_itof(input logic clk, input logic enable, input int a, output int result);
import "DPI-C" context function void dpi_utof(input logic clk, input logic enable, input int a, output int result);

`endif