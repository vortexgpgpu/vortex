`ifndef FLOAT_DPI
`define FLOAT_DPI

import "DPI-C" context function int dpi_register();

import "DPI-C" context function void dpi_fadd(int inst, input logic enable, input int a, input int b, output int result);
import "DPI-C" context function void dpi_fsub(int inst, input logic enable, input int a, input int b, output int result);
import "DPI-C" context function void dpi_fmul(int inst, input logic enable, input int a, input int b, output int result);
import "DPI-C" context function void dpi_fmadd(int inst, input logic enable, input int a, input int b, input int c, output int result);
import "DPI-C" context function void dpi_fmsub(int inst, input logic enable, input int a, input int b, input int c, output int result);
import "DPI-C" context function void dpi_fdiv(int inst, input logic enable, input int a, input int b, output int result);
import "DPI-C" context function void dpi_fsqrt(int inst, input logic enable, input int a, output int result);
import "DPI-C" context function void dpi_ftoi(int inst, input logic enable, input int a, output int result);
import "DPI-C" context function void dpi_ftou(int inst, input logic enable, input int a, output int result);
import "DPI-C" context function void dpi_itof(int inst, input logic enable, input int a, output int result);
import "DPI-C" context function void dpi_utof(int inst, input logic enable, input int a, output int result);

import "DPI-C" context function void dpi_delayed_assert(int inst, input logic cond);

`endif