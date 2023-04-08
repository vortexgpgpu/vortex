`include "VX_config.vh"

`ifndef FLOAT_DPI
`define FLOAT_DPI

`ifdef MODE_32_BIT
`define INT_LEN int
`else
`define INT_LEN longint
`endif

import "DPI-C" function void dpi_fadd(input logic enable, input `INT_LEN a, input `INT_LEN b, input bit[2:0] frm, output `INT_LEN result, output bit[4:0] fflags);
import "DPI-C" function void dpi_fsub(input logic enable, input `INT_LEN a, input `INT_LEN b, input bit[2:0] frm, output `INT_LEN result, output bit[4:0] fflags);
import "DPI-C" function void dpi_fmul(input logic enable, input `INT_LEN a, input `INT_LEN b, input bit[2:0] frm, output `INT_LEN result, output bit[4:0] fflags);
import "DPI-C" function void dpi_fmadd(input logic enable, input `INT_LEN a, input `INT_LEN b, input `INT_LEN c, input bit[2:0] frm, output `INT_LEN result, output bit[4:0] fflags);
import "DPI-C" function void dpi_fmsub(input logic enable, input `INT_LEN a, input `INT_LEN b, input `INT_LEN c, input bit[2:0] frm, output `INT_LEN result, output bit[4:0] fflags);
import "DPI-C" function void dpi_fnmadd(input logic enable, input `INT_LEN a, input `INT_LEN b, input `INT_LEN c, input bit[2:0] frm, output `INT_LEN result, output bit[4:0] fflags);
import "DPI-C" function void dpi_fnmsub(input logic enable, input `INT_LEN a, input `INT_LEN b, input `INT_LEN c, input bit[2:0] frm, output `INT_LEN result, output bit[4:0] fflags);

import "DPI-C" function void dpi_fdiv(input logic enable, input `INT_LEN a, input `INT_LEN b, input bit[2:0] frm, output `INT_LEN result, output bit[4:0] fflags);
import "DPI-C" function void dpi_fsqrt(input logic enable, input `INT_LEN a, input bit[2:0] frm, output `INT_LEN result, output bit[4:0] fflags);

import "DPI-C" function void dpi_ftoi(input logic enable, input `INT_LEN a, input bit[2:0] frm, output `INT_LEN result, output bit[4:0] fflags);
import "DPI-C" function void dpi_ftou(input logic enable, input `INT_LEN a, input bit[2:0] frm, output `INT_LEN result, output bit[4:0] fflags);
import "DPI-C" function void dpi_itof(input logic enable, input `INT_LEN a, input bit[2:0] frm, output `INT_LEN result, output bit[4:0] fflags);
import "DPI-C" function void dpi_utof(input logic enable, input `INT_LEN a, input bit[2:0] frm, output `INT_LEN result, output bit[4:0] fflags);

import "DPI-C" function void dpi_fclss(input logic enable, input `INT_LEN a, output `INT_LEN result);
import "DPI-C" function void dpi_fsgnj(input logic enable, input `INT_LEN a, input `INT_LEN b, output `INT_LEN result);
import "DPI-C" function void dpi_fsgnjn(input logic enable, input `INT_LEN a, input `INT_LEN b, output `INT_LEN result);
import "DPI-C" function void dpi_fsgnjx(input logic enable, input `INT_LEN a, input `INT_LEN b, output `INT_LEN result);

import "DPI-C" function void dpi_flt(input logic enable, input `INT_LEN a, input `INT_LEN b, output `INT_LEN result, output bit[4:0] fflags);
import "DPI-C" function void dpi_fle(input logic enable, input `INT_LEN a, input `INT_LEN b, output `INT_LEN result, output bit[4:0] fflags);
import "DPI-C" function void dpi_feq(input logic enable, input `INT_LEN a, input `INT_LEN b, output `INT_LEN result, output bit[4:0] fflags);
import "DPI-C" function void dpi_fmin(input logic enable, input `INT_LEN a, input `INT_LEN b, output `INT_LEN result, output bit[4:0] fflags);
import "DPI-C" function void dpi_fmax(input logic enable, input `INT_LEN a, input `INT_LEN b, output `INT_LEN result, output bit[4:0] fflags);

`endif
