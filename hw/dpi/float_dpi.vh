`ifndef FLOAT_DPI
`define FLOAT_DPI

import "DPI-C" function void dpi_fadd(input logic enable, input int a, input int b, input bit[2:0] frm, output int result, output bit[4:0] fflags);
import "DPI-C" function void dpi_fsub(input logic enable, input int a, input int b, input bit[2:0] frm, output int result, output bit[4:0] fflags);
import "DPI-C" function void dpi_fmul(input logic enable, input int a, input int b, input bit[2:0] frm, output int result, output bit[4:0] fflags);
import "DPI-C" function void dpi_fmadd(input logic enable, input int a, input int b, input int c, input bit[2:0] frm, output int result, output bit[4:0] fflags);
import "DPI-C" function void dpi_fmsub(input logic enable, input int a, input int b, input int c, input bit[2:0] frm, output int result, output bit[4:0] fflags);
import "DPI-C" function void dpi_fnmadd(input logic enable, input int a, input int b, input int c, input bit[2:0] frm, output int result, output bit[4:0] fflags);
import "DPI-C" function void dpi_fnmsub(input logic enable, input int a, input int b, input int c, input bit[2:0] frm, output int result, output bit[4:0] fflags);

import "DPI-C" function void dpi_fdiv(input logic enable, input int a, input int b, input bit[2:0] frm, output int result, output bit[4:0] fflags);
import "DPI-C" function void dpi_fsqrt(input logic enable, input int a, input bit[2:0] frm, output int result, output bit[4:0] fflags);

import "DPI-C" function void dpi_ftoi(input logic enable, input int a, input bit[2:0] frm, output int result, output bit[4:0] fflags);
import "DPI-C" function void dpi_ftou(input logic enable, input int a, input bit[2:0] frm, output int result, output bit[4:0] fflags);
import "DPI-C" function void dpi_itof(input logic enable, input int a, input bit[2:0] frm, output int result, output bit[4:0] fflags);
import "DPI-C" function void dpi_utof(input logic enable, input int a, input bit[2:0] frm, output int result, output bit[4:0] fflags);

import "DPI-C" function void dpi_fclss(input logic enable, input int a, output int result);
import "DPI-C" function void dpi_fsgnj(input logic enable, input int a, input int b, output int result);
import "DPI-C" function void dpi_fsgnjn(input logic enable, input int a, input int b, output int result);
import "DPI-C" function void dpi_fsgnjx(input logic enable, input int a, input int b, output int result);

import "DPI-C" function void dpi_flt(input logic enable, input int a, input int b, output int result, output bit[4:0] fflags);
import "DPI-C" function void dpi_fle(input logic enable, input int a, input int b, output int result, output bit[4:0] fflags);
import "DPI-C" function void dpi_feq(input logic enable, input int a, input int b, output int result, output bit[4:0] fflags);
import "DPI-C" function void dpi_fmin(input logic enable, input int a, input int b, output int result, output bit[4:0] fflags);
import "DPI-C" function void dpi_fmax(input logic enable, input int a, input int b, output int result, output bit[4:0] fflags);

`endif
