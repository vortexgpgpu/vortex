`ifndef FLOAT_DPI
`define FLOAT_DPI

import "DPI-C" context function void dpi_fadd(input int a, input int b, input bit[2:0] frm, output int result, output bit[4:0] fflags);
import "DPI-C" context function void dpi_fsub(input int a, input int b, input bit[2:0] frm, output int result, output bit[4:0] fflags);
import "DPI-C" context function void dpi_fmul(input int a, input int b, input bit[2:0] frm, output int result, output bit[4:0] fflags);
import "DPI-C" context function void dpi_fmadd(input int a, input int b, input int c, input bit[2:0] frm, output int result, output bit[4:0] fflags);
import "DPI-C" context function void dpi_fmsub(input int a, input int b, input int c, input bit[2:0] frm, output int result, output bit[4:0] fflags);
import "DPI-C" context function void dpi_fnmadd(input int a, input int b, input int c, input bit[2:0] frm, output int result, output bit[4:0] fflags);
import "DPI-C" context function void dpi_fnmsub(input int a, input int b, input int c, input bit[2:0] frm, output int result, output bit[4:0] fflags);

import "DPI-C" context function void dpi_fdiv(input int a, input int b, input bit[2:0] frm, output int result, output bit[4:0] fflags);
import "DPI-C" context function void dpi_fsqrt(input int a, input bit[2:0] frm, output int result, output bit[4:0] fflags);

import "DPI-C" context function void dpi_ftoi(input int a, input bit[2:0] frm, output int result, output bit[4:0] fflags);
import "DPI-C" context function void dpi_ftou(input int a, input bit[2:0] frm, output int result, output bit[4:0] fflags);
import "DPI-C" context function void dpi_itof(input int a, input bit[2:0] frm, output int result, output bit[4:0] fflags);
import "DPI-C" context function void dpi_utof(input int a, input bit[2:0] frm, output int result, output bit[4:0] fflags);

import "DPI-C" context function void dpi_fclss(input int a, output int result);
import "DPI-C" context function void dpi_fsgnj(input int a, input int b, output int result);
import "DPI-C" context function void dpi_fsgnjn(input int a, input int b, output int result);
import "DPI-C" context function void dpi_fsgnjx(input int a, input int b, output int result);

import "DPI-C" context function void dpi_flt(input int a, input int b, output int result, output bit[4:0] fflags);
import "DPI-C" context function void dpi_fle(input int a, input int b, output int result, output bit[4:0] fflags);
import "DPI-C" context function void dpi_feq(input int a, input int b, output int result, output bit[4:0] fflags);
import "DPI-C" context function void dpi_fmin(input int a, input int b, output int result, output bit[4:0] fflags);
import "DPI-C" context function void dpi_fmax(input int a, input int b, output int result, output bit[4:0] fflags);

`endif