`include "../VX_define.vh"

`ifndef VX_WSTALL_INTER

`define VX_WSTALL_INTER


interface VX_wstall_inter();
	wire           wstall;
	wire[`NW_BITS-1:0] warp_num;
endinterface



`endif