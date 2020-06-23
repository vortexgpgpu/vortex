`include "../VX_define.v"

`ifndef VX_WSTALL_INTER

`define VX_WSTALL_INTER


interface VX_wstall_inter();
	wire           wstall;
	wire[`NW_M1:0] warp_num;
endinterface



`endif