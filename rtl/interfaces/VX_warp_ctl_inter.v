
`include "../VX_define.v"

`ifndef VX_WARP_CTL_INTER

`define VX_WARP_CTL_INTER

interface VX_warp_ctl_inter ();

	wire[`NW_M1:0] warp_num;
	wire           change_mask;
	wire[`NT_M1:0] thread_mask;

	wire           wspawn;
	wire[31:0]     wspawn_pc;
	wire[`NW-1:0]  wspawn_new_active;

	wire           ebreak;


	wire           is_split;
	wire[`NW_M1:0] split_warp_num;
	wire[`NT_M1:0] split_new_mask;
	wire[`NT_M1:0] split_later_mask;
	wire[31:0]     split_save_pc;


endinterface


`endif