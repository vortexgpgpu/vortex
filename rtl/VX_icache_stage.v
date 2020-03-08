`include "VX_define.v"

module VX_icache_stage (
	input  wire              clk,
	input  wire              reset,
	output wire              icache_stage_delay,
	output wire[`NW_M1:0]    icache_stage_wid,
	output wire[`NT-1:0]     icache_stage_valids,
	VX_inst_meta_inter       fe_inst_meta_fi,
	VX_inst_meta_inter       fe_inst_meta_id,
	VX_icache_response_inter icache_response,
	VX_icache_request_inter  icache_request
);

		wire   valid_inst = (|fe_inst_meta_fi.valid);

		assign icache_request.pc_address 						= fe_inst_meta_fi.inst_pc;
		assign icache_request.out_cache_driver_in_valid 		= fe_inst_meta_fi.valid != 0;
		assign icache_request.out_cache_driver_in_mem_read		= `LW_MEM_READ;
		assign icache_request.out_cache_driver_in_mem_write		= `NO_MEM_WRITE;
	  	assign icache_request.out_cache_driver_in_data			= 32'b0;



	  	assign icache_stage_delay = icache_response.delay;

		assign fe_inst_meta_id.instruction = (!valid_inst || icache_response.delay) ? 32'b0 : icache_response.instruction;
		assign fe_inst_meta_id.inst_pc     = fe_inst_meta_fi.inst_pc;
		assign fe_inst_meta_id.warp_num    = fe_inst_meta_fi.warp_num;
		assign fe_inst_meta_id.valid       = fe_inst_meta_fi.valid & {`NT{!icache_stage_delay}};

		assign icache_stage_wid            = fe_inst_meta_fi.warp_num;
		assign icache_stage_valids         = fe_inst_meta_fi.valid & {`NT{!icache_stage_delay}};


endmodule