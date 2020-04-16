`include "../VX_define.vh"

`ifndef VX_GPU_INST_REQ_IN

`define VX_GPU_INST_REQ_IN

interface VX_gpu_inst_req_inter();

	wire[`NUM_THREADS-1:0]       valid;
	wire[`NW_BITS-1:0]       warp_num;
	wire                 is_wspawn;
	wire                 is_tmc;   
	wire                 is_split; 

	wire                 is_barrier;

	wire[31:0]           pc_next;

	wire[`NUM_THREADS-1:0][31:0] a_reg_data;
	wire[31:0]           rd2;



endinterface


`endif