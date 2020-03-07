

`include "../VX_cache/VX_cache_config.v"

`ifndef VX_GPU_DCACHE_REQ

`define VX_GPU_DCACHE_REQ

interface VX_gpu_dcache_req_inter
	#(
	parameter NUMBER_REQUESTS               = 32
	)
	();

	// Core Request
	wire [NUMBER_REQUESTS-1:0]         core_req_valid;
	wire [NUMBER_REQUESTS-1:0][31:0]   core_req_addr;
	wire [NUMBER_REQUESTS-1:0][31:0]   core_req_writedata;
	wire [2:0]                         core_req_mem_read;
	wire [2:0]                         core_req_mem_write;
    wire [4:0]                         core_req_rd;
    wire [1:0]                         core_req_wb;
    wire [`NW_M1:0]                    core_req_warp_num;
    wire [31:0]                        core_req_pc;

    // Can't WB
    wire                              core_no_wb_slot;

endinterface


`endif