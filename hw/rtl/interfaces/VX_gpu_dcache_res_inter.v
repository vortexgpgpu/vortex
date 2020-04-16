

`include "../generic_cache/VX_cache_config.vh"

`ifndef VX_GPU_DCACHE_RES

`define VX_GPU_DCACHE_RES

interface VX_gpu_dcache_res_inter
	#(
		parameter NUMBER_REQUESTS = 32
	)
	();

	// Cache WB
    wire [NUMBER_REQUESTS-1:0]        core_wb_valid;
    wire [4:0]                        core_wb_req_rd;
    wire [1:0]                        core_wb_req_wb;
    wire [`NW_BITS-1:0]                   core_wb_warp_num;
    wire [NUMBER_REQUESTS-1:0][31:0]  core_wb_readdata;
    wire [NUMBER_REQUESTS-1:0][31:0]  core_wb_pc;

    // Cache Full
    wire                              delay_req;

endinterface


`endif