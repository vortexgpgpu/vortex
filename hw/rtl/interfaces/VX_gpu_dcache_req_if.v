`ifndef VX_GPU_DCACHE_REQ
`define VX_GPU_DCACHE_REQ

`include "../generic_cache/VX_cache_config.vh"

interface VX_gpu_dcache_req_if #(
	parameter NUM_REQUESTS = 32
) ();

	// Core Request
	wire [NUM_REQUESTS-1:0]         core_req_valid;
	wire [NUM_REQUESTS-1:0][31:0]   core_req_addr;
	wire [NUM_REQUESTS-1:0][31:0]   core_req_writedata;
	wire [NUM_REQUESTS-1:0][2:0]    core_req_mem_read;
	wire [NUM_REQUESTS-1:0][2:0]    core_req_mem_write;
    wire [4:0]                      core_req_rd;
    wire [NUM_REQUESTS-1:0][1:0]    core_req_wb;
    wire [`NW_BITS-1:0]             core_req_warp_num;
    wire [31:0]                     core_req_pc;

    // Can't WB
    wire                            core_no_wb_slot;

endinterface


`endif