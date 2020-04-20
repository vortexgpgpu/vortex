`ifndef VX_GPU_DCACHE_RSP
`define VX_GPU_DCACHE_RSP

`include "../generic_cache/VX_cache_config.vh"

interface VX_gpu_dcache_rsp_if #(
    parameter NUM_REQUESTS = 32
) ();

	// Core response
    wire [NUM_REQUESTS-1:0]        core_wb_valid;
`IGNORE_WARNINGS_BEGIN
    wire [4:0]                     core_wb_req_rd;
    wire [1:0]                     core_wb_req_wb;
`IGNORE_WARNINGS_END    
    wire [NUM_REQUESTS-1:0][31:0]  core_wb_pc;    
    wire [NUM_REQUESTS-1:0][31:0]  core_wb_readdata;
    wire                           core_no_wb_slot;
    
    // Core response meta data
    wire [`NW_BITS-1:0]            core_wb_warp_num;      

endinterface

`endif