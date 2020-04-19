`ifndef VX_GPU_DCACHE_RSP
`define VX_GPU_DCACHE_RSP

`include "../generic_cache/VX_cache_config.vh"

interface VX_gpu_dcache_rsp_if #(
    parameter NUM_REQUESTS = 32
) ();

	// Cache WB
    wire [NUM_REQUESTS-1:0]        core_wb_valid;
`IGNORE_WARNINGS_BEGIN
    wire [4:0]                     core_wb_req_rd;
    wire [1:0]                     core_wb_req_wb;
`IGNORE_WARNINGS_END
    wire [`NW_BITS-1:0]            core_wb_warp_num;
    wire [NUM_REQUESTS-1:0][31:0]  core_wb_readdata;
    wire [NUM_REQUESTS-1:0][31:0]  core_wb_pc;

    // Cache Full
    wire                           delay_req;

endinterface

`endif