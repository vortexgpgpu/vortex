`ifndef VX_CACHE_CORE_RSP_IF
`define VX_CACHE_CORE_RSP_IF

`include "../cache/VX_cache_config.vh"

interface VX_cache_core_rsp_if #(
    parameter NUM_REQUESTS = 32
) ();

    // Core response
    wire [NUM_REQUESTS-1:0]        core_rsp_valid;
`IGNORE_WARNINGS_BEGIN
    wire [4:0]                     core_rsp_read;
    wire [1:0]                     core_rsp_write;
`IGNORE_WARNINGS_END    
    wire [NUM_REQUESTS-1:0][31:0]  core_rsp_pc;    
    wire [NUM_REQUESTS-1:0][31:0]  core_rsp_data;
    wire                           core_rsp_ready;
    
    // Core response meta data
    wire [`NW_BITS-1:0]            core_rsp_warp_num;      

endinterface

`endif