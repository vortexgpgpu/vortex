`ifndef VX_CACHE_CORE_RSP_IF
`define VX_CACHE_CORE_RSP_IF

`include "../cache/VX_cache_config.vh"

interface VX_cache_core_rsp_if #(
    parameter NUM_REQS         = 1,
    parameter WORD_SIZE        = 1,
    parameter CORE_TAG_WIDTH   = 1,
    parameter CORE_TAG_ID_BITS = 0
) ();

    wire [NUM_REQS-1:0]                                valid;    
    wire [NUM_REQS-1:0][`WORD_WIDTH-1:0]               data;
    wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0] tag;    
    wire                                               ready;      

endinterface

`endif