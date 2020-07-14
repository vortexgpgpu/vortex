`ifndef VX_CACHE_CORE_REQ_IF
`define VX_CACHE_CORE_REQ_IF

`include "../cache/VX_cache_config.vh"

interface VX_cache_core_req_if #(
    parameter NUM_REQUESTS     = 1,
    parameter WORD_SIZE        = 1,
    parameter CORE_TAG_WIDTH   = 1,
    parameter CORE_TAG_ID_BITS = 0
) ();

    wire [NUM_REQUESTS-1:0]                             valid;
    wire [NUM_REQUESTS-1:0]                             rw;
    wire [NUM_REQUESTS-1:0][WORD_SIZE-1:0]              byteen;
    wire [NUM_REQUESTS-1:0][`WORD_ADDR_WIDTH-1:0]       addr;
    wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]            data;
    wire [`CORE_REQ_TAG_COUNT-1:0][CORE_TAG_WIDTH-1:0]  tag;    
    wire                                                ready;

endinterface

`endif