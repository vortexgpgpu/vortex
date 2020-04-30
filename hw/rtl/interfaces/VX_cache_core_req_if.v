`ifndef VX_CACHE_CORE_REQ_IF
`define VX_CACHE_CORE_REQ_IF

`include "../cache/VX_cache_config.vh"

interface VX_cache_core_req_if #(
    parameter NUM_REQUESTS      = 1,
    parameter WORD_SIZE         = 1,
    parameter CORE_TAG_WIDTH    = 1
) ();

    wire [NUM_REQUESTS-1:0]                     core_req_valid;
    wire [NUM_REQUESTS-1:0][`WORD_SEL_BITS-1:0] core_req_read;
    wire [NUM_REQUESTS-1:0][`WORD_SEL_BITS-1:0] core_req_write;
    wire [NUM_REQUESTS-1:0][31:0]               core_req_addr;
    wire [NUM_REQUESTS-1:0][`WORD_WIDTH-1:0]    core_req_data;
    wire [NUM_REQUESTS-1:0][CORE_TAG_WIDTH-1:0] core_req_tag;    
    wire                                        core_req_ready;

endinterface

`endif