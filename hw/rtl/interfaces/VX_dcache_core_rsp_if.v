`ifndef VX_DCACHE_CORE_RSP_IF
`define VX_DCACHE_CORE_RSP_IF

`include "../cache/VX_cache_config.vh"

interface VX_dcache_core_rsp_if #(
    parameter LANES          = 1,
    parameter WORD_SIZE      = 1,
    parameter CORE_TAG_WIDTH = 1
) ();

    wire [LANES-1:0]                valid;    
    wire [LANES-1:0][`WORD_WIDTH-1:0]data;
    wire [CORE_TAG_WIDTH-1:0]       tag;    
    wire                            ready;      

endinterface

`endif