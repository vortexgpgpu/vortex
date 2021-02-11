`ifndef VX_ICACHE_CORE_RSP_IF
`define VX_ICACHE_CORE_RSP_IF

`include "../cache/VX_cache_config.vh"

interface VX_icache_core_rsp_if #(
    parameter WORD_SIZE        = 1,
    parameter CORE_TAG_WIDTH   = 1,
    parameter CORE_TAG_ID_BITS = 0
) ();

    wire                        valid;    
    wire [`WORD_WIDTH-1:0]      data;
    wire [CORE_TAG_WIDTH-1:0]   tag;    
    wire                        ready;      

endinterface

`endif