`ifndef VX_CACHE_DRAM_RSP_IF
`define VX_CACHE_DRAM_RSP_IF

`include "../cache/VX_cache_config.vh"

interface VX_cache_dram_rsp_if #(
    parameter DRAM_LINE_WIDTH = 1,
    parameter DRAM_TAG_WIDTH  = 1
) ();

    wire                        valid;
    
    wire [DRAM_LINE_WIDTH-1:0]  data;
    wire [DRAM_TAG_WIDTH-1:0]   tag;  

    wire                        ready;      

endinterface

`endif