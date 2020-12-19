`ifndef VX_CACHE_DRAM_REQ_IF
`define VX_CACHE_DRAM_REQ_IF

`include "../cache/VX_cache_config.vh"

interface VX_cache_dram_req_if #(
    parameter DRAM_LINE_WIDTH = 1,
    parameter DRAM_ADDR_WIDTH = 1,
    parameter DRAM_TAG_WIDTH  = 1
) ();

    wire                           valid;    
    wire                           rw;    
    wire [(DRAM_LINE_WIDTH/8)-1:0] byteen;
    wire [DRAM_ADDR_WIDTH-1:0]     addr;
    wire [DRAM_LINE_WIDTH-1:0]     data;  
    wire [DRAM_TAG_WIDTH-1:0]      tag;  
    wire                           ready;

endinterface

`endif