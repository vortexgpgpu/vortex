`ifndef VX_CACHE_MEM_RSP_IF
`define VX_CACHE_MEM_RSP_IF

`include "../cache/VX_cache_define.vh"

interface VX_cache_mem_rsp_if #(
    parameter MEM_LINE_WIDTH = 1,
    parameter MEM_TAG_WIDTH  = 1
) ();

    wire                        valid;    
    wire [MEM_LINE_WIDTH-1:0]   data;
    wire [MEM_TAG_WIDTH-1:0]    tag;  
    wire                        ready;      

endinterface

`endif