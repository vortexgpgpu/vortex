`ifndef VX_CACHE_MEM_REQ_IF
`define VX_CACHE_MEM_REQ_IF

`include "../cache/VX_cache_define.vh"

interface VX_cache_mem_req_if #(
    parameter MEM_LINE_WIDTH = 1,
    parameter MEM_ADDR_WIDTH = 1,
    parameter MEM_TAG_WIDTH  = 1,
    parameter MEM_LINE_SIZE  = MEM_LINE_WIDTH / 8
) ();

    wire                        valid;    
    wire                        rw;    
    wire [MEM_LINE_SIZE-1:0]    byteen;
    wire [MEM_ADDR_WIDTH-1:0]   addr;
    wire [MEM_LINE_WIDTH-1:0]   data;  
    wire [MEM_TAG_WIDTH-1:0]    tag;  
    wire                        ready;

endinterface

`endif