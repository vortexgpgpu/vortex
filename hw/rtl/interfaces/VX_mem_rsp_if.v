`ifndef VX_MEM_RSP_IF
`define VX_MEM_RSP_IF

`include "../cache/VX_cache_define.vh"

interface VX_mem_rsp_if #(
    parameter LINE_WIDTH = 1,
    parameter TAG_WIDTH  = 1
) ();

    wire                    valid;    
    wire [LINE_WIDTH-1:0]   data;
    wire [TAG_WIDTH-1:0]    tag;  
    wire                    ready;      

endinterface

`endif