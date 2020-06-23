`ifndef VX_CACHE_DRAM_REQ_IF
`define VX_CACHE_DRAM_REQ_IF

`include "../cache/VX_cache_config.vh"

interface VX_cache_dram_req_if #(
    parameter DRAM_LINE_WIDTH = 1,
    parameter DRAM_ADDR_WIDTH = 1,
    parameter DRAM_TAG_WIDTH  = 1
) ();

    wire                        dram_req_valid;
    wire                        dram_req_rw;    
    wire [(DRAM_LINE_WIDTH/8)-1:0] dram_req_byteen;
    wire [DRAM_ADDR_WIDTH-1:0]  dram_req_addr;
    wire [DRAM_LINE_WIDTH-1:0]  dram_req_data;  
    wire [DRAM_TAG_WIDTH-1:0]   dram_req_tag;    
    wire                        dram_req_ready;

endinterface

`endif