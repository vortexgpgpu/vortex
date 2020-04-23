`ifndef VX_CACHE_DRAM_REQ_IF
`define VX_CACHE_DRAM_REQ_IF

`include "../cache/VX_cache_config.vh"

interface VX_cache_dram_req_if #(
    parameter BANK_LINE_WORDS = 2
) ();

    // DRAM Request
    wire                              dram_req_write;
    wire                              dram_req_read;
    wire [31:0]                       dram_req_addr;
    wire [BANK_LINE_WORDS-1:0][31:0]  dram_req_data;    
    wire                              dram_req_ready;

    wire                              dram_rsp_ready;    

endinterface

`endif