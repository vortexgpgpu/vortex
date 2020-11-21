`ifndef VX_CACHE_SNP_REQ_IF
`define VX_CACHE_SNP_REQ_IF

`include "../cache/VX_cache_config.vh"

interface VX_cache_snp_req_if #(
    parameter DRAM_ADDR_WIDTH = 0,
    parameter SNP_TAG_WIDTH   = 0
) ();

    wire                        valid;
    
    wire [DRAM_ADDR_WIDTH-1:0]  addr; 
    wire                        invalidate;   
    wire [SNP_TAG_WIDTH-1:0]    tag;   

    wire                        ready;

endinterface

`endif