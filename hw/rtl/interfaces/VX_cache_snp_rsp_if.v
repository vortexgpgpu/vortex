`ifndef VX_CACHE_SNP_RSP_IF
`define VX_CACHE_SNP_RSP_IF

`include "../cache/VX_cache_config.vh"

interface VX_cache_snp_rsp_if #(
    parameter SNP_TAG_WIDTH = 0
) ();

    wire                     valid;
    
    wire [SNP_TAG_WIDTH-1:0] tag;    

    wire                     ready;

endinterface

`endif