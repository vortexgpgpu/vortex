`ifndef VX_CACHE_SNP_RSP_IF
`define VX_CACHE_SNP_RSP_IF

`include "../cache/VX_cache_config.vh"

interface VX_cache_snp_rsp_if #(
    parameter SNP_TAG_WIDTH = 0
) ();

    wire                        snp_rsp_valid;
    wire [SNP_TAG_WIDTH-1:0]    snp_rsp_tag;    
    wire                        snp_rsp_ready;

endinterface

`endif