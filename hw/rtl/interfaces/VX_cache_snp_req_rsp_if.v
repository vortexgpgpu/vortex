`ifndef VX_CACHE_SNP_REQ_RSP_IF
`define VX_CACHE_SNP_REQ_RSP_IF

`include "../cache/VX_cache_config.vh"

interface VX_cache_snp_req_rsp_if ();

    // Snoop request
    wire        snp_req_valid;
    wire [31:0] snp_req_addr;    
    wire        snp_req_ready;

    // Snoop Response
    // TODO:

endinterface

`endif