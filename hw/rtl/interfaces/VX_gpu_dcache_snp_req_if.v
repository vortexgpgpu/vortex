`ifndef VX_GPU_SNP_REQ
`define VX_GPU_SNP_REQ

`include "../cache/VX_cache_config.vh"

interface VX_gpu_dcache_snp_req_if ();
    // Snoop Req
    wire            snp_req_valid;
    wire [31:0]     snp_req_addr;

endinterface

`endif