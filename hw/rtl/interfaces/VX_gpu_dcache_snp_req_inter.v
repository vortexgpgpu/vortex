


`include "../VX_cache/VX_cache_config.v"

`ifndef VX_GPU_SNP_REQ

`define VX_GPU_SNP_REQ

interface VX_gpu_dcache_snp_req_inter ();
	// Snoop Req
    wire                              snp_req;
    wire [31:0]                       snp_req_addr;

endinterface


`endif