`ifndef VX_GPU_SNP_REQ_RSP
`define VX_GPU_SNP_REQ_RSP

`include "../cache/VX_cache_config.vh"

interface VX_gpu_snp_req_rsp_if ();

	// Snoop request
	wire        snp_req_valid;
	wire [31:0] snp_req_addr;	
	wire        snp_req_ready;

	// Snoop Response
	// TODO:

endinterface

`endif