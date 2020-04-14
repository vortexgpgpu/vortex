`include "../VX_cache/VX_cache_config.v"

`ifndef VX_GPU_SNP_REQ_RSP

`define VX_GPU_SNP_REQ_RSP

interface VX_gpu_snp_req_rsp
	();

	// Snoop request
	wire        snp_req;
	wire[31:0]  snp_req_addr;

	// Snoop Response
	wire        snp_delay;

endinterface


`endif