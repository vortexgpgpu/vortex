`ifndef VX_GPU_DRAM_DCACHE_RSP
`define VX_GPU_DRAM_DCACHE_RSP

`include "../generic_cache/VX_cache_config.vh"

interface VX_gpu_dcache_dram_rsp_inter #(
	parameter BANK_LINE_WORDS = 2
) ();
	// DRAM Response
    wire                             dram_rsp_valid;
    wire [31:0]                      dram_rsp_addr;
    wire [BANK_LINE_WORDS-1:0][31:0] dram_rsp_data;

endinterface

`endif