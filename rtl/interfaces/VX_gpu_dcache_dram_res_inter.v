


`include "../VX_cache/VX_cache_config.v"

`ifndef VX_GPU_DRAM_DCACHE_RES

`define VX_GPU_DRAM_DCACHE_RES

interface VX_gpu_dcache_dram_res_inter ();
	// DRAM Rsponse
    wire                              dram_fill_rsp;
    wire [31:0]                       dram_fill_rsp_addr;
    wire [`BANK_LINE_SIZE_RNG][31:0]  dram_fill_rsp_data;

endinterface


`endif