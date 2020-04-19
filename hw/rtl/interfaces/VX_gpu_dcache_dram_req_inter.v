`ifndef VX_GPU_DRAM_DCACHE_REQ
`define VX_GPU_DRAM_DCACHE_REQ

`include "../generic_cache/VX_cache_config.vh"

interface VX_gpu_dcache_dram_req_inter #(
    parameter BANK_LINE_WORDS = 2
) ();

	// DRAM Request
    wire                              dram_req_write;
    wire                              dram_req_read;
    wire [31:0]                       dram_req_addr;
    wire [BANK_LINE_WORDS-1:0][31:0]  dram_req_data;    
    wire                              dram_req_full;

    wire                              dram_rsp_ready;    

endinterface

`endif