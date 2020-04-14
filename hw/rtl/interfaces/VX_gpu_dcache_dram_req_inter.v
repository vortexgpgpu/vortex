

`include "../VX_cache/VX_cache_config.v"

`ifndef VX_GPU_DRAM_DCACHE_REQ

`define VX_GPU_DRAM_DCACHE_REQ

interface VX_gpu_dcache_dram_req_inter
    #(
        parameter BANK_LINE_SIZE_WORDS = 2
    )
    ();

	// DRAM Request
    wire                              dram_req;
    wire                              dram_req_write;
    wire                              dram_req_read;
    wire [31:0]                       dram_req_addr;
    wire [31:0]                       dram_req_size;
    wire [BANK_LINE_SIZE_WORDS-1:0][31:0]  dram_req_data;

    // Snoop
    wire                              dram_because_of_snp;
    wire                              dram_snp_full;

    // DRAM Cache can't accept response
    wire                              dram_fill_accept;


    // DRAM Cache can't accept request
    wire                              dram_req_delay;

endinterface


`endif