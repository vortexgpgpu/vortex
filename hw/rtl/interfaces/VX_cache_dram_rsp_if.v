`ifndef VX_CACHE_DRAM_RSP_IF
`define VX_CACHE_DRAM_RSP_IF

`include "../cache/VX_cache_config.vh"

interface VX_cache_dram_rsp_if #(
    parameter BANK_LINE_WORDS = 2
) ();
    // DRAM Response
    wire                             dram_rsp_valid;
    wire [31:0]                      dram_rsp_addr;
    wire [BANK_LINE_WORDS-1:0][31:0] dram_rsp_data;

endinterface

`endif