`include "VX_cache_config.vh"

module VX_cache_dram_req_arb #(
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE     = 1, 
    // Number of banks
    parameter NUM_BANKS          = 1, 
    // Size of a word in bytes
    parameter WORD_SIZE          = 1
) (
    input  wire                                 clk,
    input  wire                                 reset,
    
    // Inputs
    input  wire [NUM_BANKS-1:0]                 per_bank_dram_req_valid,
    input  wire [NUM_BANKS-1:0]                 per_bank_dram_req_rw,
    input  wire [NUM_BANKS-1:0][BANK_LINE_SIZE-1:0] per_bank_dram_req_byteen,
    input  wire [NUM_BANKS-1:0][`DRAM_ADDR_WIDTH-1:0] per_bank_dram_req_addr,
    input  wire [NUM_BANKS-1:0][`BANK_LINE_WIDTH-1:0] per_bank_dram_req_data,
    output wire [NUM_BANKS-1:0]                 per_bank_dram_req_ready,
    
    // Output
    output wire                                 dram_req_valid,
    output wire                                 dram_req_rw,    
    output wire [BANK_LINE_SIZE-1:0]            dram_req_byteen,
    output wire [`DRAM_ADDR_WIDTH-1:0]          dram_req_addr,
    output wire [`BANK_LINE_WIDTH-1:0]          dram_req_data,
    input wire                                  dram_req_ready
);

    wire sel_valid;
    wire [`BANK_BITS-1:0] sel_idx;
    wire [NUM_BANKS-1:0] sel_1hot;
    
    VX_fixed_arbiter #(
        .N(NUM_BANKS)
    ) sel_arb (
        .clk         (clk),
        .reset       (reset),
        .requests    (per_bank_dram_req_valid),      
        .grant_valid (sel_valid),
        .grant_index (sel_idx),
        .grant_onehot(sel_1hot)
    );    

    wire stall = ~dram_req_ready && dram_req_valid;

    VX_generic_register #(
        .N(1 + 1 + BANK_LINE_SIZE + `DRAM_ADDR_WIDTH + `BANK_LINE_WIDTH)
    ) core_wb_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (1'b0),
        .in    ({sel_valid,      per_bank_dram_req_rw[sel_idx], per_bank_dram_req_byteen[sel_idx], per_bank_dram_req_addr[sel_idx], per_bank_dram_req_data[sel_idx]}),
        .out   ({dram_req_valid, dram_req_rw,                   dram_req_byteen,                   dram_req_addr,                   dram_req_data})
    );

    for (genvar i = 0; i < NUM_BANKS; i++) begin
        assign per_bank_dram_req_ready[i] = sel_1hot[i] && !stall;
    end

endmodule
