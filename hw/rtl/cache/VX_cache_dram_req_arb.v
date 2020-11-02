`include "VX_cache_config.vh"

module VX_cache_dram_req_arb #(
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE                = 0, 
    // Number of banks {1, 2, 4, 8,...}
    parameter NUM_BANKS                     = 0, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 0
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

    wire [`BANK_BITS-1:0] sel_bank;
    wire sel_valid;
    
    VX_fixed_arbiter #(
        .N(NUM_BANKS)
    ) sel_arb (
        .clk         (clk),
        .reset       (reset),
        .requests    (per_bank_dram_req_valid),
        .grant_index (sel_bank),
        .grant_valid (sel_valid),
        `UNUSED_PIN  (grant_onehot)
    );    

    assign dram_req_valid  = sel_valid;  
    assign dram_req_rw     = per_bank_dram_req_rw[sel_bank];    
    assign dram_req_byteen = per_bank_dram_req_byteen[sel_bank];
    assign dram_req_addr   = per_bank_dram_req_addr[sel_bank];
    assign dram_req_data   = per_bank_dram_req_data[sel_bank];
    
    for (genvar i = 0; i < NUM_BANKS; i++) begin
        assign per_bank_dram_req_ready[i] = dram_req_ready && (sel_bank == `BANK_BITS'(i));
    end

endmodule
