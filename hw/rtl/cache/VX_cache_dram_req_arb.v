`include "VX_cache_config.vh"

module VX_cache_dram_req_arb #(
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE                = 0, 
    // Number of banks {1, 2, 4, 8,...}
    parameter NUM_BANKS                     = 0, 
    // Size of a word in bytes
    parameter WORD_SIZE                     = 0, 
    // Dram Fill Req Queue Size
    parameter DFQQ_SIZE                     = 0
) (
    input  wire                                 clk,
    input  wire                                 reset,

    // Fill Request    
    input  wire [NUM_BANKS-1:0]                 per_bank_dram_fill_req_valid,
    input  wire [NUM_BANKS-1:0][`DRAM_ADDR_WIDTH-1:0] per_bank_dram_fill_req_addr,
    output wire                                 dram_fill_req_ready,
    
    // Writeback Request    
    input  wire [NUM_BANKS-1:0]                 per_bank_dram_wb_req_valid,
    input  wire [NUM_BANKS-1:0][BANK_LINE_SIZE-1:0] per_bank_dram_wb_req_byteen,
    input  wire [NUM_BANKS-1:0][`DRAM_ADDR_WIDTH-1:0] per_bank_dram_wb_req_addr,
    input  wire [NUM_BANKS-1:0][`BANK_LINE_WIDTH-1:0] per_bank_dram_wb_req_data,
    output wire [NUM_BANKS-1:0]                 per_bank_dram_wb_req_ready,
    
    // Merged Request
    output wire                                 dram_req_valid,
    output wire                                 dram_req_rw,    
    output wire [BANK_LINE_SIZE-1:0]            dram_req_byteen,
    output wire [`DRAM_ADDR_WIDTH-1:0]          dram_req_addr,
    output wire [`BANK_LINE_WIDTH-1:0]          dram_req_data,

    input wire                                  dram_req_ready
);

    wire dwb_valid;
    wire dfqq_req;
    
    wire[`DRAM_ADDR_WIDTH-1:0] dfqq_req_addr;
    
`DEBUG_BEGIN
    wire dfqq_empty;    
`DEBUG_END

    wire dfqq_pop  = !dwb_valid && dfqq_req && dram_req_ready; // If no dwb, and dfqq has valids, then pop
    wire dfqq_push = (| per_bank_dram_fill_req_valid);
    wire dfqq_full;

    VX_cache_dram_fill_arb #(
        .BANK_LINE_SIZE(BANK_LINE_SIZE),
        .NUM_BANKS(NUM_BANKS),
        .DFQQ_SIZE(DFQQ_SIZE)
    ) dram_fill_arb (
        .clk                            (clk),
        .reset                          (reset),
        .dfqq_push                      (dfqq_push),
        .per_bank_dram_fill_req_valid   (per_bank_dram_fill_req_valid),
        .per_bank_dram_fill_req_addr    (per_bank_dram_fill_req_addr),
        .dfqq_pop                       (dfqq_pop),
        .dfqq_req                       (dfqq_req),
        .dfqq_req_addr                  (dfqq_req_addr),
        .dfqq_empty                     (dfqq_empty),
        .dfqq_full                      (dfqq_full)
    );

    assign dram_fill_req_ready = !dfqq_full;

    wire [`BANK_BITS-1:0] dwb_bank;
    
    VX_fixed_arbiter #(
        .N(NUM_BANKS)
    ) sel_dwb (
        .clk         (clk),
        .reset       (reset),
        .requests    (per_bank_dram_wb_req_valid),
        .grant_index (dwb_bank),
        .grant_valid (dwb_valid),
        `UNUSED_PIN  (grant_onehot)
    );    

    for (genvar i = 0; i < NUM_BANKS; i++) begin
        assign per_bank_dram_wb_req_ready[i] = dram_req_ready && (dwb_bank == `BANK_BITS'(i));
    end

    assign dram_req_valid  = dwb_valid || dfqq_req;    
    assign dram_req_rw     = dwb_valid;    
    assign dram_req_byteen = dwb_valid ? per_bank_dram_wb_req_byteen[dwb_bank] : {BANK_LINE_SIZE{1'b1}};    
    assign dram_req_addr   = dwb_valid ? per_bank_dram_wb_req_addr[dwb_bank] : dfqq_req_addr;
    assign {dram_req_data} = dwb_valid ? per_bank_dram_wb_req_data[dwb_bank] : 0;

endmodule
