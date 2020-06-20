`include "VX_cache_config.vh"

module VX_cache_dram_fill_arb #(
    // Size of line inside a bank in bytes
    parameter BANK_LINE_SIZE                = 0, 
    // Number of banks {1, 2, 4, 8,...}
    parameter NUM_BANKS                     = 0, 
    // Dram Fill Req Queue Size
    parameter DFQQ_SIZE                     = 0
) (
    input  wire                         clk,
    input  wire                         reset,
    input  wire                         dfqq_push,
    input  wire[NUM_BANKS-1:0]          per_bank_dram_fill_req_valid,
    input  wire[NUM_BANKS-1:0][`DRAM_ADDR_WIDTH-1:0] per_bank_dram_fill_req_addr,

    input  wire                         dfqq_pop,
    output wire                         dfqq_req,
    output wire[`DRAM_ADDR_WIDTH-1:0]   dfqq_req_addr,
    output wire                         dfqq_empty,
    output wire                         dfqq_full
);
    reg [NUM_BANKS-1:0] use_per_bank_dram_fill_req_valid;
    reg [NUM_BANKS-1:0][`DRAM_ADDR_WIDTH-1:0] use_per_bank_dram_fill_req_addr;

    wire [NUM_BANKS-1:0] out_per_bank_dram_fill_req_valid;
    wire [NUM_BANKS-1:0][`DRAM_ADDR_WIDTH-1:0] out_per_bank_dram_fill_req_addr;

    wire [NUM_BANKS-1:0] use_per_bqual_bank_dram_fill_req_valid;
    wire [NUM_BANKS-1:0][`DRAM_ADDR_WIDTH-1:0] qual_bank_dram_fill_req_addr;

    wire [NUM_BANKS-1:0] updated_bank_dram_fill_req_valid;

    wire o_empty;

    wire use_empty = !(| use_per_bank_dram_fill_req_valid);
    wire out_empty = !(| out_per_bank_dram_fill_req_valid) || o_empty;

    wire push_qual = dfqq_push && !dfqq_full;
    wire pop_qual  = dfqq_pop && use_empty && !out_empty;

    VX_generic_queue #(
        .DATAW(NUM_BANKS * (1+`DRAM_ADDR_WIDTH)), 
        .SIZE(DFQQ_SIZE)
    ) dfqq_queue (
        .clk     (clk),
        .reset   (reset),
        .push    (push_qual),
        .data_in ({per_bank_dram_fill_req_valid, per_bank_dram_fill_req_addr}),
        .pop     (pop_qual),
        .data_out({out_per_bank_dram_fill_req_valid, out_per_bank_dram_fill_req_addr}),
        .empty   (o_empty),
        .full    (dfqq_full),
        `UNUSED_PIN (size)
    );

    assign use_per_bqual_bank_dram_fill_req_valid = use_empty ? (out_per_bank_dram_fill_req_valid & {NUM_BANKS{!o_empty}}) : (use_per_bank_dram_fill_req_valid & {NUM_BANKS{!use_empty}}); 
    assign qual_bank_dram_fill_req_addr = use_empty ? out_per_bank_dram_fill_req_addr : use_per_bank_dram_fill_req_addr;

    wire[`BANK_BITS-1:0] qual_request_index;
    wire                 qual_has_request;

    VX_fixed_arbiter #(
        .N(NUM_BANKS)
    ) sel_bank (
        .clk         (clk),
        .reset       (reset),
        .requests    (use_per_bqual_bank_dram_fill_req_valid),
        .grant_index (qual_request_index),
        .grant_valid (qual_has_request),
        `UNUSED_PIN  (grant_onehot)
    );

    assign dfqq_empty    = !qual_has_request;
    assign dfqq_req      = use_per_bqual_bank_dram_fill_req_valid [qual_request_index];
    assign dfqq_req_addr = qual_bank_dram_fill_req_addr[qual_request_index];

    assign updated_bank_dram_fill_req_valid = use_per_bqual_bank_dram_fill_req_valid & (~(1 << qual_request_index));

    always @(posedge clk) begin
        if (reset) begin
            use_per_bank_dram_fill_req_valid <= 0;
            use_per_bank_dram_fill_req_addr  <= 0;
        end else begin
            if (dfqq_pop && qual_has_request) begin
                use_per_bank_dram_fill_req_valid <= updated_bank_dram_fill_req_valid;
                use_per_bank_dram_fill_req_addr  <= qual_bank_dram_fill_req_addr;
            end
        end
    end

endmodule