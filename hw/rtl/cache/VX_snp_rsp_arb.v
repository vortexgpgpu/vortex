`include "VX_cache_config.vh"

module VX_snp_rsp_arb #(
    parameter NUM_BANKS         = 1,
    parameter BANK_LINE_SIZE    = 1,
    parameter SNP_REQ_TAG_WIDTH = 1
) (
    input  wire clk,
    input  wire reset,
    
    input  wire [NUM_BANKS-1:0]         per_bank_snp_rsp_valid,
    input  wire [NUM_BANKS-1:0][SNP_REQ_TAG_WIDTH-1:0] per_bank_snp_rsp_tag,
    output wire [NUM_BANKS-1:0]         per_bank_snp_rsp_ready,

    output wire                         snp_rsp_valid,
    output wire [SNP_REQ_TAG_WIDTH-1:0] snp_rsp_tag,
    input  wire                         snp_rsp_ready    
);

    wire sel_valid;
    wire [`BANK_BITS-1:0] sel_idx;
    wire [NUM_BANKS-1:0] sel_1hot;

    VX_fixed_arbiter #(
        .N(NUM_BANKS)
    ) sel_arb (
        .clk         (clk),
        .reset       (reset),
        .requests    (per_bank_snp_rsp_valid),        
        .grant_valid (sel_valid),
        .grant_index (sel_idx),
        .grant_onehot(sel_1hot)
    );

    wire stall = ~snp_rsp_ready && snp_rsp_valid;

    VX_generic_register #(
        .N(1 + SNP_REQ_TAG_WIDTH)
    ) core_wb_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (1'b0),
        .in    ({sel_valid,     per_bank_snp_rsp_tag[sel_idx]}),
        .out   ({snp_rsp_valid, snp_rsp_tag})
    );

    for (genvar i = 0; i < NUM_BANKS; i++) begin
        assign per_bank_snp_rsp_ready[i] = sel_1hot[i] && !stall;
    end

endmodule