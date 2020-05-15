`include "VX_cache_config.vh"

module VX_snp_rsp_arb #(
    parameter NUM_BANKS         = 0,
    parameter BANK_LINE_SIZE    = 0,
    parameter SNP_REQ_TAG_WIDTH = 0
) (
    input  wire [NUM_BANKS-1:0]         per_bank_snp_rsp_valid,
    input  wire [NUM_BANKS-1:0][SNP_REQ_TAG_WIDTH-1:0] per_bank_snp_rsp_tag,
    output wire [NUM_BANKS-1:0]         per_bank_snp_rsp_ready,

    output wire                         snp_rsp_valid,
    output wire [SNP_REQ_TAG_WIDTH-1:0] snp_rsp_tag,
    input  wire                         snp_rsp_ready    
);

    wire [`BANK_BITS-1:0] fsq_bank;
    wire                  fsq_valid;

    VX_generic_priority_encoder #(
        .N(NUM_BANKS)
    ) sel_ffsq (
        .valids (per_bank_snp_rsp_valid),
        .index  (fsq_bank),
        .found  (fsq_valid)
    );

    assign snp_rsp_valid = fsq_valid;
    assign snp_rsp_tag   = per_bank_snp_rsp_tag[fsq_bank];

    genvar i;
    for (i = 0; i < NUM_BANKS; i++) begin
        assign per_bank_snp_rsp_ready[i] = snp_rsp_ready && (fsq_bank == `BANK_BITS'(i));
    end

endmodule