`include "VX_define.vh"

module VX_tex_lsu_arb #(    
    parameter NUM_REQS      = 1, 
    parameter LANES         = 1,
    parameter WORD_SIZE     = 1,
    parameter TAG_IN_WIDTH  = 1,
    parameter TAG_OUT_WIDTH = 1,
    parameter LOG_NUM_REQS  = `CLOG2(NUM_REQS)
) (
    input wire clk,
    input wire reset,

    // input requests    
    input wire [NUM_REQS-1:0][LANES-1:0]                    req_valid_in, 
    input wire [NUM_REQS-1:0][LANES-1:0]                    req_rw_in,   
    input wire [NUM_REQS-1:0][LANES-1:0][WORD_SIZE-1:0]     req_byteen_in, 
    input wire [NUM_REQS-1:0][LANES-1:0][`WORD_ADDR_WIDTH-1:0] req_addr_in, 
    input wire [NUM_REQS-1:0][LANES-1:0][`WORD_WIDTH-1:0]   req_data_in,    
    input wire [NUM_REQS-1:0][LANES-1:0][TAG_IN_WIDTH-1:0]  req_tag_in,  
    output wire [NUM_REQS-1:0][LANES-1:0]                   req_ready_in,

    // output request
    output wire [LANES-1:0]                                 req_valid_out,
    output wire [LANES-1:0]                                 req_rw_out,  
    output wire [LANES-1:0][WORD_SIZE-1:0]                  req_byteen_out,  
    output wire [LANES-1:0][`WORD_ADDR_WIDTH-1:0]           req_addr_out, 
    output wire [LANES-1:0][`WORD_WIDTH-1:0]                req_data_out,   
    output wire [LANES-1:0][TAG_OUT_WIDTH-1:0]              req_tag_out,    
    input wire  [LANES-1:0]                                 req_ready_out,

    // input response
    input wire [LANES-1:0]                                  rsp_valid_in,    
    input wire [LANES-1:0][`WORD_WIDTH-1:0]                 rsp_data_in,
    input wire [TAG_OUT_WIDTH-1:0]                          rsp_tag_in,
    output wire                                             rsp_ready_in,

    // output responses
    output wire [NUM_REQS-1:0][LANES-1:0]                   rsp_valid_out,
    output wire [NUM_REQS-1:0][LANES-1:0][`WORD_WIDTH-1:0]  rsp_data_out,
    output wire [NUM_REQS-1:0][TAG_IN_WIDTH-1:0]            rsp_tag_out,
    input wire  [NUM_REQS-1:0]                              rsp_ready_out    
);  
    localparam REQ_DATAW = LANES * (1 + TAG_IN_WIDTH + `WORD_ADDR_WIDTH + 1 + WORD_SIZE + `WORD_WIDTH);

    if (NUM_REQS > 1) begin

        wire [NUM_REQS-1:0][REQ_DATAW-1:0] req_merged_data_in;
        wire [NUM_REQS-1:0] req_valid_in_any;

        for (genvar i = 0; i < NUM_REQS; i++) begin
            assign req_merged_data_in[i] = {req_valid_in[i], req_tag_in[i], req_addr_in[i], req_rw_in[i], req_byteen_in[i], req_data_in[i]};
            assign req_valid_in_any[i] = (| req_valid_in[i]);
        end

        wire                    sel_valid;
        wire [LOG_NUM_REQS-1:0] sel_idx;
        wire [NUM_REQS-1:0]     sel_1hot;

        wire sel_enable = (| req_ready_out);

        VX_rr_arbiter #(
            .NUM_REQS(NUM_REQS),
            .LOCK_ENABLE(1)
        ) sel_arb (
            .clk          (clk),
            .reset        (reset),
            .requests     (req_valid_in_any),
            .enable       (sel_enable),      
            .grant_valid  (sel_valid),
            .grant_index  (sel_idx),
            .grant_onehot (sel_1hot)
        );

        wire [LANES-1:0] req_valid_out_unqual;
        wire [LANES-1:0][TAG_IN_WIDTH-1:0] req_tag_out_unqual;

        assign {req_valid_out_unqual, req_tag_out_unqual, req_addr_out, req_rw_out, req_byteen_out, req_data_out} = req_merged_data_in[sel_idx];

        assign req_valid_out = req_valid_out_unqual & {LANES{sel_valid}};

        for (genvar i = 0; i < LANES; i++) begin
            assign req_tag_out[i] = {req_tag_out_unqual[i], sel_idx};
        end

        for (genvar i = 0; i < NUM_REQS; i++) begin
            assign req_ready_in[i] = req_ready_out & {LANES{sel_1hot[i]}};
        end

        ///////////////////////////////////////////////////////////////////////

        wire [LOG_NUM_REQS-1:0] rsp_sel = rsp_tag_in[LOG_NUM_REQS-1:0];

        reg [NUM_REQS-1:0][LANES-1:0] rsp_valid_out_unqual;
        always @(*) begin
            rsp_valid_out_unqual = '0;
            rsp_valid_out_unqual[rsp_sel] = rsp_valid_in;
        end
        assign rsp_valid_out = rsp_valid_out_unqual;

        for (genvar i = 0; i < NUM_REQS; i++) begin
            assign rsp_data_out[i] = rsp_data_in;
            assign rsp_tag_out[i]  = rsp_tag_in[LOG_NUM_REQS +: TAG_IN_WIDTH];            
        end

        assign rsp_ready_in = rsp_ready_out[rsp_sel];

    end else begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        assign req_valid_out  = req_valid_in;
        assign req_tag_out    = req_tag_in;
        assign req_addr_out   = req_addr_in;
        assign req_rw_out     = req_rw_in;
        assign req_byteen_out = req_byteen_in;
        assign req_data_out   = req_data_in;
        assign req_ready_in   = req_ready_out;

        assign rsp_valid_out  = rsp_valid_in;
        assign rsp_tag_out    = rsp_tag_in;
        assign rsp_data_out   = rsp_data_in;
        assign rsp_ready_in   = rsp_ready_out;

    end

endmodule