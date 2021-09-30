`include "VX_define.vh"

module VX_smem_arb #(
    parameter NUM_REQS      = 1, 
    parameter LANES         = 1,
    parameter DATA_SIZE     = 1,
    parameter TAG_IN_WIDTH  = 1,
    parameter TAG_SEL_IDX   = 0,   
    parameter BUFFERED_REQ  = 0,
    parameter BUFFERED_RSP  = 0,
    parameter TYPE          = "P",

    parameter ADDR_WIDTH    = (32-`CLOG2(DATA_SIZE)),
    parameter DATA_WIDTH    = (8 * DATA_SIZE),
    parameter LOG_NUM_REQS  = `CLOG2(NUM_REQS),
    parameter TAG_OUT_WIDTH = TAG_IN_WIDTH - LOG_NUM_REQS
) (
    input wire clk,
    input wire reset,

    // input request
    input wire [LANES-1:0]                                  req_valid_in,
    input wire [LANES-1:0]                                  req_rw_in,  
    input wire [LANES-1:0][DATA_SIZE-1:0]                   req_byteen_in,  
    input wire [LANES-1:0][ADDR_WIDTH-1:0]                  req_addr_in, 
    input wire [LANES-1:0][DATA_WIDTH-1:0]                  req_data_in,   
    input wire [LANES-1:0][TAG_IN_WIDTH-1:0]                req_tag_in,    
    output wire  [LANES-1:0]                                req_ready_in,

    // output requests    
    output wire [NUM_REQS-1:0][LANES-1:0]                   req_valid_out, 
    output wire [NUM_REQS-1:0][LANES-1:0]                   req_rw_out,   
    output wire [NUM_REQS-1:0][LANES-1:0][DATA_SIZE-1:0]    req_byteen_out, 
    output wire [NUM_REQS-1:0][LANES-1:0][ADDR_WIDTH-1:0]   req_addr_out, 
    output wire [NUM_REQS-1:0][LANES-1:0][DATA_WIDTH-1:0]   req_data_out,    
    output wire [NUM_REQS-1:0][LANES-1:0][TAG_OUT_WIDTH-1:0] req_tag_out,  
    input wire [NUM_REQS-1:0][LANES-1:0]                    req_ready_out,

    // input responses
    input wire [NUM_REQS-1:0]                               rsp_valid_in,
    input wire [NUM_REQS-1:0][LANES-1:0]                    rsp_tmask_in,
    input wire [NUM_REQS-1:0][LANES-1:0][DATA_WIDTH-1:0]    rsp_data_in,
    input wire [NUM_REQS-1:0][TAG_OUT_WIDTH-1:0]            rsp_tag_in,
    output wire  [NUM_REQS-1:0]                             rsp_ready_in,    

    // output response
    output wire                                             rsp_valid_out,    
    output wire [LANES-1:0]                                 rsp_tmask_out,    
    output wire [LANES-1:0][DATA_WIDTH-1:0]                 rsp_data_out,
    output wire [TAG_IN_WIDTH-1:0]                          rsp_tag_out,
    input wire                                              rsp_ready_out
);  
    localparam REQ_DATAW = TAG_OUT_WIDTH + ADDR_WIDTH + 1 + DATA_SIZE + DATA_WIDTH;
    localparam RSP_DATAW = LANES * (1 + DATA_WIDTH) + TAG_IN_WIDTH;

    if (NUM_REQS > 1) begin

        wire [LANES-1:0][REQ_DATAW-1:0] req_data_in_merged;
        wire [NUM_REQS-1:0][LANES-1:0][REQ_DATAW-1:0] req_data_out_merged;

        wire [LANES-1:0][LOG_NUM_REQS-1:0] req_sel;
        wire [LANES-1:0][TAG_OUT_WIDTH-1:0] req_tag_in_w;

        for (genvar i = 0; i < LANES; ++i) begin
            assign req_sel[i] = req_tag_in[i][TAG_SEL_IDX +: LOG_NUM_REQS];            
            
            VX_bits_remove #( 
                .N   (TAG_IN_WIDTH),
                .S   (LOG_NUM_REQS),
                .POS (TAG_SEL_IDX)
            ) bits_remove (
                .data_in  (req_tag_in[i]),
                .data_out (req_tag_in_w[i])
            );

            assign req_data_in_merged[i] = {req_tag_in_w[i], req_addr_in[i], req_rw_in[i], req_byteen_in[i], req_data_in[i]};
        end

        VX_stream_demux #(
            .NUM_REQS (NUM_REQS),
            .LANES    (LANES),
            .DATAW    (REQ_DATAW),
            .BUFFERED (BUFFERED_REQ)
        ) req_demux (
            .clk       (clk),
            .reset     (reset),
            .sel_in    (req_sel),
            .valid_in  (req_valid_in),
            .data_in   (req_data_in_merged),
            .ready_in  (req_ready_in),
            .valid_out (req_valid_out),
            .data_out  (req_data_out_merged),
            .ready_out (req_ready_out)
        );
        
        for (genvar i = 0; i < NUM_REQS; i++) begin
            for (genvar j = 0; j < LANES; ++j) begin
                assign {req_tag_out[i][j], req_addr_out[i][j], req_rw_out[i][j], req_byteen_out[i][j], req_data_out[i][j]} = req_data_out_merged[i][j];
            end
        end

        ///////////////////////////////////////////////////////////////////////        

        wire [NUM_REQS-1:0][RSP_DATAW-1:0] rsp_data_in_merged;
        
        for (genvar i = 0; i < NUM_REQS; i++) begin
            wire [TAG_IN_WIDTH-1:0] rsp_tag_in_w;
            
            VX_bits_insert #( 
                .N   (TAG_OUT_WIDTH),
                .S   (LOG_NUM_REQS),
                .POS (TAG_SEL_IDX)
            ) bits_insert (
                .data_in  (rsp_tag_in[i]),
                .sel_in   (LOG_NUM_REQS'(i)),
                .data_out (rsp_tag_in_w)
            );

            assign rsp_data_in_merged[i] = {rsp_tag_in_w, rsp_tmask_in[i], rsp_data_in[i]};
        end

        VX_stream_arbiter #(            
            .NUM_REQS (NUM_REQS),
            .LANES    (1),
            .DATAW    (RSP_DATAW),
            .BUFFERED (BUFFERED_RSP),
            .TYPE     (TYPE)
        ) rsp_arb (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (rsp_valid_in),
            .data_in   (rsp_data_in_merged),
            .ready_in  (rsp_ready_in),
            .valid_out (rsp_valid_out),
            .data_out  ({rsp_tag_out, rsp_tmask_out, rsp_data_out}),
            .ready_out (rsp_ready_out)
        );

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
        assign rsp_tmask_out  = rsp_tmask_in;
        assign rsp_tag_out    = rsp_tag_in;
        assign rsp_data_out   = rsp_data_in;
        assign rsp_ready_in   = rsp_ready_out;

    end

endmodule