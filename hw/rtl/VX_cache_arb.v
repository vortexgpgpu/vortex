`include "VX_define.vh"

module VX_cache_arb #(    
    parameter NUM_REQS      = 1, 
    parameter LANES         = 1,
    parameter DATA_SIZE     = 1,
    parameter TAG_IN_WIDTH  = 1,
    parameter TAG_SEL_IDX   = 0,   
    parameter BUFFERED_REQ  = 0,
    parameter BUFFERED_RSP  = 0,
    parameter TYPE          = "R",

    localparam ADDR_WIDTH   = (32-`CLOG2(DATA_SIZE)),
    localparam DATA_WIDTH   = (8 * DATA_SIZE),
    localparam LOG_NUM_REQS = `CLOG2(NUM_REQS),
    localparam TAG_OUT_WIDTH = TAG_IN_WIDTH + LOG_NUM_REQS
) (
    input wire clk,
    input wire reset,

    // input requests    
    input wire [NUM_REQS-1:0][LANES-1:0]                    req_valid_in, 
    input wire [NUM_REQS-1:0][LANES-1:0]                    req_rw_in,   
    input wire [NUM_REQS-1:0][LANES-1:0][DATA_SIZE-1:0]     req_byteen_in, 
    input wire [NUM_REQS-1:0][LANES-1:0][ADDR_WIDTH-1:0]    req_addr_in, 
    input wire [NUM_REQS-1:0][LANES-1:0][DATA_WIDTH-1:0]    req_data_in,    
    input wire [NUM_REQS-1:0][LANES-1:0][TAG_IN_WIDTH-1:0]  req_tag_in,  
    output wire [NUM_REQS-1:0][LANES-1:0]                   req_ready_in,

    // output request
    output wire [LANES-1:0]                                 req_valid_out,
    output wire [LANES-1:0]                                 req_rw_out,  
    output wire [LANES-1:0][DATA_SIZE-1:0]                  req_byteen_out,  
    output wire [LANES-1:0][ADDR_WIDTH-1:0]                 req_addr_out, 
    output wire [LANES-1:0][DATA_WIDTH-1:0]                 req_data_out,   
    output wire [LANES-1:0][TAG_OUT_WIDTH-1:0]              req_tag_out,    
    input wire  [LANES-1:0]                                 req_ready_out,

    // input response
    input wire                                              rsp_valid_in,    
    input wire [LANES-1:0]                                  rsp_tmask_in,    
    input wire [LANES-1:0][DATA_WIDTH-1:0]                  rsp_data_in,
    input wire [TAG_OUT_WIDTH-1:0]                          rsp_tag_in,
    output wire                                             rsp_ready_in,

    // output responses
    output wire [NUM_REQS-1:0]                              rsp_valid_out,
    output wire [NUM_REQS-1:0][LANES-1:0]                   rsp_tmask_out,
    output wire [NUM_REQS-1:0][LANES-1:0][DATA_WIDTH-1:0]   rsp_data_out,
    output wire [NUM_REQS-1:0][TAG_IN_WIDTH-1:0]            rsp_tag_out,
    input wire  [NUM_REQS-1:0]                              rsp_ready_out    
);  
    localparam REQ_DATAW = TAG_OUT_WIDTH + ADDR_WIDTH + 1 + DATA_SIZE + DATA_WIDTH;
    localparam RSP_DATAW = LANES * (1 + DATA_WIDTH) + TAG_IN_WIDTH;

    if (NUM_REQS > 1) begin

        wire [NUM_REQS-1:0][LANES-1:0][REQ_DATAW-1:0] req_data_in_merged;
        wire [LANES-1:0][REQ_DATAW-1:0] req_data_out_merged;

        for (genvar i = 0; i < NUM_REQS; i++) begin
            for (genvar j = 0; j < LANES; ++j) begin            
                wire [TAG_OUT_WIDTH-1:0] req_tag_in_w;

                VX_bits_insert #( 
                    .N   (TAG_IN_WIDTH),
                    .S   (LOG_NUM_REQS),
                    .POS (TAG_SEL_IDX)
                ) bits_insert (
                    .data_in  (req_tag_in[i][j]),
                    .sel_in   (LOG_NUM_REQS'(i)),
                    .data_out (req_tag_in_w)
                );

                assign req_data_in_merged[i][j] = {req_tag_in_w, req_addr_in[i][j], req_rw_in[i][j], req_byteen_in[i][j], req_data_in[i][j]};
            end
        end

        VX_stream_arbiter #(            
            .NUM_REQS (NUM_REQS),
            .LANES    (LANES),
            .DATAW    (REQ_DATAW),
            .BUFFERED (BUFFERED_REQ),
            .TYPE     (TYPE)
        ) req_arb (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (req_valid_in),
            .data_in   (req_data_in_merged),
            .ready_in  (req_ready_in),
            .valid_out (req_valid_out),
            .data_out  (req_data_out_merged),
            .ready_out (req_ready_out)
        );

        for (genvar i = 0; i < LANES; ++i) begin
            assign {req_tag_out[i], req_addr_out[i], req_rw_out[i], req_byteen_out[i], req_data_out[i]} = req_data_out_merged[i];
        end

        ///////////////////////////////////////////////////////////////////////

        wire [NUM_REQS-1:0][RSP_DATAW-1:0] rsp_data_out_merged;

        wire [LOG_NUM_REQS-1:0] rsp_sel = rsp_tag_in[TAG_SEL_IDX +: LOG_NUM_REQS];

        wire [TAG_IN_WIDTH-1:0] rsp_tag_in_w;

        VX_bits_remove #( 
            .N   (TAG_OUT_WIDTH),
            .S   (LOG_NUM_REQS),
            .POS (TAG_SEL_IDX)
        ) bits_remove (
            .data_in  (rsp_tag_in),
            .data_out (rsp_tag_in_w)
        );

        VX_stream_demux #(
            .NUM_REQS (NUM_REQS),
            .LANES    (1),
            .DATAW    (RSP_DATAW),
            .BUFFERED (BUFFERED_RSP)
        ) rsp_demux (
            .clk       (clk),
            .reset     (reset),
            .sel_in    (rsp_sel),
            .valid_in  (rsp_valid_in),
            .data_in   ({rsp_tmask_in, rsp_tag_in_w, rsp_data_in}),
            .ready_in  (rsp_ready_in),
            .valid_out (rsp_valid_out),
            .data_out  (rsp_data_out_merged),
            .ready_out (rsp_ready_out)
        );
        
        for (genvar i = 0; i < NUM_REQS; i++) begin
            assign {rsp_tmask_out[i], rsp_tag_out[i], rsp_data_out[i]} = rsp_data_out_merged[i];
        end

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