`include "VX_define.vh"

module VX_cache_mux #(    
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
    VX_cache_req_if.slave     req_in_if[NUM_REQS],

    // input responses
    VX_cache_rsp_if.master    rsp_in_if[NUM_REQS],
    
    // output request
    VX_cache_req_if.master    req_out_if,

    // output response
    VX_cache_rsp_if.slave     rsp_out_if
);  
    localparam REQ_DATAW = TAG_OUT_WIDTH + ADDR_WIDTH + 1 + DATA_SIZE + DATA_WIDTH;
    localparam RSP_DATAW = TAG_IN_WIDTH + DATA_WIDTH;

    if (NUM_REQS > 1) begin

        wire [NUM_REQS-1:0][LANES-1:0] req_valid_in;
        wire [NUM_REQS-1:0][LANES-1:0][REQ_DATAW-1:0] req_data_in;
        wire [NUM_REQS-1:0][LANES-1:0] req_ready_in;

        wire [LANES-1:0][REQ_DATAW-1:0] req_data_out;

        for (genvar i = 0; i < NUM_REQS; ++i) begin
            for (genvar j = 0; j < LANES; ++j) begin            
                wire [TAG_OUT_WIDTH-1:0] req_tag_in;

                VX_bits_insert #( 
                    .N   (TAG_IN_WIDTH),
                    .S   (LOG_NUM_REQS),
                    .POS (TAG_SEL_IDX)
                ) bits_insert (
                    .data_in  (req_in_if[i].tag[j]),
                    .sel_in   (LOG_NUM_REQS'(i)),
                    .data_out (req_tag_in)
                );
            
                assign req_valid_in[i][j] = req_in_if[i].valid[j];
                assign req_data_in[i][j] = {req_tag_in, req_in_if[i].addr[j], req_in_if[i].rw[j], req_in_if[i].byteen[j], req_in_if[i].data[j]};
                assign req_in_if[i].ready[j] = req_ready_in[i][j];
            end
        end

        VX_stream_mux #(            
            .NUM_REQS (NUM_REQS),
            .LANES    (LANES),
            .DATAW    (REQ_DATAW),
            .BUFFERED (BUFFERED_REQ),
            .TYPE     (TYPE)
        ) req_mux (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (req_valid_in),
            .data_in   (req_data_in),
            .ready_in  (req_ready_in),
            .valid_out (req_out_if.valid),
            .data_out  (req_data_out),
            .ready_out (req_out_if.ready)
        );

        for (genvar i = 0; i < LANES; ++i) begin
            assign {req_out_if.tag[i], req_out_if.addr[i], req_out_if.rw[i], req_out_if.byteen[i], req_out_if.data[i]} = req_data_out[i];
        end

        ///////////////////////////////////////////////////////////////////////

        wire [NUM_REQS-1:0][LANES-1:0] rsp_valid_in;
        wire [NUM_REQS-1:0][LANES-1:0][RSP_DATAW-1:0] rsp_data_in;
        wire [NUM_REQS-1:0][LANES-1:0] rsp_ready_in;

        wire [LANES-1:0][RSP_DATAW-1:0] rsp_data_out;
        wire [LANES-1:0][LOG_NUM_REQS-1:0] rsp_sel;

        for (genvar i = 0; i < LANES; ++i) begin

            wire [TAG_IN_WIDTH-1:0] rsp_tag_out;

            VX_bits_remove #( 
                .N   (TAG_OUT_WIDTH),
                .S   (LOG_NUM_REQS),
                .POS (TAG_SEL_IDX)
            ) bits_remove (
                .data_in  (rsp_out_if.tag[i]),
                .data_out (rsp_tag_out)
            );            

            assign rsp_sel[i]= rsp_out_if.tag[i][TAG_SEL_IDX +: LOG_NUM_REQS];
            assign rsp_data_out[i] = {rsp_tag_out, rsp_out_if.data[i]};
        end

        VX_stream_demux #(
            .NUM_REQS (NUM_REQS),
            .LANES    (LANES),
            .DATAW    (RSP_DATAW),
            .BUFFERED (BUFFERED_RSP)
        ) rsp_demux (
            .clk       (clk),
            .reset     (reset),
            .sel_in    (rsp_sel),
            .valid_in  (rsp_out_if.valid),
            .data_in   (rsp_data_out),
            .ready_in  (rsp_out_if.ready),
            .valid_out (rsp_valid_in),
            .data_out  (rsp_data_in),
            .ready_out (rsp_ready_in)
        );
        
        for (genvar i = 0; i < NUM_REQS; i++) begin
            for (genvar j = 0; j < LANES; ++j) begin
                assign rsp_in_if[i].valid[j] = rsp_valid_in[i][j];
                assign {rsp_in_if[i].tag[j], rsp_in_if[i].data[j]} = rsp_data_in[i][j];        
                assign rsp_ready_in[i][j] = rsp_in_if[i].ready[j];
            end
        end

    end else begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        assign req_out_if.valid  = req_in_if.valid;
        assign req_out_if.tag    = req_in_if.tag;
        assign req_out_if.addr   = req_in_if.addr;
        assign req_out_if.rw     = req_in_if.rw;
        assign req_out_if.byteen = req_in_if.byteen;
        assign req_out_if.data   = req_in_if.data;
        assign req_in_if.ready   = req_out_if.ready;

        assign rsp_in_if.valid  = rsp_out_if.valid;
        assign rsp_in_if.tag    = rsp_out_if.tag;
        assign rsp_in_if.data   = rsp_out_if.data;
        assign rsp_out_if.ready   = rsp_in_if.ready;

    end

endmodule
