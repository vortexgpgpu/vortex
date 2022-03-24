`include "VX_define.vh"

module VX_cache_demux #(
    parameter NUM_REQS      = 1, 
    parameter LANES         = 1,
    parameter DATA_SIZE     = 1,
    parameter TAG_IN_WIDTH  = 1,
    parameter TAG_SEL_IDX   = 0,   
    parameter BUFFERED_REQ  = 0,
    parameter BUFFERED_RSP  = 0,
    parameter ARBITER       = "P",

    parameter ADDR_WIDTH    = (32-`CLOG2(DATA_SIZE)),
    parameter DATA_WIDTH    = (8 * DATA_SIZE),
    parameter LOG_NUM_REQS  = `CLOG2(NUM_REQS),
    parameter TAG_OUT_WIDTH = TAG_IN_WIDTH - LOG_NUM_REQS
) (
    input wire clk,
    input wire reset,

    // input requests        
    VX_cache_req_if.slave     req_in_if,

    // input responses
    VX_cache_rsp_if.master    rsp_in_if,
    
    // output request
    VX_cache_req_if.master    req_out_if[NUM_REQS],

    // output response
    VX_cache_rsp_if.slave     rsp_out_if[NUM_REQS]    
);  
    localparam REQ_DATAW = TAG_OUT_WIDTH + ADDR_WIDTH + 1 + DATA_SIZE + DATA_WIDTH;
    localparam RSP_DATAW = TAG_IN_WIDTH + DATA_WIDTH;

    if (NUM_REQS > 1) begin
        wire [NUM_REQS-1:0][LANES-1:0] req_valid_out;
        wire [NUM_REQS-1:0][LANES-1:0][REQ_DATAW-1:0] req_data_out;
        wire [NUM_REQS-1:0][LANES-1:0] req_ready_out;

        wire [LANES-1:0][REQ_DATAW-1:0] req_data_in;        
        wire [LANES-1:0][LOG_NUM_REQS-1:0] req_sel;
        
        for (genvar i = 0; i < LANES; ++i) begin

            wire [TAG_OUT_WIDTH-1:0] req_tag_in;
            
            VX_bits_remove #( 
                .N   (TAG_IN_WIDTH),
                .S   (LOG_NUM_REQS),
                .POS (TAG_SEL_IDX)
            ) bits_remove (
                .data_in  (req_in_if.tag[i]),
                .data_out (req_tag_in)
            );            

            assign req_sel[i] = req_in_if.tag[i][TAG_SEL_IDX +: LOG_NUM_REQS];
            assign req_data_in[i] = {req_tag_in, req_in_if.addr[i], req_in_if.rw[i], req_in_if.byteen[i], req_in_if.data[i]};
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
            .valid_in  (req_in_if.valid),
            .data_in   (req_data_in),
            .ready_in  (req_in_if.ready),
            .valid_out (req_valid_out),
            .data_out  (req_data_out),
            .ready_out (req_ready_out)
        );
        
        for (genvar i = 0; i < NUM_REQS; i++) begin
            for (genvar j = 0; j < LANES; ++j) begin
                assign req_out_if[i].valid[j] = req_valid_out[i][j];
                assign {req_out_if[i].tag[j], req_out_if[i].addr[j], req_out_if[i].rw[j], req_out_if[i].byteen[j], req_out_if[i].data[j]} = req_data_out[i][j];
                assign req_ready_out[i][j] = req_out_if[i].ready[j];
            end
        end

        ///////////////////////////////////////////////////////////////////////        

        wire [NUM_REQS-1:0][LANES-1:0] rsp_valid_out;
        wire [NUM_REQS-1:0][LANES-1:0][RSP_DATAW-1:0] rsp_data_out;
        wire [NUM_REQS-1:0][LANES-1:0] rsp_ready_out;

        wire [LANES-1:0][RSP_DATAW-1:0] rsp_data_in;
        
        for (genvar i = 0; i < NUM_REQS; i++) begin
            for (genvar j = 0; j < LANES; ++j) begin     
                wire [TAG_IN_WIDTH-1:0] rsp_tag_out;
                
                VX_bits_insert #( 
                    .N   (TAG_OUT_WIDTH),
                    .S   (LOG_NUM_REQS),
                    .POS (TAG_SEL_IDX)
                ) bits_insert (
                    .data_in  (rsp_out_if[i].tag[j]),
                    .sel_in   (LOG_NUM_REQS'(i)),
                    .data_out (rsp_tag_out)
                );

                assign rsp_valid_out[i][j] = rsp_out_if[i].valid[j];
                assign rsp_data_out[i][j] = {rsp_tag_out, rsp_out_if[i].data[j]};
                assign rsp_out_if[i].ready[j] = rsp_ready_out[i][j];
            end
        end

        VX_stream_mux #(            
            .NUM_REQS (NUM_REQS),
            .LANES    (LANES),
            .DATAW    (RSP_DATAW),
            .BUFFERED (BUFFERED_RSP),
            .ARBITER  (ARBITER)
        ) rsp_mux (
            .clk       (clk),
            .reset     (reset),
            `UNUSED_PIN (sel_in),
            .valid_in  (rsp_valid_out),
            .data_in   (rsp_data_out),
            .ready_in  (rsp_ready_out),
            .valid_out (rsp_in_if.valid),
            .data_out  (rsp_data_in),
            .ready_out (rsp_in_if.ready)
        );

        for (genvar i = 0; i < LANES; ++i) begin
            assign {rsp_in_if.tag[i], rsp_in_if.data[i]} = rsp_data_in[i];
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
        assign rsp_out_if.ready = rsp_in_if.ready;

    end

endmodule
