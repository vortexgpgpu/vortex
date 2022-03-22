`include "VX_define.vh"

module VX_mem_arb #(    
    parameter NUM_REQS      = 1, 
    parameter DATA_WIDTH    = 1,
    parameter ADDR_WIDTH    = 1,
    parameter TAG_IN_WIDTH  = 1,    
    parameter TAG_SEL_IDX   = 0,
    parameter BUFFERED_REQ  = 0,
    parameter BUFFERED_RSP  = 0,
    parameter TYPE          = "P"
) (
    input wire              clk,
    input wire              reset,

    // input requests        
    VX_mem_req_if.slave     req_in_if[NUM_REQS],
    
    // output request
    VX_mem_req_if.master    req_out_if,

    // input response
    VX_mem_rsp_if.slave     rsp_in_if,

    // output responses
    VX_mem_rsp_if.master    rsp_out_if[NUM_REQS]
);   
    
    localparam DATA_SIZE     = (DATA_WIDTH / 8);
    localparam LOG_NUM_REQS  = `CLOG2(NUM_REQS);
    localparam TAG_OUT_WIDTH = TAG_IN_WIDTH + LOG_NUM_REQS;
    localparam REQ_DATAW = TAG_OUT_WIDTH + ADDR_WIDTH + 1 + DATA_SIZE + DATA_WIDTH;
    localparam RSP_DATAW = TAG_IN_WIDTH + DATA_WIDTH;

    if (NUM_REQS > 1) begin

        wire [NUM_REQS-1:0] req_valid_in;
        wire [NUM_REQS-1:0][REQ_DATAW-1:0] req_data_in;
        wire [NUM_REQS-1:0] req_ready_in;

        for (genvar i = 0; i < NUM_REQS; i++) begin
            wire [TAG_OUT_WIDTH-1:0] req_tag_in;

            VX_bits_insert #( 
                .N   (TAG_IN_WIDTH),
                .S   (LOG_NUM_REQS),
                .POS (TAG_SEL_IDX)
            ) bits_insert (
                .data_in  (req_in_if[i].tag),
                .sel_in   (LOG_NUM_REQS'(i)),
                .data_out (req_tag_in)
            );

            assign req_valid_in[i] = req_in_if[i].valid;
            assign req_data_in[i] = {req_tag_in, req_in_if[i].addr, req_in_if[i].rw, req_in_if[i].byteen, req_in_if[i].data};
            assign req_in_if[i].ready = req_ready_in[i];
        end        

        VX_stream_mux #(            
            .NUM_REQS (NUM_REQS),
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
            .data_out  ({req_out_if.tag, req_out_if.addr, req_out_if.rw, req_out_if.byteen, req_out_if.data}),
            .ready_out (req_out_if.ready)
        );

        ///////////////////////////////////////////////////////////////////////

        wire [NUM_REQS-1:0] rsp_valid_out;
        wire [NUM_REQS-1:0][RSP_DATAW-1:0] rsp_data_out;
        wire [NUM_REQS-1:0] rsp_ready_out;

        wire [LOG_NUM_REQS-1:0] rsp_sel = rsp_in_if.tag[TAG_SEL_IDX +: LOG_NUM_REQS];

        wire [TAG_IN_WIDTH-1:0] rsp_tag_in;

        VX_bits_remove #( 
            .N   (TAG_OUT_WIDTH),
            .S   (LOG_NUM_REQS),
            .POS (TAG_SEL_IDX)
        ) bits_remove (
            .data_in  (rsp_in_if.tag),
            .data_out (rsp_tag_in)
        );

        VX_stream_demux #(
            .NUM_REQS (NUM_REQS),
            .DATAW    (RSP_DATAW),
            .BUFFERED (BUFFERED_RSP)
        ) rsp_demux (
            .clk       (clk),
            .reset     (reset),
            .sel_in    (rsp_sel),
            .valid_in  (rsp_in_if.valid),
            .data_in   ({rsp_tag_in, rsp_in_if.data}),
            .ready_in  (rsp_in_if.ready),
            .valid_out (rsp_valid_out),
            .data_out  (rsp_data_out),
            .ready_out (rsp_ready_out)
        );
        
        for (genvar i = 0; i < NUM_REQS; i++) begin
            assign rsp_out_if[i].valid = rsp_valid_out[i];
            assign {rsp_out_if[i].tag, rsp_out_if[i].data} = rsp_data_out[i];            
            assign rsp_ready_out[i] = rsp_out_if[i].ready;
        end                

    end else begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        assign req_out_if.valid  = req_in_if[0].valid;
        assign req_out_if.tag    = req_in_if[0].tag;
        assign req_out_if.addr   = req_in_if[0].addr;
        assign req_out_if.rw     = req_in_if[0].rw;
        assign req_out_if.byteen = req_in_if[0].byteen;
        assign req_out_if.data   = req_in_if[0].data;
        assign req_in_if[0].ready= req_out_if.ready;

        assign rsp_out_if[0].valid = rsp_in_if.valid;
        assign rsp_out_if[0].tag   = rsp_in_if.tag;
        assign rsp_out_if[0].data  = rsp_in_if.data;
        assign rsp_in_if.ready  = rsp_out_if[0].ready;

    end

endmodule
