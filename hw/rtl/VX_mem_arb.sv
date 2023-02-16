`include "VX_define.vh"

module VX_mem_arb #(    
    parameter NUM_REQS       = 1, 
    parameter DATA_WIDTH     = 1,
    parameter DATA_SIZE      = (DATA_WIDTH / 8),
    parameter ADDR_WIDTH     = (`XLEN - `CLOG2(DATA_SIZE)),
    parameter TAG_WIDTH      = 1,    
    parameter TAG_SEL_IDX    = 0,
    parameter BUFFERED_REQ   = 0,
    parameter BUFFERED_RSP   = 0,
    parameter `STRING ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,

    // input requests        
    VX_mem_req_if.slave     req_in_if [NUM_REQS],

    // input responses
    VX_mem_rsp_if.master    rsp_in_if [NUM_REQS],
    
    // output request
    VX_mem_req_if.master    req_out_if,

    // output response
    VX_mem_rsp_if.slave     rsp_out_if
);   
    
    localparam LOG_NUM_REQS  = `CLOG2(NUM_REQS);
    localparam TAG_OUT_WIDTH = TAG_WIDTH + LOG_NUM_REQS;
    localparam REQ_DATAW = TAG_OUT_WIDTH + ADDR_WIDTH + 1 + DATA_SIZE + DATA_WIDTH;
    localparam RSP_DATAW = TAG_WIDTH + DATA_WIDTH;

    wire [NUM_REQS-1:0]                req_valid_in;
    wire [NUM_REQS-1:0][REQ_DATAW-1:0] req_data_in;
    wire [NUM_REQS-1:0]                req_ready_in;

    for (genvar i = 0; i < NUM_REQS; ++i) begin
        wire [TAG_OUT_WIDTH-1:0] req_tag_in;

        VX_bits_insert #( 
            .N   (TAG_WIDTH),
            .S   (LOG_NUM_REQS),
            .POS (TAG_SEL_IDX)
        ) bits_insert (
            .data_in  (req_in_if[i].tag),
            .sel_in   (`UP(LOG_NUM_REQS)'(i)),
            .data_out (req_tag_in)
        );

        assign req_valid_in[i] = req_in_if[i].valid;
        assign req_data_in[i] = {req_tag_in, req_in_if[i].addr, req_in_if[i].rw, req_in_if[i].byteen, req_in_if[i].data};
        assign req_in_if[i].ready = req_ready_in[i];
    end        

    VX_stream_arb #(            
        .NUM_INPUTS (NUM_REQS),
        .DATAW      (REQ_DATAW),
        .ARBITER    (ARBITER),
        .BUFFERED   (BUFFERED_REQ),
        .MAX_FANOUT (4)
    ) req_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (req_valid_in),
        .ready_in  (req_ready_in),
        .data_in   (req_data_in),                
        .data_out  ({req_out_if.tag, req_out_if.addr, req_out_if.rw, req_out_if.byteen, req_out_if.data}),
        .valid_out (req_out_if.valid),
        .ready_out (req_out_if.ready)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire [NUM_REQS-1:0]                rsp_valid_in;
    wire [NUM_REQS-1:0][RSP_DATAW-1:0] rsp_data_in;
    wire [NUM_REQS-1:0]                rsp_ready_in;
    wire [`UP(LOG_NUM_REQS)-1:0]       rsp_sel;

    if (NUM_REQS > 1) begin
        assign rsp_sel = rsp_out_if.tag[TAG_SEL_IDX +: LOG_NUM_REQS];
    end else begin
        assign rsp_sel = '0;
    end

    wire [TAG_WIDTH-1:0] rsp_tag_out;

    VX_bits_remove #( 
        .N   (TAG_OUT_WIDTH),
        .S   (LOG_NUM_REQS),
        .POS (TAG_SEL_IDX)
    ) bits_remove (
        .data_in  (rsp_out_if.tag),
        .data_out (rsp_tag_out)
    );

    VX_stream_switch #(
        .NUM_OUTPUTS (NUM_REQS),
        .DATAW       (RSP_DATAW),
        .BUFFERED    (BUFFERED_RSP),
        .MAX_FANOUT  (4)
    ) rsp_switch (
        .clk       (clk),
        .reset     (reset),
        .sel_in    (rsp_sel),
        .valid_in  (rsp_out_if.valid),
        .ready_in  (rsp_out_if.ready),
        .data_in   ({rsp_tag_out, rsp_out_if.data}),        
        .data_out  (rsp_data_in),
        .valid_out (rsp_valid_in),        
        .ready_out (rsp_ready_in)
    );
    
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign rsp_in_if[i].valid = rsp_valid_in[i];
        assign {rsp_in_if[i].tag, rsp_in_if[i].data} = rsp_data_in[i];        
        assign rsp_ready_in[i] = rsp_in_if[i].ready;
    end

endmodule
