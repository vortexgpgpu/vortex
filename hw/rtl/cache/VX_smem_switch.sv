`include "VX_define.vh"

module VX_smem_switch #(
    parameter NUM_REQS       = 1, 
    parameter NUM_LANES      = 1,
    parameter DATA_SIZE      = 1,
    parameter TAG_WIDTH      = 1,
    parameter TAG_SEL_IDX    = 0,   
    parameter BUFFERED_REQ   = 0,
    parameter BUFFERED_RSP   = 0,
    parameter `STRING ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,

    // input requests        
    VX_cache_req_if.slave   req_in_if,

    // input responses
    VX_cache_rsp_if.master  rsp_in_if,
    
    // output request
    VX_cache_req_if.master  req_out_if [NUM_REQS],

    // output response
    VX_cache_rsp_if.slave   rsp_out_if [NUM_REQS]    
);  
    localparam ADDR_WIDTH    = (`XLEN-`CLOG2(DATA_SIZE));
    localparam DATA_WIDTH    = (8 * DATA_SIZE);
    localparam LOG_NUM_REQS  = `CLOG2(NUM_REQS);
    localparam TAG_OUT_WIDTH = TAG_WIDTH - LOG_NUM_REQS;
    localparam REQ_DATAW     = TAG_OUT_WIDTH + ADDR_WIDTH + 1 + DATA_SIZE + DATA_WIDTH;
    localparam RSP_DATAW     = TAG_WIDTH + DATA_WIDTH;      
        
    for (genvar i = 0; i < NUM_LANES; ++i) begin

        wire [NUM_REQS-1:0]                req_valid_out;
        wire [NUM_REQS-1:0][REQ_DATAW-1:0] req_data_out;
        wire [NUM_REQS-1:0]                req_ready_out;

        wire [REQ_DATAW-1:0]         req_data_in;
        wire [TAG_OUT_WIDTH-1:0]     req_tag_in;
        wire [`UP(LOG_NUM_REQS)-1:0] req_sel_in;
        
        VX_bits_remove #( 
            .N   (TAG_WIDTH),
            .S   (LOG_NUM_REQS),
            .POS (TAG_SEL_IDX)
        ) bits_remove (
            .data_in  (req_in_if.tag[i]),
            .data_out (req_tag_in)
        );            

        if (NUM_REQS > 1) begin
            assign req_sel_in = req_in_if.tag[i][TAG_SEL_IDX +: LOG_NUM_REQS];
        end else begin
            assign req_sel_in = '0;
        end

        assign req_data_in = {req_tag_in, req_in_if.addr[i], req_in_if.rw[i], req_in_if.byteen[i], req_in_if.data[i]};

        VX_stream_switch #(
            .NUM_OUTPUTS (NUM_REQS),
            .DATAW       (REQ_DATAW),
            .BUFFERED    (BUFFERED_REQ),
            .MAX_FANOUT  (4)
        ) req_switch (
            .clk       (clk),
            .reset     (reset),
            .sel_in    (req_sel_in),
            .valid_in  (req_in_if.valid[i]),
            .ready_in  (req_in_if.ready[i]),
            .data_in   (req_data_in),
            .data_out  (req_data_out),
            .valid_out (req_valid_out),
            .ready_out (req_ready_out)
        );
    
        for (genvar j = 0; j < NUM_REQS; ++j) begin
            assign req_out_if[j].valid[i] = req_valid_out[j];
            assign {req_out_if[j].tag[i], req_out_if[j].addr[i], req_out_if[j].rw[i], req_out_if[j].byteen[i], req_out_if[j].data[i]} = req_data_out[j];
            assign req_ready_out[j] = req_out_if[j].ready[i];
        end
    end

    ///////////////////////////////////////////////////////////////////////        

    wire [NUM_REQS-1:0][NUM_LANES-1:0]                rsp_valid_out;
    wire [NUM_REQS-1:0][NUM_LANES-1:0][RSP_DATAW-1:0] rsp_data_out;
    wire [NUM_REQS-1:0][NUM_LANES-1:0]                rsp_ready_out;
    wire [NUM_LANES-1:0][RSP_DATAW-1:0]               rsp_data_in;
    
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        for (genvar j = 0; j < NUM_LANES; ++j) begin     
            wire [TAG_WIDTH-1:0] rsp_tag_out;
            
            VX_bits_insert #( 
                .N   (TAG_OUT_WIDTH),
                .S   (LOG_NUM_REQS),
                .POS (TAG_SEL_IDX)
            ) bits_insert (
                .data_in  (rsp_out_if[i].tag[j]),
                .sel_in   (`UP(LOG_NUM_REQS)'(i)),
                .data_out (rsp_tag_out)
            );

            assign rsp_valid_out[i][j] = rsp_out_if[i].valid[j];
            assign rsp_data_out[i][j] = {rsp_tag_out, rsp_out_if[i].data[j]};
            assign rsp_out_if[i].ready[j] = rsp_ready_out[i][j];
        end
    end

    VX_stream_arb #(            
        .NUM_INPUTS (NUM_REQS),
        .NUM_LANES  (NUM_LANES),
        .DATAW      (RSP_DATAW),        
        .ARBITER    (ARBITER),
        .BUFFERED   (BUFFERED_RSP)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rsp_valid_out),        
        .ready_in  (rsp_ready_out),
        .data_in   (rsp_data_out),     
        .data_out  (rsp_data_in),
        .valid_out (rsp_in_if.valid),
        .ready_out (rsp_in_if.ready)
    );

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign {rsp_in_if.tag[i], rsp_in_if.data[i]} = rsp_data_in[i];
    end

endmodule
