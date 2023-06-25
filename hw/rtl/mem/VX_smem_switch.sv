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
    
    VX_cache_bus_if.slave   bus_in_if,
    VX_cache_bus_if.master  bus_out_if [NUM_REQS]
);  
    localparam ADDR_WIDTH    = (`MEM_ADDR_WIDTH-`CLOG2(DATA_SIZE));
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
            .data_in  (bus_in_if.req_tag[i]),
            .data_out (req_tag_in)
        );            

        if (NUM_REQS > 1) begin
            assign req_sel_in = bus_in_if.req_tag[i][TAG_SEL_IDX +: LOG_NUM_REQS];
        end else begin
            assign req_sel_in = '0;
        end

        assign req_data_in = {req_tag_in, bus_in_if.req_addr[i], bus_in_if.req_rw[i], bus_in_if.req_byteen[i], bus_in_if.req_data[i]};

        VX_stream_switch #(
            .NUM_OUTPUTS (NUM_REQS),
            .DATAW       (REQ_DATAW),
            .BUFFERED    (BUFFERED_REQ),
            .MAX_FANOUT  (4)
        ) req_switch (
            .clk       (clk),
            .reset     (reset),
            .sel_in    (req_sel_in),
            .valid_in  (bus_in_if.req_valid[i]),
            .ready_in  (bus_in_if.req_ready[i]),
            .data_in   (req_data_in),
            .data_out  (req_data_out),
            .valid_out (req_valid_out),
            .ready_out (req_ready_out)
        );
    
        for (genvar j = 0; j < NUM_REQS; ++j) begin
            assign bus_out_if[j].req_valid[i] = req_valid_out[j];
            assign {bus_out_if[j].req_tag[i], bus_out_if[j].req_addr[i], bus_out_if[j].req_rw[i], bus_out_if[j].req_byteen[i], bus_out_if[j].req_data[i]} = req_data_out[j];
            assign req_ready_out[j] = bus_out_if[j].req_ready[i];
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
                .data_in  (bus_out_if[i].rsp_tag[j]),
                .sel_in   (`UP(LOG_NUM_REQS)'(i)),
                .data_out (rsp_tag_out)
            );

            assign rsp_valid_out[i][j] = bus_out_if[i].rsp_valid[j];
            assign rsp_data_out[i][j] = {rsp_tag_out, bus_out_if[i].rsp_data[j]};
            assign bus_out_if[i].rsp_ready[j] = rsp_ready_out[i][j];
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
        .valid_out (bus_in_if.rsp_valid),
        .ready_out (bus_in_if.rsp_ready)
    );

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign {bus_in_if.rsp_tag[i], bus_in_if.rsp_data[i]} = rsp_data_in[i];
    end

endmodule
