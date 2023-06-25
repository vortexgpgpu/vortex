`include "VX_define.vh"

module VX_mem_arb #(    
    parameter NUM_REQS       = 1, 
    parameter DATA_WIDTH     = 1,
    parameter ADDR_WIDTH     = (`MEM_ADDR_WIDTH - `CLOG2(DATA_SIZE)),
    parameter TAG_WIDTH      = 1,    
    parameter TAG_SEL_IDX    = 0,
    parameter BUFFERED_REQ   = 0,
    parameter BUFFERED_RSP   = 0,
    parameter `STRING ARBITER = "R",
    parameter DATA_SIZE      = (DATA_WIDTH / 8)    
) (
    input wire              clk,
    input wire              reset,

    VX_mem_bus_if.slave     bus_in_if [NUM_REQS],    
    VX_mem_bus_if.master    bus_out_if
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
            .data_in  (bus_in_if[i].req_tag),
            .sel_in   (`UP(LOG_NUM_REQS)'(i)),
            .data_out (req_tag_in)
        );

        assign req_valid_in[i] = bus_in_if[i].req_valid;
        assign req_data_in[i] = {req_tag_in, bus_in_if[i].req_addr, bus_in_if[i].req_rw, bus_in_if[i].req_byteen, bus_in_if[i].req_data};
        assign bus_in_if[i].req_ready = req_ready_in[i];
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
        .data_out  ({bus_out_if.req_tag, bus_out_if.req_addr, bus_out_if.req_rw, bus_out_if.req_byteen, bus_out_if.req_data}),
        .valid_out (bus_out_if.req_valid),
        .ready_out (bus_out_if.req_ready)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire [NUM_REQS-1:0]                rsp_valid_in;
    wire [NUM_REQS-1:0][RSP_DATAW-1:0] rsp_data_in;
    wire [NUM_REQS-1:0]                rsp_ready_in;
    wire [`UP(LOG_NUM_REQS)-1:0]       rsp_sel;

    if (NUM_REQS > 1) begin
        assign rsp_sel = bus_out_if.rsp_tag[TAG_SEL_IDX +: LOG_NUM_REQS];
    end else begin
        assign rsp_sel = '0;
    end

    wire [TAG_WIDTH-1:0] rsp_tag_out;

    VX_bits_remove #( 
        .N   (TAG_OUT_WIDTH),
        .S   (LOG_NUM_REQS),
        .POS (TAG_SEL_IDX)
    ) bits_remove (
        .data_in  (bus_out_if.rsp_tag),
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
        .valid_in  (bus_out_if.rsp_valid),
        .ready_in  (bus_out_if.rsp_ready),
        .data_in   ({rsp_tag_out, bus_out_if.rsp_data}),        
        .data_out  (rsp_data_in),
        .valid_out (rsp_valid_in),        
        .ready_out (rsp_ready_in)
    );
    
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        assign bus_in_if[i].rsp_valid = rsp_valid_in[i];
        assign {bus_in_if[i].rsp_tag, bus_in_if[i].rsp_data} = rsp_data_in[i];        
        assign rsp_ready_in[i] = bus_in_if[i].rsp_ready;
    end

endmodule
