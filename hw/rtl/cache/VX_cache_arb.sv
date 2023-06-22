`include "VX_define.vh"

module VX_cache_arb #(    
    parameter NUM_INPUTS     = 1, 
    parameter NUM_OUTPUTS    = 1,
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
    
    VX_cache_bus_if.slave   bus_in_if [NUM_INPUTS],
    VX_cache_bus_if.master  bus_out_if [NUM_OUTPUTS]
);     

    localparam ADDR_WIDTH    = (`XLEN-`CLOG2(DATA_SIZE));
    localparam DATA_WIDTH    = (8 * DATA_SIZE);
    localparam LOG_NUM_REQS  = `ARB_SEL_BITS(NUM_INPUTS, NUM_OUTPUTS);
    localparam NUM_REQS      = 1 << LOG_NUM_REQS;
    localparam TAG_OUT_WIDTH = TAG_WIDTH + LOG_NUM_REQS;    
    localparam REQ_DATAW = TAG_OUT_WIDTH + ADDR_WIDTH + 1 + DATA_SIZE + DATA_WIDTH;
    localparam RSP_DATAW = TAG_WIDTH + DATA_WIDTH;

    `STATIC_ASSERT ((NUM_INPUTS >= NUM_OUTPUTS), ("invalid parameter"))

    wire [NUM_INPUTS-1:0][NUM_LANES-1:0]                 req_valid_in;
    wire [NUM_INPUTS-1:0][NUM_LANES-1:0][REQ_DATAW-1:0]  req_data_in;
    wire [NUM_INPUTS-1:0][NUM_LANES-1:0]                 req_ready_in;
    
    wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0]                req_valid_out;
    wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0][REQ_DATAW-1:0] req_data_out;
    wire [NUM_OUTPUTS-1:0][NUM_LANES-1:0]                req_ready_out;

    for (genvar i = 0; i < NUM_INPUTS; ++i) begin
        for (genvar j = 0; j < NUM_LANES; ++j) begin

            assign req_valid_in[i][j] = bus_in_if[i].req_valid[j];
            assign bus_in_if[i].req_ready[j] = req_ready_in[i][j];

            if (NUM_INPUTS > NUM_OUTPUTS) begin
                wire [TAG_OUT_WIDTH-1:0] req_tag_in;
                localparam r = i % NUM_REQS;
                VX_bits_insert #( 
                    .N   (TAG_WIDTH),
                    .S   (LOG_NUM_REQS),
                    .POS (TAG_SEL_IDX)
                ) bits_insert (
                    .data_in  (bus_in_if[i].req_tag[j]),
                    .sel_in   (LOG_NUM_REQS'(r)),
                    .data_out (req_tag_in)
                );
                assign req_data_in[i][j] = {req_tag_in, bus_in_if[i].req_addr[j], bus_in_if[i].req_rw[j], bus_in_if[i].req_byteen[j], bus_in_if[i].req_data[j]};
            end else begin
                assign req_data_in[i][j] = {bus_in_if[i].req_tag[j], bus_in_if[i].req_addr[j], bus_in_if[i].req_rw[j], bus_in_if[i].req_byteen[j], bus_in_if[i].req_data[j]};
            end            
        end
    end

    VX_stream_arb #(            
        .NUM_INPUTS  (NUM_INPUTS),
        .NUM_OUTPUTS (NUM_OUTPUTS),
        .NUM_LANES   (NUM_LANES),
        .DATAW       (REQ_DATAW),
        .ARBITER     (ARBITER),
        .BUFFERED    (BUFFERED_REQ),
        .MAX_FANOUT  (4)
    ) req_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (req_valid_in),
        .ready_in  (req_ready_in),
        .data_in   (req_data_in),
        .data_out  (req_data_out),     
        .valid_out (req_valid_out),        
        .ready_out (req_ready_out)
    );

    for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
        for (genvar j = 0; j < NUM_LANES; ++j) begin
            assign bus_out_if[i].req_valid[j] = req_valid_out[i][j];
            assign {bus_out_if[i].req_tag[j], bus_out_if[i].req_addr[j], bus_out_if[i].req_rw[j], bus_out_if[i].req_byteen[j], bus_out_if[i].req_data[j]} = req_data_out[i][j];
            assign req_ready_out[i][j] = bus_out_if[i].req_ready[j];
        end
    end

    ///////////////////////////////////////////////////////////////////////////

     for (genvar i = 0; i < NUM_LANES; ++i) begin

        wire [NUM_INPUTS-1:0]                 rsp_valid_out;
        wire [NUM_INPUTS-1:0][RSP_DATAW-1:0]  rsp_data_out;
        wire [NUM_INPUTS-1:0]                 rsp_ready_out;

        wire [NUM_OUTPUTS-1:0]                rsp_valid_in;
        wire [NUM_OUTPUTS-1:0][RSP_DATAW-1:0] rsp_data_in;
        wire [NUM_OUTPUTS-1:0]                rsp_ready_in;

        if (NUM_INPUTS > NUM_OUTPUTS) begin

            wire [NUM_OUTPUTS-1:0][LOG_NUM_REQS-1:0] rsp_sel_in;

            for (genvar j = 0; j < NUM_OUTPUTS; ++j) begin
                wire [TAG_WIDTH-1:0] rsp_tag_out;

                VX_bits_remove #( 
                    .N   (TAG_OUT_WIDTH),
                    .S   (LOG_NUM_REQS),
                    .POS (TAG_SEL_IDX)
                ) bits_remove (
                    .data_in  (bus_out_if[j].rsp_tag[i]),
                    .data_out (rsp_tag_out)
                );

                assign rsp_valid_in[j] = bus_out_if[j].rsp_valid[i];
                assign rsp_data_in[j] = {rsp_tag_out, bus_out_if[j].rsp_data[i]};
                assign bus_out_if[j].rsp_ready[i] = rsp_ready_in[j];

                if (NUM_INPUTS > 1) begin
                    assign rsp_sel_in[j] = bus_out_if[j].rsp_tag[i][TAG_SEL_IDX +: LOG_NUM_REQS];
                end else begin
                    assign rsp_sel_in[j] = '0;
                end
            end        

            VX_stream_switch #(
                .NUM_INPUTS  (NUM_OUTPUTS),
                .NUM_OUTPUTS (NUM_INPUTS),        
                .DATAW       (RSP_DATAW),
                .BUFFERED    (BUFFERED_RSP),
                .MAX_FANOUT  (4)
            ) rsp_switch (
                .clk       (clk),
                .reset     (reset),
                .sel_in    (rsp_sel_in),
                .valid_in  (rsp_valid_in),
                .ready_in  (rsp_ready_in),
                .data_in   (rsp_data_in),
                .data_out  (rsp_data_out),              
                .valid_out (rsp_valid_out),                
                .ready_out (rsp_ready_out)
            );

        end else begin
            
            for (genvar j = 0; j < NUM_OUTPUTS; ++j) begin
                assign rsp_valid_in[j] = bus_out_if[j].rsp_valid[i];
                assign rsp_data_in[j] = {bus_out_if[j].rsp_tag[i], bus_out_if[j].rsp_data[i]};
                assign bus_out_if[j].rsp_ready[i] = rsp_ready_in[j];
            end

            VX_stream_arb #(            
                .NUM_INPUTS  (NUM_OUTPUTS),
                .NUM_OUTPUTS (NUM_INPUTS),
                .DATAW       (RSP_DATAW),
                .ARBITER     (ARBITER),
                .BUFFERED    (BUFFERED_RSP),
                .MAX_FANOUT  (4)
            ) req_arb (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (rsp_valid_in),
                .ready_in  (rsp_ready_in),
                .data_in   (rsp_data_in),                
                .data_out  (rsp_data_out),
                .valid_out (rsp_valid_out),                
                .ready_out (rsp_ready_out)
            );

        end
        
        for (genvar j = 0; j < NUM_INPUTS; ++j) begin
            assign bus_in_if[j].rsp_valid[i] = rsp_valid_out[j];
            assign {bus_in_if[j].rsp_tag[i], bus_in_if[j].rsp_data[i]} = rsp_data_out[j];
            assign rsp_ready_out[j] = bus_in_if[j].rsp_ready[i];
        end
    end    

endmodule
