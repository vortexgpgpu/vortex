`include "VX_define.vh"

module VX_mem_arb #(    
    parameter NUM_REQS      = 1, 
    parameter DATA_WIDTH    = 1,
    parameter TAG_IN_WIDTH  = 1,
    parameter TAG_OUT_WIDTH = 1,
    
    parameter DATA_SIZE  = (DATA_WIDTH / 8), 
    parameter ADDR_WIDTH = 32 - `CLOG2(DATA_SIZE),
    parameter REQS_BITS  = `CLOG2(NUM_REQS)
) (
    input wire clk,
    input wire reset,

    // input requests    
    input wire [NUM_REQS-1:0]                   req_valid_in,    
    input wire [NUM_REQS-1:0][TAG_IN_WIDTH-1:0] req_tag_in,  
    input wire [NUM_REQS-1:0][ADDR_WIDTH-1:0]   req_addr_in,
    input wire [NUM_REQS-1:0]                   req_rw_in,  
    input wire [NUM_REQS-1:0][DATA_SIZE-1:0]    req_byteen_in,  
    input wire [NUM_REQS-1:0][DATA_WIDTH-1:0]   req_data_in,  
    output wire [NUM_REQS-1:0]                  req_ready_in,

    // output request
    output wire                                 req_valid_out,
    output wire [TAG_OUT_WIDTH-1:0]             req_tag_out,   
    output wire [ADDR_WIDTH-1:0]                req_addr_out, 
    output wire                                 req_rw_out,  
    output wire [DATA_SIZE-1:0]                 req_byteen_out,  
    output wire [DATA_WIDTH-1:0]                req_data_out,    
    input wire                                  req_ready_out,

    // input response
    output wire [NUM_REQS-1:0]                  rsp_valid_out,
    output wire [NUM_REQS-1:0][TAG_IN_WIDTH-1:0] rsp_tag_out,
    output wire [NUM_REQS-1:0][DATA_WIDTH-1:0]  rsp_data_out,
    input wire  [NUM_REQS-1:0]                  rsp_ready_out,

    // output response
    input wire                                  rsp_valid_in,
    input wire [TAG_OUT_WIDTH-1:0]              rsp_tag_in,
    input wire [DATA_WIDTH-1:0]                 rsp_data_in,
    output wire                                 rsp_ready_in
);
    localparam DATAW = TAG_OUT_WIDTH + ADDR_WIDTH + 1 + DATA_SIZE + DATA_WIDTH;

    if (NUM_REQS > 1) begin

        wire [NUM_REQS-1:0][DATAW-1:0] data_in;
        for (genvar i = 0; i < NUM_REQS; i++) begin
            assign data_in[i] = {{req_tag_in[i], REQS_BITS'(i)}, req_addr_in[i], req_rw_in[i], req_byteen_in[i], req_data_in[i]};
        end

        // Inputs buffering
        wire [NUM_REQS-1:0]            req_valid_in_qual; 
        wire [NUM_REQS-1:0][DATAW-1:0] req_data_in_qual;
        wire [NUM_REQS-1:0]            req_ready_in_qual;
        for (genvar i = 0; i < NUM_REQS; ++i) begin
            VX_skid_buffer #(
                .DATAW    (DATAW),
                .PASSTHRU (NUM_REQS < 4)
            ) req_buffer (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (req_valid_in[i]),        
                .data_in   (data_in[i]),
                .ready_in  (req_ready_in[i]),        
                .valid_out (req_valid_in_qual[i]),
                .data_out  (req_data_in_qual[i]),
                .ready_out (req_ready_in_qual[i])
            );
        end

        VX_stream_arbiter #(
            .NUM_REQS (NUM_REQS),
            .DATAW    (DATAW),
            .BUFFERED (NUM_REQS >= 4)
        ) req_arb (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (req_valid_in_qual),
            .data_in    (req_data_in_qual),
            .ready_in   (req_ready_in_qual),
            .valid_out  (req_valid_out),
            .data_out   ({req_tag_out, req_addr_out, req_rw_out, req_byteen_out, req_data_out}),
            .ready_out  (req_ready_out)
        );

        ///////////////////////////////////////////////////////////////////////

        wire [REQS_BITS-1:0] rsp_sel = rsp_tag_in [REQS_BITS-1:0];
        
        for (genvar i = 0; i < NUM_REQS; i++) begin                
            assign rsp_valid_out [i] = rsp_valid_in && (rsp_sel == REQS_BITS'(i));
            assign rsp_tag_out [i]   = rsp_tag_in[REQS_BITS +: TAG_IN_WIDTH];        
            assign rsp_data_out [i]  = rsp_data_in;      
        end
        
        assign rsp_ready_in = rsp_ready_out [rsp_sel];        

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
        assign rsp_tag_out    = rsp_tag_in;
        assign rsp_data_out   = rsp_data_in;
        assign rsp_ready_in   = rsp_ready_out;

    end

endmodule