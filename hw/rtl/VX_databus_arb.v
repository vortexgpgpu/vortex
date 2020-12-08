`include "VX_define.vh"

module VX_databus_arb #(    
    parameter NUM_REQS      = 1, 
    parameter WORD_SIZE     = 1, 
    parameter TAG_IN_WIDTH  = 1,
    parameter TAG_OUT_WIDTH = 1,

    parameter WORD_WIDTH   = WORD_SIZE * 8,
    parameter ADDR_WIDTH   = 32 - `CLOG2(WORD_SIZE),
    parameter LOG_NUM_REQS = `CLOG2(NUM_REQS)
) (
    input wire clk,
    input wire reset,

    // input requests    
    input wire [NUM_REQS-1:0][`NUM_THREADS-1:0]                 req_valid_in,  
    input wire [NUM_REQS-1:0][TAG_IN_WIDTH-1:0]                 req_tag_in,   
    input wire [NUM_REQS-1:0][`NUM_THREADS-1:0][ADDR_WIDTH-1:0] req_addr_in, 
    input wire [NUM_REQS-1:0]                                   req_rw_in,  
    input wire [NUM_REQS-1:0][`NUM_THREADS-1:0][WORD_SIZE-1:0]  req_byteen_in,  
    input wire [NUM_REQS-1:0][`NUM_THREADS-1:0][WORD_WIDTH-1:0] req_data_in,  
    output wire [NUM_REQS-1:0]                                  req_ready_in,

    // output request
    output wire [`NUM_THREADS-1:0]                  req_valid_out,  
    output wire [TAG_OUT_WIDTH-1:0]                 req_tag_out,     
    output wire [`NUM_THREADS-1:0][ADDR_WIDTH-1:0]  req_addr_out, 
    output wire                                     req_rw_out,  
    output wire [`NUM_THREADS-1:0][WORD_SIZE-1:0]   req_byteen_out,
    output wire [`NUM_THREADS-1:0][WORD_WIDTH-1:0]  req_data_out,  
    input wire                                      req_ready_out,

    // input response
    input wire                                      rsp_valid_in,
    input wire [TAG_OUT_WIDTH-1:0]                  rsp_tag_in,
    input wire [WORD_WIDTH-1:0]                     rsp_data_in,
    output wire                                     rsp_ready_in,

    // output responses
    output wire [NUM_REQS-1:0]                      rsp_valid_out,
    output wire [NUM_REQS-1:0][TAG_IN_WIDTH-1:0]    rsp_tag_out,
    output wire [NUM_REQS-1:0][WORD_WIDTH-1:0]      rsp_data_out,
    input wire  [NUM_REQS-1:0]                      rsp_ready_out
);
    localparam DATAW = `NUM_THREADS + TAG_OUT_WIDTH + (`NUM_THREADS * ADDR_WIDTH) + 1 + (`NUM_THREADS * WORD_SIZE) + (`NUM_THREADS * WORD_WIDTH);

    if (NUM_REQS > 1) begin

        wire [NUM_REQS-1:0] valids;
        wire [NUM_REQS-1:0][DATAW-1:0] data_in;
        wire [`NUM_THREADS-1:0] req_tmask_out;
        wire req_valid_out_unqual;

        for (genvar i = 0; i < NUM_REQS; i++) begin
            assign valids[i]  = (| req_valid_in[i]);
            assign data_in[i] = {req_valid_in[i], {req_tag_in[i], LOG_NUM_REQS'(i)}, req_addr_in[i], req_rw_in[i], req_byteen_in[i], req_data_in[i]};
        end

        VX_stream_arbiter #(
            .NUM_REQS   (NUM_REQS),
            .DATAW      (DATAW),
            .IN_BUFFER  (NUM_REQS >= 4),
            .OUT_BUFFER (NUM_REQS >= 4)
        ) req_arb (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (valids), 
            .data_in    (data_in),
            .ready_in   (req_ready_in),
            .valid_out  (req_valid_out_unqual),
            .data_out   ({req_tmask_out, req_tag_out, req_addr_out, req_rw_out, req_byteen_out, req_data_out}),
            .ready_out  (req_ready_out)
        );

        assign req_valid_out = {`NUM_THREADS{req_valid_out_unqual}} & req_tmask_out;

        ///////////////////////////////////////////////////////////////////////

        wire [LOG_NUM_REQS-1:0] rsp_sel = rsp_tag_in[LOG_NUM_REQS-1:0];
        
        for (genvar i = 0; i < NUM_REQS; i++) begin                
            assign rsp_valid_out[i] = rsp_valid_in && (rsp_sel == LOG_NUM_REQS'(i));
            assign rsp_tag_out[i]   = rsp_tag_in[LOG_NUM_REQS +: TAG_IN_WIDTH];   
            assign rsp_data_out[i]  = rsp_data_in;           
        end
        
        assign rsp_ready_in = rsp_ready_out[rsp_sel];
        
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

        assign rsp_valid_out = rsp_valid_in;
        assign rsp_tag_out   = rsp_tag_in;
        assign rsp_data_out  = rsp_data_in;
        assign rsp_ready_in  = rsp_ready_out;

    end

endmodule