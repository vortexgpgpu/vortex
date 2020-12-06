`include "VX_define.vh"

module VX_io_arb #(    
    parameter NUM_REQS      = 1, 
    parameter WORD_SIZE     = 1, 
    parameter TAG_IN_WIDTH  = 1,
    parameter TAG_OUT_WIDTH = 1,

    parameter WORD_WIDTH = WORD_SIZE * 8,
    parameter ADDR_WIDTH = 32 - `CLOG2(WORD_SIZE),
    parameter REQS_BITS  = `CLOG2(NUM_REQS)
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
    output wire [NUM_REQS-1:0]                      rsp_valid_in,
    output wire [NUM_REQS-1:0][TAG_IN_WIDTH-1:0]    rsp_tag_in,
    output wire [NUM_REQS-1:0][WORD_WIDTH-1:0]      rsp_data_in,
    input wire  [NUM_REQS-1:0]                      rsp_ready_in,

    // output response
    input wire                                      rsp_valid_out,
    input wire [TAG_OUT_WIDTH-1:0]                  rsp_tag_out,
    input wire [WORD_WIDTH-1:0]                     rsp_data_out,
    output wire                                     rsp_ready_out
);
    localparam DATAW = `NUM_THREADS + TAG_OUT_WIDTH + (`NUM_THREADS * ADDR_WIDTH) + 1 + (`NUM_THREADS * WORD_SIZE) + (`NUM_THREADS * WORD_WIDTH);

    if (NUM_REQS > 1) begin

        wire [NUM_REQS-1:0] valids;
        for (genvar i = 0; i < NUM_REQS; i++) begin
            assign valids[i] = (| req_valid_in[i]);
        end
        
        wire [NUM_REQS-1:0][DATAW-1:0] data_in;
        for (genvar i = 0; i < NUM_REQS; i++) begin
            assign data_in[i] = {req_valid_in[i], {req_tag_in[i], REQS_BITS'(i)}, req_addr_in[i], req_rw_in[i], req_byteen_in[i], req_data_in[i]};
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
                .valid_in  (valids[i]),        
                .data_in   (data_in[i]),
                .ready_in  (req_ready_in[i]),        
                .valid_out (req_valid_in_qual[i]),
                .data_out  (req_data_in_qual[i]),
                .ready_out (req_ready_in_qual[i])
            );
        end

        wire [`NUM_THREADS-1:0] req_tmask_out;
        wire req_valid_out_unqual;

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
            .valid_out  (req_valid_out_unqual),
            .data_out   ({req_tmask_out, req_tag_out, req_addr_out, req_rw_out, req_byteen_out, req_data_out}),
            .ready_out  (req_ready_out)
        );

        assign req_valid_out = {`NUM_THREADS{req_valid_out_unqual}} & req_tmask_out;

        ///////////////////////////////////////////////////////////////////////

        wire [REQS_BITS-1:0] rsp_sel = rsp_tag_out[REQS_BITS-1:0];
        
        for (genvar i = 0; i < NUM_REQS; i++) begin                
            assign rsp_valid_in[i] = rsp_valid_out && (rsp_sel == REQS_BITS'(i));
            assign rsp_tag_in[i]   = rsp_tag_out[REQS_BITS +: TAG_IN_WIDTH];   
            assign rsp_data_in[i]  = rsp_data_out;           
        end
        
        assign rsp_ready_out = rsp_ready_in[rsp_sel];
        
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

        assign rsp_valid_in   = rsp_valid_out;
        assign rsp_tag_in     = rsp_tag_out;
        assign rsp_data_in    = rsp_data_out;
        assign rsp_ready_out  = rsp_ready_in;

    end

endmodule