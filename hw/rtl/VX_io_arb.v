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
    input wire [NUM_REQS-1:0][`NUM_THREADS-1:0]     io_req_valid_in,  
    input wire [NUM_REQS-1:0][TAG_IN_WIDTH-1:0]     io_req_tag_in,   
    input wire [NUM_REQS-1:0][`NUM_THREADS-1:0][ADDR_WIDTH-1:0] io_req_addr_in, 
    input wire [NUM_REQS-1:0]                       io_req_rw_in,  
    input wire [NUM_REQS-1:0][`NUM_THREADS-1:0][WORD_SIZE-1:0] io_req_byteen_in,  
    input wire [NUM_REQS-1:0][`NUM_THREADS-1:0][WORD_WIDTH-1:0] io_req_data_in,  
    output wire [NUM_REQS-1:0]                      io_req_ready_in,

    // output request
    output wire [`NUM_THREADS-1:0]                  io_req_valid_out,  
    output wire [TAG_OUT_WIDTH-1:0]                 io_req_tag_out,     
    output wire [`NUM_THREADS-1:0][ADDR_WIDTH-1:0]  io_req_addr_out, 
    output wire                                     io_req_rw_out,  
    output wire [`NUM_THREADS-1:0][WORD_SIZE-1:0]   io_req_byteen_out,
    output wire [`NUM_THREADS-1:0][WORD_WIDTH-1:0]  io_req_data_out,  
    input wire                                      io_req_ready_out,

    // input response
    output wire [NUM_REQS-1:0]                      io_rsp_valid_in,
    output wire [NUM_REQS-1:0][TAG_IN_WIDTH-1:0]    io_rsp_tag_in,
    output wire [NUM_REQS-1:0][WORD_WIDTH-1:0]      io_rsp_data_in,
    input wire  [NUM_REQS-1:0]                      io_rsp_ready_in,

    // output response
    input wire                                      io_rsp_valid_out,
    input wire [TAG_OUT_WIDTH-1:0]                  io_rsp_tag_out,
    input wire [WORD_WIDTH-1:0]                     io_rsp_data_out,
    output wire                                     io_rsp_ready_out
);
    wire [NUM_REQS-1:0] valids;
    for (genvar i = 0; i < NUM_REQS; i++) begin
        assign valids[i] = (| io_req_valid_in[i]);
    end
    
    wire [NUM_REQS-1:0][(`NUM_THREADS + TAG_OUT_WIDTH + (`NUM_THREADS * ADDR_WIDTH) + 1 + (`NUM_THREADS * WORD_SIZE) + (`NUM_THREADS * WORD_WIDTH))-1:0] data_in;
    for (genvar i = 0; i < NUM_REQS; i++) begin
        assign data_in[i] = {{io_req_valid_in[i], io_req_tag_in[i], REQS_BITS'(i)}, io_req_addr_in[i], io_req_rw_in[i], io_req_byteen_in[i], io_req_data_in[i]};
    end

    wire [`NUM_THREADS-1:0] io_req_tmask_out;
    wire io_req_valid_out_unqual;

    VX_stream_arbiter #(
        .NUM_REQS(NUM_REQS),
        .DATAW(`NUM_THREADS + TAG_OUT_WIDTH + (`NUM_THREADS * ADDR_WIDTH) + 1 + (`NUM_THREADS * WORD_SIZE) + (`NUM_THREADS * WORD_WIDTH)),
        .BUFFERED(NUM_REQS >= 4)
    ) req_arb (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (valids), 
        .valid_out  (io_req_valid_out_unqual),
        .data_in    (data_in),                        
        .data_out   ({io_req_tmask_out, io_req_tag_out, io_req_addr_out, io_req_rw_out, io_req_byteen_out, io_req_data_out}),  
        .ready_in   (io_req_ready_in),
        .ready_out  (io_req_ready_out)
    );

    assign io_req_valid_out = {`NUM_THREADS{io_req_valid_out_unqual}} & io_req_tmask_out;

    ///////////////////////////////////////////////////////////////////////

    if (NUM_REQS > 1) begin

        wire [REQS_BITS-1:0] rsp_sel = io_rsp_tag_out[REQS_BITS-1:0];
        
        for (genvar i = 0; i < NUM_REQS; i++) begin                
            assign io_rsp_valid_in[i] = io_rsp_valid_out && (rsp_sel == REQS_BITS'(i));
            assign io_rsp_tag_in[i]   = io_rsp_tag_out[REQS_BITS +: TAG_IN_WIDTH];   
            assign io_rsp_data_in[i]  = io_rsp_data_out;           
        end
        
        assign io_rsp_ready_out = io_rsp_ready_in[rsp_sel];
        
    end else begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        assign io_rsp_valid_in   = io_rsp_valid_out;
        assign io_rsp_tag_in     = io_rsp_tag_out;
        assign io_rsp_data_in    = io_rsp_data_out;
        assign io_rsp_ready_out  = io_rsp_ready_in;

    end

endmodule