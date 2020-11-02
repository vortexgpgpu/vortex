`include "VX_define.vh"

module VX_io_arb #(    
    parameter NUM_REQUESTS  = 1, 
    parameter WORD_SIZE     = 1, 
    parameter TAG_IN_WIDTH  = 1,
    parameter TAG_OUT_WIDTH = 1,

    parameter WORD_WIDTH = WORD_SIZE * 8,
    parameter ADDR_WIDTH = 32 - `CLOG2(WORD_SIZE),
    parameter REQS_BITS  = `CLOG2(NUM_REQUESTS)
) (
    input wire clk,
    input wire reset,

    // input requests    
    input wire [NUM_REQUESTS-1:0][`NUM_THREADS-1:0]     io_req_valid_in,
    input wire [NUM_REQUESTS-1:0]                       io_req_rw_in,  
    input wire [NUM_REQUESTS-1:0][`NUM_THREADS-1:0][WORD_SIZE-1:0]  io_req_byteen_in,  
    input wire [NUM_REQUESTS-1:0][`NUM_THREADS-1:0][ADDR_WIDTH-1:0] io_req_addr_in,
    input wire [NUM_REQUESTS-1:0][`NUM_THREADS-1:0][WORD_WIDTH-1:0] io_req_data_in,    
    input wire [NUM_REQUESTS-1:0][TAG_IN_WIDTH-1:0]     io_req_tag_in,    
    output wire [NUM_REQUESTS-1:0]                      io_req_ready_in,

    // input response
    output wire [NUM_REQUESTS-1:0]                      io_rsp_valid_in,
    output wire [NUM_REQUESTS-1:0][WORD_WIDTH-1:0]      io_rsp_data_in,
    output wire [NUM_REQUESTS-1:0][TAG_IN_WIDTH-1:0]    io_rsp_tag_in,
    input wire  [NUM_REQUESTS-1:0]                      io_rsp_ready_in,

    // output request
    output wire [`NUM_THREADS-1:0]                      io_req_valid_out,
    output wire                                         io_req_rw_out,  
    output wire [`NUM_THREADS-1:0][WORD_SIZE-1:0]       io_req_byteen_out,  
    output wire [`NUM_THREADS-1:0][ADDR_WIDTH-1:0]      io_req_addr_out,
    output wire [`NUM_THREADS-1:0][WORD_WIDTH-1:0]      io_req_data_out,    
    output wire [TAG_OUT_WIDTH-1:0]                     io_req_tag_out,    
    input wire                                          io_req_ready_out,

    // output response
    input wire                                          io_rsp_valid_out,
    input wire [WORD_WIDTH-1:0]                         io_rsp_data_out,
    input wire [TAG_OUT_WIDTH-1:0]                      io_rsp_tag_out,
    output wire                                         io_rsp_ready_out
);
    if (NUM_REQUESTS == 1) begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        assign io_req_valid_out  = io_req_valid_in;
        assign io_req_rw_out     = io_req_rw_in;
        assign io_req_byteen_out = io_req_byteen_in;
        assign io_req_addr_out   = io_req_addr_in;
        assign io_req_data_out   = io_req_data_in;
        assign io_req_tag_out    = io_req_tag_in;
        assign io_req_ready_in   = io_req_ready_out;

        assign io_rsp_valid_in   = io_rsp_valid_out;
        assign io_rsp_data_in    = io_rsp_data_out;
        assign io_rsp_tag_in     = io_rsp_tag_out;
        assign io_rsp_ready_out  = io_rsp_ready_in;

    end else begin

        reg [REQS_BITS-1:0] bus_req_sel;

        wire [NUM_REQUESTS-1:0] valid_requests;

        for (genvar i = 0; i < NUM_REQUESTS; i++) begin
            assign valid_requests[i] = (| io_req_valid_in[i]);
        end

        VX_rr_arbiter #(
            .N(NUM_REQUESTS)
        ) arbiter (
            .clk         (clk),
            .reset       (reset),
            .requests    (valid_requests),
            .grant_index (bus_req_sel),
            `UNUSED_PIN  (grant_valid),
            `UNUSED_PIN  (grant_onehot)
        );

        assign io_req_valid_out  = io_req_valid_in [bus_req_sel];
        assign io_req_rw_out     = io_req_rw_in   [bus_req_sel];
        assign io_req_byteen_out = io_req_byteen_in [bus_req_sel];
        assign io_req_addr_out   = io_req_addr_in [bus_req_sel];
        assign io_req_data_out   = io_req_data_in [bus_req_sel];
        assign io_req_tag_out    = {io_req_tag_in [bus_req_sel], REQS_BITS'(bus_req_sel)};

        for (genvar i = 0; i < NUM_REQUESTS; i++) begin
            assign io_req_ready_in[i] = io_req_ready_out && (bus_req_sel == REQS_BITS'(i));
        end

        wire [REQS_BITS-1:0] bus_rsp_sel = io_rsp_tag_out[REQS_BITS-1:0];
        
        for (genvar i = 0; i < NUM_REQUESTS; i++) begin                
            assign io_rsp_valid_in[i] = io_rsp_valid_out && (bus_rsp_sel == REQS_BITS'(i));
            assign io_rsp_data_in[i]  = io_rsp_data_out;
            assign io_rsp_tag_in[i]   = io_rsp_tag_out[REQS_BITS +: TAG_IN_WIDTH];              
        end
        assign io_rsp_ready_out = io_rsp_ready_in[bus_rsp_sel];

    end

endmodule