`include "VX_define.vh"

module VX_csr_arb #(
    parameter NUM_REQS     = 1,     
    parameter DATA_WIDTH   = 1,
    parameter BUFFERED_REQ = 0,
    parameter BUFFERED_RSP = 0,
    
    parameter DATA_SIZE    = (DATA_WIDTH / 8), 
    parameter ADDR_WIDTH   = 32 - `CLOG2(DATA_SIZE),
    parameter LOG_NUM_REQS = `LOG2UP(NUM_REQS)
) (
    input wire clk,
    input wire reset,
        
    input wire [LOG_NUM_REQS-1:0]   request_id,

    // input requests    
    input wire                      req_valid_in,
    input wire [ADDR_WIDTH-1:0]     req_addr_in,
    input wire                      req_rw_in,
    input wire [DATA_WIDTH-1:0]     req_data_in,
    output wire                     req_ready_in,

    // output request
    output wire [NUM_REQS-1:0]                 req_valid_out,
    output wire [NUM_REQS-1:0][ADDR_WIDTH-1:0] req_addr_out,
    output wire [NUM_REQS-1:0]                 req_rw_out,
    output wire [NUM_REQS-1:0][DATA_WIDTH-1:0] req_data_out,
    input wire [NUM_REQS-1:0]                  req_ready_out,

    // input response
    input wire [NUM_REQS-1:0]                 rsp_valid_in,
    input wire [NUM_REQS-1:0][DATA_WIDTH-1:0] rsp_data_in,
    output wire [NUM_REQS-1:0]                rsp_ready_in,   

    // output response
    output wire                     rsp_valid_out,
    output wire [DATA_WIDTH-1:0]    rsp_data_out,
    input wire                      rsp_ready_out
);
    localparam REQ_DATAW = ADDR_WIDTH + 1 + DATA_WIDTH;
    localparam RSP_DATAW = DATA_WIDTH;

    wire [NUM_REQS-1:0][REQ_DATAW-1:0] req_merged_data_out;
    for (genvar i = 0; i < NUM_REQS; i++) begin
        assign {req_addr_out[i], req_rw_out[i], req_data_out[i]} = req_merged_data_out[i];
    end

    VX_stream_demux #(
        .NUM_REQS (NUM_REQS),
        .DATAW    (REQ_DATAW),
        .BUFFERED (BUFFERED_REQ)
    ) req_demux (
        .clk       (clk),
        .reset     (reset),
        .sel       (request_id),
        .valid_in  (req_valid_in),
        .data_in   ({req_addr_in, req_rw_in, req_data_in}),
        .ready_in  (req_ready_in),
        .valid_out (req_valid_out),
        .data_out  (req_merged_data_out),
        .ready_out (req_ready_out)
    );

    VX_stream_arbiter #(
        .NUM_REQS (NUM_REQS),
        .DATAW    (RSP_DATAW),
        .BUFFERED (BUFFERED_RSP),
        .TYPE     ("X") // fixed arbitration
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rsp_valid_in),
        .data_in   (rsp_data_in),
        .ready_in  (rsp_ready_in),
        .valid_out (rsp_valid_out),
        .data_out  (rsp_data_out),
        .ready_out (rsp_ready_out)
    );

endmodule
