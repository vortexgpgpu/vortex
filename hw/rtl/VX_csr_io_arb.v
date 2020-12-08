`include "VX_define.vh"

module VX_csr_io_arb #(
    parameter NUM_REQS   = 1,     
    parameter DATA_WIDTH = 1,
    
    parameter DATA_SIZE    = (DATA_WIDTH / 8), 
    parameter ADDR_WIDTH   = 32 - `CLOG2(DATA_SIZE),
    parameter LOG_NUM_REQS = `LOG2UP(NUM_REQS)
) (
    input wire                          clk,
    input wire                          reset,
        
    input wire [LOG_NUM_REQS-1:0]       request_id,

    // input requests    
    input wire                          req_valid_in,
    input wire [ADDR_WIDTH-1:0]         req_addr_in,
    input wire                          req_rw_in,
    input wire [DATA_WIDTH-1:0]         req_data_in,
    output wire                         req_ready_in,

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
    output wire                         rsp_valid_out,
    output wire [DATA_WIDTH-1:0]        rsp_data_out,
    input wire                          rsp_ready_out
);
    if (NUM_REQS > 1) begin       

        for (genvar i = 0; i < NUM_REQS; i++) begin                
            assign req_valid_out[i] = req_valid_in && (request_id == LOG_NUM_REQS'(i));
            assign req_addr_out[i]  = req_addr_in;
            assign req_rw_out[i]    = req_rw_in;
            assign req_data_out[i]  = req_data_in;            
        end

        assign req_ready_in = req_ready_out[request_id];

    end else begin

        `UNUSED_VAR (request_id)

        assign req_valid_out = req_valid_in;    
        assign req_addr_out  = req_addr_in;
        assign req_rw_out    = req_rw_in;    
        assign req_data_out  = req_data_in;
        assign req_ready_in  = req_ready_out;
        
    end

    ///////////////////////////////////////////////////////////////////////

    VX_stream_arbiter #(
        .NUM_REQS   (NUM_REQS),
        .DATAW      (DATA_WIDTH),
        .IN_BUFFER  (NUM_REQS >= 4),
        .OUT_BUFFER (NUM_REQS >= 4)
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