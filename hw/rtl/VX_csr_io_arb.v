`include "VX_define.vh"

module VX_csr_io_arb #(
    parameter NUM_REQS   = 1,     
    parameter DATA_WIDTH = 1,
    
    parameter DATA_SIZE  = (DATA_WIDTH / 8), 
    parameter ADDR_WIDTH = 32 - `CLOG2(DATA_SIZE),
    parameter REQS_BITS  = `LOG2UP(NUM_REQS)
) (
    input wire                          clk,
    input wire                          reset,
        
    input wire [REQS_BITS-1:0]          request_id,

    // input requests    
    input wire                          csr_io_req_valid_in,
    input wire [ADDR_WIDTH-1:0]         csr_io_req_addr_in,
    input wire                          csr_io_req_rw_in,
    input wire [DATA_WIDTH-1:0]         csr_io_req_data_in,
    output wire                         csr_io_req_ready_in,

    // output request
    output wire [NUM_REQS-1:0]          csr_io_req_valid_out,
    output wire [NUM_REQS-1:0][ADDR_WIDTH-1:0] csr_io_req_addr_out,
    output wire [NUM_REQS-1:0]          csr_io_req_rw_out,
    output wire [NUM_REQS-1:0][DATA_WIDTH-1:0] csr_io_req_data_out,
    input wire [NUM_REQS-1:0]           csr_io_req_ready_out,

    // input response
    input wire [NUM_REQS-1:0]           csr_io_rsp_valid_in,
    input wire [NUM_REQS-1:0][DATA_WIDTH-1:0] csr_io_rsp_data_in,
    output wire [NUM_REQS-1:0]          csr_io_rsp_ready_in,

    // output response
    output wire                         csr_io_rsp_valid_out,
    output wire [DATA_WIDTH-1:0]        csr_io_rsp_data_out,
    input wire                          csr_io_rsp_ready_out
);
    if (NUM_REQS > 1) begin       

        for (genvar i = 0; i < NUM_REQS; i++) begin                
            assign csr_io_req_valid_out[i] = csr_io_req_valid_in && (request_id == `REQS_BITS'(i));
            assign csr_io_req_addr_out[i]  = csr_io_req_addr_in;
            assign csr_io_req_rw_out[i]    = csr_io_req_rw_in;
            assign csr_io_req_data_out[i]  = csr_io_req_data_in;            
        end

        assign csr_io_req_ready_in = csr_io_req_ready_out[request_id];

    end else begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (request_id)

        assign csr_io_req_valid_out = csr_io_req_valid_in;    
        assign csr_io_req_addr_out  = csr_io_req_addr_in;
        assign csr_io_req_rw_out    = csr_io_req_rw_in;    
        assign csr_io_req_data_out  = csr_io_req_data_in;
        assign csr_io_req_ready_in  = csr_io_req_ready_out;
        
    end

    ///////////////////////////////////////////////////////////////////////

    VX_stream_arbiter #(
        .NUM_REQS(NUM_REQS),
        .DATAW(DATA_WIDTH),
        .BUFFERED(NUM_REQS >= 4)
    ) rsp_arb (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (csr_io_rsp_valid_in),
        .valid_out  (csr_io_rsp_valid_out),
        .data_in    (csr_io_rsp_data_in),
        .data_out   (csr_io_rsp_data_out),
        .ready_in   (csr_io_rsp_ready_in),
        .ready_out  (csr_io_rsp_ready_out)
    );

endmodule