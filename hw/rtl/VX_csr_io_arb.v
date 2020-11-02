`include "VX_define.vh"

module VX_csr_io_arb #(
    parameter NUM_REQUESTS = 1,
    parameter REQS_BITS = `LOG2UP(NUM_REQUESTS)
) (
    input wire                              clk,
    input wire                              reset,
        
    input wire [REQS_BITS-1:0]              request_id,

    // input requests    
    input wire                              csr_io_req_valid_in,
    input wire [11:0]                       csr_io_req_addr_in,
    input wire                              csr_io_req_rw_in,
    input wire [31:0]                       csr_io_req_data_in,
    output wire                             csr_io_req_ready_in,

    // input response
    input wire [NUM_REQUESTS-1:0]           csr_io_rsp_valid_in,
    input wire [NUM_REQUESTS-1:0][31:0]     csr_io_rsp_data_in,
    output wire [NUM_REQUESTS-1:0]          csr_io_rsp_ready_in,

    // output request
    output wire [NUM_REQUESTS-1:0]          csr_io_req_valid_out,
    output wire [NUM_REQUESTS-1:0][11:0]    csr_io_req_addr_out,
    output wire [NUM_REQUESTS-1:0]          csr_io_req_rw_out,
    output wire [NUM_REQUESTS-1:0][31:0]    csr_io_req_data_out,
    input wire [NUM_REQUESTS-1:0]           csr_io_req_ready_out,

    // output response
    output wire                             csr_io_rsp_valid_out,
    output wire [31:0]                      csr_io_rsp_data_out,
    input wire                              csr_io_rsp_ready_out
);
    if (NUM_REQUESTS == 1) begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (request_id)

        assign csr_io_req_valid_out  = csr_io_req_valid_in;
        assign csr_io_req_rw_out     = csr_io_req_rw_in;        
        assign csr_io_req_addr_out   = csr_io_req_addr_in;
        assign csr_io_req_data_out   = csr_io_req_data_in;
        assign csr_io_req_ready_in   = csr_io_req_ready_out;

        assign csr_io_rsp_valid_out  = csr_io_rsp_valid_in;
        assign csr_io_rsp_data_out   = csr_io_rsp_data_in;
        assign csr_io_rsp_ready_in   = csr_io_rsp_ready_out;

    end else begin

        for (genvar i = 0; i < NUM_REQUESTS; i++) begin                
            assign csr_io_req_valid_out[i]  = csr_io_req_valid_in && (request_id == `REQS_BITS'(i));
            assign csr_io_req_rw_out[i]     = csr_io_req_rw_in;
            assign csr_io_req_addr_out[i]   = csr_io_req_addr_in;
            assign csr_io_req_data_out[i]   = csr_io_req_data_in;            
        end

        assign csr_io_req_ready_in = csr_io_req_ready_out[request_id];

        reg [REQS_BITS-1:0] bus_rsp_sel;

        VX_fixed_arbiter #(
            .N(NUM_REQUESTS)
        ) arbiter (
            .clk         (clk),
            .reset       (reset),
            .requests    (csr_io_rsp_valid_in),
            .grant_index (bus_rsp_sel),
            `UNUSED_PIN  (grant_valid),
            `UNUSED_PIN  (grant_onehot)
        );

        assign csr_io_rsp_valid_out = csr_io_rsp_valid_in [bus_rsp_sel];
        assign csr_io_rsp_data_out  = csr_io_rsp_data_in [bus_rsp_sel];       

        for (genvar i = 0; i < NUM_REQUESTS; i++) begin
            assign csr_io_rsp_ready_in[i] = csr_io_rsp_ready_out && (bus_rsp_sel == `REQS_BITS'(i));
        end
        
    end

endmodule