`include "VX_define.vh"

module VX_csr_io_arb #(
    parameter NUM_REQUESTS = 1,
    parameter REQS_BITS = `LOG2UP(NUM_REQUESTS)
) (
    input wire                              clk,
    input wire                              reset,
        
    input wire [REQS_BITS-1:0]              request_id,

    // input requests    
    input wire                              in_csr_io_req_valid,
    input wire [11:0]                       in_csr_io_req_addr,
    input wire                              in_csr_io_req_rw,
    input wire [31:0]                       in_csr_io_req_data,
    output wire                             in_csr_io_req_ready,

    // input response
    input wire [NUM_REQUESTS-1:0]           in_csr_io_rsp_valid,
    input wire [NUM_REQUESTS-1:0][31:0]     in_csr_io_rsp_data,
    output wire [NUM_REQUESTS-1:0]          in_csr_io_rsp_ready,

    // output request
    output wire [NUM_REQUESTS-1:0]          out_csr_io_req_valid,
    output wire [NUM_REQUESTS-1:0][11:0]    out_csr_io_req_addr,
    output wire [NUM_REQUESTS-1:0]          out_csr_io_req_rw,
    output wire [NUM_REQUESTS-1:0][31:0]    out_csr_io_req_data,
    input wire [NUM_REQUESTS-1:0]           out_csr_io_req_ready,

    // output response
    output wire                             out_csr_io_rsp_valid,
    output wire [31:0]                      out_csr_io_rsp_data,
    input wire                              out_csr_io_rsp_ready
);
    if (NUM_REQUESTS == 1) begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (request_id)

        assign out_csr_io_req_valid  = in_csr_io_req_valid;
        assign out_csr_io_req_rw     = in_csr_io_req_rw;        
        assign out_csr_io_req_addr   = in_csr_io_req_addr;
        assign out_csr_io_req_data   = in_csr_io_req_data;
        assign in_csr_io_req_ready   = out_csr_io_req_ready;

        assign out_csr_io_rsp_valid  = in_csr_io_rsp_valid;
        assign out_csr_io_rsp_data   = in_csr_io_rsp_data;
        assign in_csr_io_rsp_ready   = out_csr_io_rsp_ready;

    end else begin

        genvar i;

        for (i = 0; i < NUM_REQUESTS; i++) begin                
            assign out_csr_io_req_valid[i]  = in_csr_io_req_valid && (request_id == `REQS_BITS'(i));
            assign out_csr_io_req_rw[i]     = in_csr_io_req_rw;
            assign out_csr_io_req_addr[i]   = in_csr_io_req_addr;
            assign out_csr_io_req_data[i]   = in_csr_io_req_data;            
        end

        assign in_csr_io_req_ready = out_csr_io_req_ready[request_id];

        reg [REQS_BITS-1:0] bus_rsp_sel;

        VX_fixed_arbiter #(
            .N(NUM_REQUESTS)
        ) arbiter (
            .clk         (clk),
            .reset       (reset),
            .requests    (in_csr_io_rsp_valid),
            .grant_index (bus_rsp_sel),
            `UNUSED_PIN  (grant_valid),
            `UNUSED_PIN  (grant_onehot)
        );

        assign out_csr_io_rsp_valid = in_csr_io_rsp_valid [bus_rsp_sel];
        assign out_csr_io_rsp_data  = in_csr_io_rsp_data [bus_rsp_sel];       

        for (i = 0; i < NUM_REQUESTS; i++) begin
            assign in_csr_io_rsp_ready[i] = out_csr_io_rsp_ready && (bus_rsp_sel == `REQS_BITS'(i));
        end
        
    end

endmodule