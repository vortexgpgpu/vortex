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
    if (NUM_REQUESTS > 1) begin       

        for (genvar i = 0; i < NUM_REQUESTS; i++) begin                
            assign csr_io_req_valid_out[i] = csr_io_req_valid_in && (request_id == `REQS_BITS'(i));
            assign csr_io_req_addr_out[i]  = csr_io_req_addr_in;
            assign csr_io_req_rw_out[i]    = csr_io_req_rw_in;
            assign csr_io_req_data_out[i]  = csr_io_req_data_in;            
        end

        assign csr_io_req_ready_in = csr_io_req_ready_out[request_id];

        ///////////////////////////////////////////////////////////////////////

        wire [REQS_BITS-1:0] rsp_idx;
        wire [NUM_REQUESTS-1:0] rsp_1hot;

        VX_rr_arbiter #(
            .N(NUM_REQUESTS)
        ) rsp_arb (
            .clk          (clk),
            .reset        (reset),
            .requests     (csr_io_rsp_valid_in),
            `UNUSED_PIN   (grant_valid),
            .grant_index  (rsp_idx),
            .grant_onehot (rsp_1hot)
        );   

        wire stall = ~csr_io_rsp_ready_out && csr_io_rsp_valid_out;

        VX_generic_register #(
            .N(1 + 32),
            .PASSTHRU(NUM_REQUESTS <= 2)
        ) pipe_reg (
            .clk   (clk),
            .reset (reset),
            .stall (stall),
            .flush (1'b0),
            .in    ({csr_io_rsp_valid_in[rsp_idx], csr_io_rsp_data_in[rsp_idx]}),
            .out   ({csr_io_rsp_valid_out,         csr_io_rsp_data_out})
        );  

        for (genvar i = 0; i < NUM_REQUESTS; i++) begin
            assign csr_io_rsp_ready_in[i] = rsp_1hot[i] && ~stall;
        end

    end else begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        `UNUSED_VAR (request_id)

        assign csr_io_req_valid_out = csr_io_req_valid_in;    
        assign csr_io_req_addr_out  = csr_io_req_addr_in;
        assign csr_io_req_rw_out    = csr_io_req_rw_in;    
        assign csr_io_req_data_out  = csr_io_req_data_in;
        assign csr_io_req_ready_in  = csr_io_req_ready_out;

        assign csr_io_rsp_valid_out = csr_io_rsp_valid_in;
        assign csr_io_rsp_data_out  = csr_io_rsp_data_in;
        assign csr_io_rsp_ready_in  = csr_io_rsp_ready_out;
        
    end

endmodule