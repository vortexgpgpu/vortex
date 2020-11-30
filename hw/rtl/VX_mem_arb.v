`include "VX_define.vh"

module VX_mem_arb #(    
    parameter NUM_REQUESTS  = 1, 
    parameter DATA_WIDTH    = 1,
    parameter TAG_IN_WIDTH  = 1,
    parameter TAG_OUT_WIDTH = 1,
    
    parameter DATA_SIZE  = (DATA_WIDTH / 8), 
    parameter ADDR_WIDTH = 32 - `CLOG2(DATA_SIZE),
    parameter REQS_BITS  = `CLOG2(NUM_REQUESTS)
) (
    input wire clk,
    input wire reset,

    // input requests    
    input wire [NUM_REQUESTS-1:0]                       req_valid_in,    
    input wire [NUM_REQUESTS-1:0][TAG_IN_WIDTH-1:0]     req_tag_in,  
    input wire [NUM_REQUESTS-1:0][ADDR_WIDTH-1:0]       req_addr_in,
    input wire [NUM_REQUESTS-1:0]                       req_rw_in,  
    input wire [NUM_REQUESTS-1:0][DATA_SIZE-1:0]        req_byteen_in,  
    input wire [NUM_REQUESTS-1:0][DATA_WIDTH-1:0]       req_data_in,  
    output wire [NUM_REQUESTS-1:0]                      req_ready_in,

    // input response
    output wire [NUM_REQUESTS-1:0]                      rsp_valid_out,
    output wire [NUM_REQUESTS-1:0][TAG_IN_WIDTH-1:0]    rsp_tag_out,
    output wire [NUM_REQUESTS-1:0][DATA_WIDTH-1:0]      rsp_data_out,
    input wire  [NUM_REQUESTS-1:0]                      rsp_ready_out,

    // output request
    output wire                         req_valid_out,
    output wire [TAG_OUT_WIDTH-1:0]     req_tag_out,   
    output wire [ADDR_WIDTH-1:0]        req_addr_out, 
    output wire                         req_rw_out,  
    output wire [DATA_SIZE-1:0]         req_byteen_out,  
    output wire [DATA_WIDTH-1:0]        req_data_out,    
    input wire                          req_ready_out,

    // output response
    input wire                          rsp_valid_in,
    input wire [TAG_OUT_WIDTH-1:0]      rsp_tag_in,
    input wire [DATA_WIDTH-1:0]         rsp_data_in,
    output wire                         rsp_ready_in
);
    if (NUM_REQUESTS > 1) begin

        wire [REQS_BITS-1:0] req_idx;        
        wire [NUM_REQUESTS-1:0] req_1hot;

        VX_rr_arbiter #(
            .N(NUM_REQUESTS)
        ) req_arb (
            .clk          (clk),
            .reset        (reset),
            .requests     (req_valid_in),            
            `UNUSED_PIN   (grant_valid),
            .grant_index  (req_idx),
            .grant_onehot (req_1hot)
        );

        wire stall = ~req_ready_out && req_valid_out;

        VX_generic_register #(
            .N(1 + TAG_OUT_WIDTH + ADDR_WIDTH + 1 + DATA_SIZE + DATA_WIDTH),
            .R(1),
            .PASSTHRU(NUM_REQUESTS <= 2)
        ) pipe_reg (
            .clk   (clk),
            .reset (reset),
            .stall (stall),
            .flush (1'b0),
            .in    ({req_valid_in[req_idx],  {req_tag_in[req_idx], REQS_BITS'(req_idx)}, req_addr_in[req_idx], req_rw_in[req_idx], req_byteen_in[req_idx], req_data_in[req_idx]}),
            .out   ({req_valid_out,          req_tag_out,                                req_addr_out,         req_rw_out,         req_byteen_out,         req_data_out})
        );

        for (genvar i = 0; i < NUM_REQUESTS; i++) begin
            assign req_ready_in[i] = req_1hot[i] && ~stall;
        end

        ///////////////////////////////////////////////////////////////////////

        wire [REQS_BITS-1:0] rsp_sel = rsp_tag_in[REQS_BITS-1:0];
        
        for (genvar i = 0; i < NUM_REQUESTS; i++) begin                
            assign rsp_valid_out[i] = rsp_valid_in && (rsp_sel == REQS_BITS'(i));
            assign rsp_tag_out[i]   = rsp_tag_in[REQS_BITS +: TAG_IN_WIDTH];        
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

        assign rsp_valid_out   = rsp_valid_in;
        assign rsp_tag_out     = rsp_tag_in;
        assign rsp_data_out    = rsp_data_in;
        assign rsp_ready_in  = rsp_ready_out;

    end

endmodule