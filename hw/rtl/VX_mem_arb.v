`include "VX_define.vh"

module VX_mem_arb #(    
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
    input wire [NUM_REQUESTS-1:0]                       mem_req_valid_in,    
    input wire [NUM_REQUESTS-1:0][TAG_IN_WIDTH-1:0]     mem_req_tag_in,  
    input wire [NUM_REQUESTS-1:0][ADDR_WIDTH-1:0]       mem_req_addr_in,
    input wire [NUM_REQUESTS-1:0]                       mem_req_rw_in,  
    input wire [NUM_REQUESTS-1:0][WORD_SIZE-1:0]        mem_req_byteen_in,  
    input wire [NUM_REQUESTS-1:0][WORD_WIDTH-1:0]       mem_req_data_in,  
    output wire [NUM_REQUESTS-1:0]                      mem_req_ready_in,

    // input response
    output wire [NUM_REQUESTS-1:0]                      mem_rsp_valid_in,
    output wire [NUM_REQUESTS-1:0][TAG_IN_WIDTH-1:0]    mem_rsp_tag_in,
    output wire [NUM_REQUESTS-1:0][WORD_WIDTH-1:0]      mem_rsp_data_in,
    input wire  [NUM_REQUESTS-1:0]                      mem_rsp_ready_in,

    // output request
    output wire                         mem_req_valid_out,
    output wire [TAG_OUT_WIDTH-1:0]     mem_req_tag_out,   
    output wire [ADDR_WIDTH-1:0]        mem_req_addr_out, 
    output wire                         mem_req_rw_out,  
    output wire [WORD_SIZE-1:0]         mem_req_byteen_out,  
    output wire [WORD_WIDTH-1:0]        mem_req_data_out,    
    input wire                          mem_req_ready_out,

    // output response
    input wire                          mem_rsp_valid_out,
    input wire [TAG_OUT_WIDTH-1:0]      mem_rsp_tag_out,
    input wire [WORD_WIDTH-1:0]         mem_rsp_data_out,
    output wire                         mem_rsp_ready_out
);
    if (NUM_REQUESTS > 1) begin

        wire [REQS_BITS-1:0] req_idx;        
        wire [NUM_REQUESTS-1:0] req_1hot;

        VX_rr_arbiter #(
            .N(NUM_REQUESTS)
        ) req_arb (
            .clk          (clk),
            .reset        (reset),
            .requests     (mem_req_valid_in),            
            `UNUSED_PIN   (grant_valid),
            .grant_index  (req_idx),
            .grant_onehot (req_1hot)
        );

        wire stall = ~mem_req_ready_out && mem_req_valid_out;

        VX_generic_register #(
            .N(1 + TAG_OUT_WIDTH + ADDR_WIDTH + 1 + WORD_SIZE + WORD_WIDTH),
            .PASSTHRU(NUM_REQUESTS <= 2)
        ) pipe_reg (
            .clk   (clk),
            .reset (reset),
            .stall (stall),
            .flush (1'b0),
            .in    ({mem_req_valid_in[req_idx],  {mem_req_tag_in[req_idx], REQS_BITS'(req_idx)}, mem_req_addr_in[req_idx], mem_req_rw_in[req_idx], mem_req_byteen_in[req_idx], mem_req_data_in[req_idx]}),
            .out   ({mem_req_valid_out,          mem_req_tag_out,                                mem_req_addr_out,         mem_req_rw_out,         mem_req_byteen_out,         mem_req_data_out})
        );

        for (genvar i = 0; i < NUM_REQUESTS; i++) begin
            assign mem_req_ready_in[i] = req_1hot[i] && ~stall;
        end

        ///////////////////////////////////////////////////////////////////////

        wire [REQS_BITS-1:0] rsp_sel = mem_rsp_tag_out[REQS_BITS-1:0];
        
        for (genvar i = 0; i < NUM_REQUESTS; i++) begin                
            assign mem_rsp_valid_in[i] = mem_rsp_valid_out && (rsp_sel == REQS_BITS'(i));
            assign mem_rsp_tag_in[i]   = mem_rsp_tag_out[REQS_BITS +: TAG_IN_WIDTH];        
            assign mem_rsp_data_in[i]  = mem_rsp_data_out;      
        end
        
        assign mem_rsp_ready_out = mem_rsp_ready_in[rsp_sel];        

    end else begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        assign mem_req_valid_out  = mem_req_valid_in;
        assign mem_req_tag_out    = mem_req_tag_in;
        assign mem_req_addr_out   = mem_req_addr_in;
        assign mem_req_rw_out     = mem_req_rw_in;
        assign mem_req_byteen_out = mem_req_byteen_in;
        assign mem_req_data_out   = mem_req_data_in;
        assign mem_req_ready_in   = mem_req_ready_out;

        assign mem_rsp_valid_in   = mem_rsp_valid_out;
        assign mem_rsp_tag_in     = mem_rsp_tag_out;
        assign mem_rsp_data_in    = mem_rsp_data_out;
        assign mem_rsp_ready_out  = mem_rsp_ready_in;

    end

endmodule