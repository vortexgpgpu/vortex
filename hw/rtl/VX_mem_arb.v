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
    input wire [NUM_REQUESTS-1:0]                       in_mem_req_valid,
    input wire [NUM_REQUESTS-1:0]                       in_mem_req_rw,  
    input wire [NUM_REQUESTS-1:0][WORD_SIZE-1:0]        in_mem_req_byteen,  
    input wire [NUM_REQUESTS-1:0][ADDR_WIDTH-1:0]       in_mem_req_addr,
    input wire [NUM_REQUESTS-1:0][WORD_WIDTH-1:0]       in_mem_req_data,    
    input wire [NUM_REQUESTS-1:0][TAG_IN_WIDTH-1:0]     in_mem_req_tag,    
    output wire [NUM_REQUESTS-1:0]                      in_mem_req_ready,

    // input response
    output wire [NUM_REQUESTS-1:0]                      in_mem_rsp_valid,
    output wire [NUM_REQUESTS-1:0][WORD_WIDTH-1:0]      in_mem_rsp_data,
    output wire [NUM_REQUESTS-1:0][TAG_IN_WIDTH-1:0]    in_mem_rsp_tag,
    input wire  [NUM_REQUESTS-1:0]                      in_mem_rsp_ready,

    // output request
    output wire                         out_mem_req_valid,
    output wire                         out_mem_req_rw,  
    output wire [WORD_SIZE-1:0]         out_mem_req_byteen,  
    output wire [ADDR_WIDTH-1:0]        out_mem_req_addr,
    output wire [WORD_WIDTH-1:0]        out_mem_req_data,    
    output wire [TAG_OUT_WIDTH-1:0]     out_mem_req_tag,    
    input wire                          out_mem_req_ready,

    // output response
    input wire                          out_mem_rsp_valid,
    input wire [WORD_WIDTH-1:0]         out_mem_rsp_data,
    input wire [TAG_OUT_WIDTH-1:0]      out_mem_rsp_tag,
    output wire                         out_mem_rsp_ready
);
    if (NUM_REQUESTS == 1) begin

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)

        assign out_mem_req_valid  = in_mem_req_valid;
        assign out_mem_req_rw     = in_mem_req_rw;
        assign out_mem_req_byteen = in_mem_req_byteen;
        assign out_mem_req_addr   = in_mem_req_addr;
        assign out_mem_req_data   = in_mem_req_data;
        assign out_mem_req_tag    = in_mem_req_tag;
        assign in_mem_req_ready   = out_mem_req_ready;

        assign in_mem_rsp_valid   = out_mem_rsp_valid;
        assign in_mem_rsp_data    = out_mem_rsp_data;
        assign in_mem_rsp_tag     = out_mem_rsp_tag;
        assign out_mem_rsp_ready  = in_mem_rsp_ready;

    end else begin

        reg [REQS_BITS-1:0] bus_req_sel;

        VX_rr_arbiter #(
            .N(NUM_REQUESTS)
        ) arbiter (
            .clk         (clk),
            .reset       (reset),
            .requests    (in_mem_req_valid),
            .grant_index (bus_req_sel),
            `UNUSED_PIN  (grant_valid),
            `UNUSED_PIN  (grant_onehot)
        );

        assign out_mem_req_valid  = in_mem_req_valid [bus_req_sel];
        assign out_mem_req_rw     = in_mem_req_rw   [bus_req_sel];
        assign out_mem_req_byteen = in_mem_req_byteen [bus_req_sel];
        assign out_mem_req_addr   = in_mem_req_addr [bus_req_sel];
        assign out_mem_req_data   = in_mem_req_data [bus_req_sel];
        assign out_mem_req_tag    = {in_mem_req_tag [bus_req_sel], REQS_BITS'(bus_req_sel)};

        for (genvar i = 0; i < NUM_REQUESTS; i++) begin
            assign in_mem_req_ready[i] = out_mem_req_ready && (bus_req_sel == REQS_BITS'(i));
        end

        wire [REQS_BITS-1:0] bus_rsp_sel = out_mem_rsp_tag[REQS_BITS-1:0];
        
        for (genvar i = 0; i < NUM_REQUESTS; i++) begin                
            assign in_mem_rsp_valid[i] = out_mem_rsp_valid && (bus_rsp_sel == REQS_BITS'(i));
            assign in_mem_rsp_data[i]  = out_mem_rsp_data;
            assign in_mem_rsp_tag[i]   = out_mem_rsp_tag[REQS_BITS +: TAG_IN_WIDTH];              
        end
        assign out_mem_rsp_ready = in_mem_rsp_ready[bus_rsp_sel];

    end

endmodule