`include "VX_define.vh"

module VX_dram_arb #(    
    parameter NUM_REQUESTS   = 1, 
    parameter DRAM_LINE_SIZE = 1, 
    parameter CORE_TAG_WIDTH = 1,
    parameter DRAM_TAG_WIDTH = 1,

    parameter DRAM_LINE_WIDTH = DRAM_LINE_SIZE * 8,
    parameter DRAM_ADDR_WIDTH = 32 - `CLOG2(DRAM_LINE_SIZE)
) (
    input wire clk,
    input wire reset,

    // Core request    
    input wire [NUM_REQUESTS-1:0]                       in_dram_req_valid,
    input wire [NUM_REQUESTS-1:0]                       in_dram_req_rw,
    input wire [NUM_REQUESTS-1:0][DRAM_LINE_SIZE-1:0]   in_dram_req_byteen,
    input wire [NUM_REQUESTS-1:0][DRAM_ADDR_WIDTH-1:0]  in_dram_req_addr,
    input wire [NUM_REQUESTS-1:0][DRAM_LINE_WIDTH-1:0]  in_dram_req_data,
    input wire [NUM_REQUESTS-1:0][CORE_TAG_WIDTH-1:0]   in_dram_req_tag,
    output wire [NUM_REQUESTS-1:0]                      in_dram_req_ready,

    // Core response
    output wire [NUM_REQUESTS-1:0]                      in_dram_rsp_valid,    
    output wire [NUM_REQUESTS-1:0][DRAM_LINE_WIDTH-1:0] in_dram_rsp_data,
    output wire [NUM_REQUESTS-1:0][CORE_TAG_WIDTH-1:0]  in_dram_rsp_tag,
    input  wire [NUM_REQUESTS-1:0]                      in_dram_rsp_ready,   

    // DRAM request
    output wire                        out_dram_req_valid,
    output wire                        out_dram_req_rw,    
    output wire [DRAM_LINE_SIZE-1:0]   out_dram_req_byteen,
    output wire [DRAM_ADDR_WIDTH-1:0]  out_dram_req_addr,
    output wire [DRAM_LINE_WIDTH-1:0]  out_dram_req_data,
    output wire [DRAM_TAG_WIDTH-1:0]   out_dram_req_tag,
    input  wire                        out_dram_req_ready,
    
    // DRAM response
    input  wire                        out_dram_rsp_valid,    
    input  wire [DRAM_LINE_WIDTH-1:0]  out_dram_rsp_data,
    input  wire [DRAM_TAG_WIDTH-1:0]   out_dram_rsp_tag,
    output wire                        out_dram_rsp_ready
);
    reg [`REQS_BITS-1:0] bus_req_sel;

    always @(posedge clk) begin
        if (reset) begin      
            bus_req_sel <= 0;
        end else begin
            bus_req_sel <= bus_req_sel + 1;
        end
    end

    assign out_dram_req_valid = in_dram_req_valid [bus_req_sel];
    assign out_dram_req_rw    = in_dram_req_rw   [bus_req_sel];
    assign out_dram_req_byteen= in_dram_req_byteen [bus_req_sel];
    assign out_dram_req_addr  = in_dram_req_addr [bus_req_sel];
    assign out_dram_req_data  = in_dram_req_data [bus_req_sel];
    assign out_dram_req_tag   = {in_dram_req_tag [bus_req_sel], (`REQS_BITS)'(bus_req_sel)};

    genvar i;

    for (i = 0; i < NUM_REQUESTS; i++) begin
        assign in_dram_req_ready[i] = out_dram_req_ready && (bus_req_sel == `REQS_BITS'(i));
    end

    wire [`REQS_BITS-1:0] bus_rsp_sel = out_dram_rsp_tag[`REQS_BITS-1:0];
    
    for (i = 0; i < NUM_REQUESTS; i++) begin                
        assign in_dram_rsp_valid[i] = out_dram_rsp_valid && (bus_rsp_sel == `REQS_BITS'(i));
        assign in_dram_rsp_data[i]  = out_dram_rsp_data;
        assign in_dram_rsp_tag[i]   = out_dram_rsp_tag[`REQS_BITS +: CORE_TAG_WIDTH];              
    end
    assign out_dram_rsp_ready = in_dram_rsp_ready[bus_rsp_sel];

endmodule