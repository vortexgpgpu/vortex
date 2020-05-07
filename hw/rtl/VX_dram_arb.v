`include "VX_cache_config.vh"

module VX_dram_arb #(
    parameter BANK_LINE_SIZE = 1, 
    parameter NUM_REQUESTS   = 1, 
    parameter CORE_TAG_WIDTH = 1,
    parameter DRAM_TAG_WIDTH = 1
) (
    input wire clk,
    input wire reset,

    // Core request    
    input wire [NUM_REQUESTS-1:0]                       core_req_read,
    input wire [NUM_REQUESTS-1:0]                       core_req_write,
    input wire [NUM_REQUESTS-1:0][`DRAM_ADDR_WIDTH-1:0] core_req_addr,
    input wire [NUM_REQUESTS-1:0][`BANK_LINE_WIDTH-1:0] core_req_data,
    input wire [NUM_REQUESTS-1:0][CORE_TAG_WIDTH-1:0]   core_req_tag,
    output reg [NUM_REQUESTS-1:0]                       core_req_ready,

    // Core response
    output wire [NUM_REQUESTS-1:0]                      core_rsp_valid,    
    output wire [NUM_REQUESTS-1:0][`BANK_LINE_WIDTH-1:0]core_rsp_data,
    output wire [NUM_REQUESTS-1:0][CORE_TAG_WIDTH-1:0]  core_rsp_tag,
    input  wire [NUM_REQUESTS-1:0]                      core_rsp_ready,   

    // DRAM request
    output reg                         dram_req_read,
    output reg                         dram_req_write,    
    output reg [`DRAM_ADDR_WIDTH-1:0]  dram_req_addr,
    output reg [`BANK_LINE_WIDTH-1:0]  dram_req_data,
    output reg [DRAM_TAG_WIDTH-1:0]    dram_req_tag,
    input  wire                        dram_req_ready,
    
    // DRAM response
    input  wire                        dram_rsp_valid,    
    input  wire [`BANK_LINE_WIDTH-1:0] dram_rsp_data,
    input  wire [DRAM_TAG_WIDTH-1:0]   dram_rsp_tag,
    output wire                        dram_rsp_ready
);
    reg [`REQS_BITS-1:0] bus_req_idx;

    always @(posedge clk) begin
        if (reset) begin      
            bus_req_idx <= 0;
        end else begin
            bus_req_idx <= bus_req_idx + 1;
        end
    end

    integer i;
    generate 
        always @(*) begin
            dram_req_read  = 0;
            dram_req_write = 0;
            dram_req_addr  = 'z;
            dram_req_data  = 'z;
            dram_req_tag   = 'z;

            for (i = 0; i < NUM_REQUESTS; i++) begin
                if (bus_req_idx == (`REQS_BITS)'(i)) begin
                    dram_req_read     = core_req_read[i];
                    dram_req_write    = core_req_write[i];
                    dram_req_addr     = core_req_addr[i];
                    dram_req_data     = core_req_data[i];
                    dram_req_tag      = {core_req_tag[i], (`REQS_BITS)'(i)};
                    core_req_ready[i] = dram_req_ready;
                end else begin
                    core_req_ready[i] = 0;
                end
            end
        end
    endgenerate

    genvar j;
    wire [`REQS_BITS-1:0] bus_rsp_idx = dram_rsp_tag[`REQS_BITS-1:0];
    for (j = 0; j < NUM_REQUESTS; j++) begin                
        assign core_rsp_valid[j] = dram_rsp_valid && (bus_rsp_idx == (`REQS_BITS)'(j));
        assign core_rsp_data[j]  = dram_rsp_data;
        assign core_rsp_tag[j]   = dram_rsp_tag[`REQS_BITS +: CORE_TAG_WIDTH];              
    end
    assign dram_rsp_ready = core_rsp_ready[bus_rsp_idx];

endmodule