`include "VX_define.vh"

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
    reg [`LOG2UP(NUM_REQUESTS)-1:0] bus_sel;

    always @(posedge clk) begin
        if (reset) begin      
            bus_sel <= 0;
        end else begin
            bus_sel <= bus_sel + 1;
        end
    end

    integer i;

    generate 
        always @(*) begin
            dram_req_read  = 'z;
            dram_req_write = 'z;
            dram_req_addr  = 'z;
            dram_req_data  = 'z;
            dram_req_tag   = 'z;

            for (i = 0; i < NUM_REQUESTS; i++) begin
                if (bus_sel == (`LOG2UP(NUM_REQUESTS))'(i)) begin
                    dram_req_read     = core_req_read[i];
                    dram_req_write    = core_req_write[i];
                    dram_req_addr     = core_req_addr[i];
                    dram_req_data     = core_req_data[i];
                    dram_req_tag      = {core_req_tag[i], (`LOG2UP(NUM_REQUESTS))'(i)};
                    core_req_ready[i] = dram_req_ready;
                end else begin
                    core_req_ready[i] = 0;
                end
            end
        end
    endgenerate

    reg is_valid;

    generate 
        always @(*) begin
            dram_rsp_ready = 0;
            
            for (i = 0; i < NUM_REQUESTS; i++) begin
                is_valid = (dram_rsp_tag[`LOG2UP(NUM_REQUESTS)-1:0] == (`LOG2UP(NUM_REQUESTS))'(i));
                
                core_rsp_valid[i] = dram_rsp_valid & is_valid;
                core_rsp_data[i]  = dram_rsp_data;
                core_rsp_tag[i]   = dram_rsp_tag[`LOG2UP(NUM_REQUESTS) +: CORE_TAG_WIDTH];      

                if (is_valid) begin      
                    dram_rsp_ready = core_rsp_ready[i];
                end
            end
        end
    endgenerate    

endmodule