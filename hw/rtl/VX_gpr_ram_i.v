`include "VX_define.vh"

`TRACING_OFF

module VX_gpr_ram_i #(
    parameter DATAW = 1,    
    parameter DEPTH = 1,
    parameter ADDRW = $clog2(DEPTH)
) (
    input  wire clk,   
    input  wire wren,    
    input  wire [ADDRW-1:0] waddr,
    input  wire [DATAW-1:0] wdata,    
    input  wire [ADDRW-1:0] raddr1,
    input  wire [ADDRW-1:0] raddr2,
    output wire [DATAW-1:0] rdata1,
    output wire [DATAW-1:0] rdata2
); 
    reg [DATAW-1:0] mem [DEPTH-1:0];

    initial mem = '{default: 0};

    always @(posedge clk) begin
        if (wren) begin
            mem [waddr] <= wdata;
        end
    end

    assign rdata1 = mem [raddr1];
    assign rdata2 = mem [raddr2];

endmodule

`TRACING_ON