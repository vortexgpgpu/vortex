`include "VX_define.vh"

module VX_gpr_ram (
    input wire                          clk,
    input wire                          reset,
	input wire                          we,
    input wire [4:0]                    waddr,
	input wire [4:0]                    raddr1,
	input wire [4:0]                    raddr2,
    input wire [`NUM_THREADS-1:0]       be,
    input wire [`NUM_THREADS-1:0][31:0] wdata,
    output reg [`NUM_THREADS-1:0][31:0] q1,
	output reg [`NUM_THREADS-1:0][31:0] q2
);
    // Thread   Byte  Bit
    logic [`NUM_THREADS-1:0][3:0][7:0] ram[31:0];

    always @(posedge clk) begin
        if (reset) begin
            //--
        end else begin
            if (we) begin
                integer t;
                for (t = 0; t < `NUM_THREADS; t = t + 1) begin
                    if (be[t]) begin
                        ram[waddr][t][0] <= wdata[t][7:0];
                        ram[waddr][t][1] <= wdata[t][15:8];
                        ram[waddr][t][2] <= wdata[t][23:16];
                        ram[waddr][t][3] <= wdata[t][31:24];
                    end
                end
            end    
        end
    end
    
    assign q1 = ram[raddr1];
    assign q2 = ram[raddr2];

endmodule
