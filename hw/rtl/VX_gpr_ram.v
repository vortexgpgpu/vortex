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
                integer i;
                for (i = 0; i < `NUM_THREADS; i = i + 1) begin
                    if (be[i]) begin
                        ram[waddr][i][0] <= wdata[i][7:0];
                        ram[waddr][i][1] <= wdata[i][15:8];
                        ram[waddr][i][2] <= wdata[i][23:16];
                        ram[waddr][i][3] <= wdata[i][31:24];
                    end
                end
            end    
        end
    end
    
    assign q1 = ram[raddr1];
    assign q2 = ram[raddr2];

endmodule
