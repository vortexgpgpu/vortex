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
    reg [`NUM_THREADS-1:0][31:0] ram[31:0];

    integer i;

    `UNUSED_VAR(reset)

    always @(posedge clk) begin
        if (we) begin                
            for (i = 0; i < `NUM_THREADS; i++) begin
                if (be[i]) begin
                    ram[waddr][i] <= wdata[i];
                end
            end
        end
    end
    
    assign q1 = ram[raddr1];
    assign q2 = ram[raddr2];

endmodule
