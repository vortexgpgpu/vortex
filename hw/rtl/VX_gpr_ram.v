`include "VX_define.vh"

`TRACING_OFF

module VX_gpr_ram (
    input wire clk,   
    input wire [`NUM_THREADS-1:0] we,    
    input wire [`NW_BITS+`NR_BITS-1:0] waddr,
    input wire [`NUM_THREADS-1:0][31:0] wdata,    
    input wire [`NW_BITS+`NR_BITS-1:0] rs1,
    input wire [`NW_BITS+`NR_BITS-1:0] rs2,
    output wire [`NUM_THREADS-1:0][31:0] rs1_data,
    output wire [`NUM_THREADS-1:0][31:0] rs2_data
); 

    reg [`NUM_THREADS-1:0][3:0][7:0] mem [(`NUM_WARPS * `NUM_REGS)-1:0];
    reg [`NUM_THREADS-1:0][31:0] q1, q2;
            
    always @(posedge clk) begin
        for (integer i = 0; i < `NUM_THREADS; i++) begin
            if (we[i]) begin
                mem[waddr][i][0] <= wdata[i][07:00];
                mem[waddr][i][1] <= wdata[i][15:08];
                mem[waddr][i][2] <= wdata[i][23:16];
                mem[waddr][i][3] <= wdata[i][31:24];
            end
        end
        q1 <= mem[rs1];
        q2 <= mem[rs2];
    end

    assign rs1_data = q1;
    assign rs2_data = q2;

endmodule

`TRACING_ON