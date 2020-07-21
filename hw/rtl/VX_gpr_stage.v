`include "VX_define.vh"

module VX_gpr_stage #(
    parameter CORE_ID = 0
) (
    input wire      clk,

    // inputs    
    VX_wb_if        writeback_if,
    VX_decode_if    decode_if,

    // outputs
    VX_gpr_data_if  gpr_data_if
);
    wire [`NUM_THREADS-1:0][31:0] rs1_data_all [`NUM_WARPS-1:0];
    wire [`NUM_THREADS-1:0][31:0] rs2_data_all [`NUM_WARPS-1:0]; 
    wire [`NUM_THREADS-1:0][31:0] rs1_PC;
    wire [`NUM_THREADS-1:0][31:0] rs2_imm;
    wire [`NUM_THREADS-1:0] we [`NUM_WARPS-1:0];

    genvar i;

    for (i = 0; i < `NUM_THREADS; i++) begin
        assign rs1_PC[i]  = decode_if.curr_PC;
        assign rs2_imm[i] = decode_if.imm;
    end

    assign gpr_data_if.rs1_data = decode_if.rs1_is_PC  ? rs1_PC  : rs1_data_all[decode_if.warp_num];
    assign gpr_data_if.rs2_data = decode_if.rs2_is_imm ? rs2_imm : rs2_data_all[decode_if.warp_num];
      
    for (i = 0; i < `NUM_WARPS; i++) begin
        assign we[i] = writeback_if.valid & {`NUM_THREADS{(i == writeback_if.warp_num)}};
        VX_gpr_ram gpr_ram (
            .clk      (clk),
            .we       (we[i]),                
            .waddr    (writeback_if.rd),
            .wdata    (writeback_if.data),
            .rs1      (decode_if.rs1),
            .rs2      (decode_if.rs2),                
            .rs1_data (rs1_data_all[i]),
            .rs2_data (rs2_data_all[i])
        );
    end

    assign writeback_if.ready = 1'b1;

endmodule
