`include "VX_define.vh"

module VX_gpr_stage #(
    parameter CORE_ID = 0
) (
    input wire      clk,
    input wire      reset,

    // inputs    
    VX_wb_if        writeback_if,  

    // outputs
    VX_gpr_read_if  gpr_read_if
);
    `UNUSED_VAR (reset)

    wire [`NUM_THREADS-1:0][31:0] rs1_data [`NUM_WARPS-1:0];
    wire [`NUM_THREADS-1:0][31:0] rs2_data [`NUM_WARPS-1:0]; 

    wire [`NR_BITS-1:0] raddr1;

    genvar i;

    for (i = 0; i < `NUM_WARPS; i++) begin
        wire [`NUM_THREADS-1:0] we = writeback_if.thread_mask 
                                   & {`NUM_THREADS{writeback_if.valid && (i == writeback_if.warp_num)}};                
        VX_gpr_ram gpr_int_ram (
            .clk      (clk),
            .we       (we),                
            .waddr    (writeback_if.rd),
            .wdata    (writeback_if.data),
            .rs1      (raddr1),
            .rs2      (gpr_read_if.rs2),                
            .rs1_data (rs1_data[i]),
            .rs2_data (rs2_data[i])
        );
    end    

`ifdef EXT_F_ENABLE      
    VX_gpr_fp_ctrl VX_gpr_fp_ctrl (
        .clk        (clk),
        .reset      (reset),

        //inputs	
        .rs1_data   (rs1_data[gpr_read_if.warp_num]),
        .rs2_data   (rs2_data[gpr_read_if.warp_num]),	

        // outputs
        .raddr1     (raddr1),
        .gpr_read_if(gpr_read_if)
    );
`else
    assign raddr1 = gpr_read_if.rs1;
    assign gpr_read_if.rs1_data = rs1_data[gpr_read_if.warp_num];
    assign gpr_read_if.rs2_data = rs2_data[gpr_read_if.warp_num];
    assign gpr_read_if.rs3_data = 0;
    assign gpr_read_if.ready = 1;
    
    wire valid = gpr_read_if.valid;
    wire use_rs3 = gpr_read_if.use_rs3; 
    wire [`NR_BITS-1:0] rs3 = gpr_read_if.rs3;
    `UNUSED_VAR (valid);
    `UNUSED_VAR (use_rs3);
    `UNUSED_VAR (rs3);
`endif

    assign writeback_if.ready = 1'b1;

endmodule
