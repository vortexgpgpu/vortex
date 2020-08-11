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

    wire [`NUM_THREADS-1:0][31:0] rs1_data;
    wire [`NUM_THREADS-1:0][31:0] rs2_data; 
    wire [`NW_BITS+`NR_BITS-1:0]  raddr1;             

    VX_gpr_ram gpr_ram (
        .clk      (clk),
        .we       ({`NUM_THREADS{writeback_if.valid}} & writeback_if.thread_mask),                
        .waddr    ({writeback_if.warp_num, writeback_if.rd}),
        .wdata    (writeback_if.data),
        .rs1      (raddr1),
        .rs2      ({gpr_read_if.warp_num, gpr_read_if.rs2}),
        .rs1_data (rs1_data),
        .rs2_data (rs2_data)
    );    

`ifdef EXT_F_ENABLE      
    VX_gpr_fp_ctrl VX_gpr_fp_ctrl (
        .clk        (clk),
        .reset      (reset),
        .rs1_data   (rs1_data),
        .rs2_data   (rs2_data),	
        .raddr1     (raddr1),
        .gpr_read_if(gpr_read_if)
    );
`else
    assign raddr1 = {gpr_read_if.warp_num, gpr_read_if.rs1};
    assign gpr_read_if.rs1_data = rs1_data;
    assign gpr_read_if.rs2_data = rs2_data;
    assign gpr_read_if.rs3_data = 0;
    assign gpr_read_if.ready    = 1;
    
    wire valid = gpr_read_if.valid;
    wire use_rs3 = gpr_read_if.use_rs3; 
    wire [`NR_BITS-1:0] rs3 = gpr_read_if.rs3;
    `UNUSED_VAR (valid);
    `UNUSED_VAR (use_rs3);
    `UNUSED_VAR (rs3);
`endif

    assign writeback_if.ready = 1'b1; // writes are stall-free

endmodule
