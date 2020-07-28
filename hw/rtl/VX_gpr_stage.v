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

    wire [`NUM_THREADS-1:0][31:0] rs1_int_data [`NUM_WARPS-1:0];
    wire [`NUM_THREADS-1:0][31:0] rs2_int_data [`NUM_WARPS-1:0]; 

    wire [`NR_BITS-1:0] raddr1;
	wire [`NR_BITS-1:0] raddr2;

    genvar i;

    for (i = 0; i < `NUM_WARPS; i++) begin
        wire [`NUM_WARPS-1:0] we = writeback_if.thread_mask & {`NUM_THREADS{writeback_if.valid && ~writeback_if.rd_is_fp && (i == writeback_if.warp_num)}};                
        VX_gpr_ram gpr_int_ram (
            .clk      (clk),
            .we       (we),                
            .waddr    (writeback_if.rd),
            .wdata    (writeback_if.data),
            .rs1      (raddr1),
            .rs2      (raddr2),                
            .rs1_data (rs1_int_data[i]),
            .rs2_data (rs2_int_data[i])
        );
    end    

`ifdef EXT_F_ENABLE    

    wire [`NUM_THREADS-1:0][31:0] rs1_fp_data [`NUM_WARPS-1:0];
    wire [`NUM_THREADS-1:0][31:0] rs2_fp_data [`NUM_WARPS-1:0]; 

    for (i = 0; i < `NUM_WARPS; i++) begin        
        wire [`NUM_WARPS-1:0] we = writeback_if.thread_mask & {`NUM_THREADS{writeback_if.valid && writeback_if.rd_is_fp && (i == writeback_if.warp_num)}};
        VX_gpr_ram gpr_fp_ram (
            .clk      (clk),
            .we       (we),                
            .waddr    (writeback_if.rd),
            .wdata    (writeback_if.data),
            .rs1      (raddr1),
            .rs2      (raddr2),                
            .rs1_data (rs1_fp_data[i]),
            .rs2_data (rs2_fp_data[i])
        );
    end
    
    VX_gpr_fp_ctrl VX_gpr_fp_ctrl (
        .clk           (clk),
        .reset         (reset),

        //inputs	
        .rs1_int_data   (rs1_int_data[gpr_read_if.warp_num]),
        .rs2_int_data   (rs2_int_data[gpr_read_if.warp_num]),			
        .rs1_fp_data    (rs1_fp_data[gpr_read_if.warp_num]),
        .rs2_fp_data    (rs2_fp_data[gpr_read_if.warp_num]),

        // outputs
        .raddr1         (raddr1),
        .raddr2         (raddr2),
        .gpr_read_if    (gpr_read_if)
    );

`else
    assign raddr1 = gpr_read_if.rs1;
    assign raddr2 = gpr_read_if.rs2;
    assign gpr_read_if.rs1_data = rs1_int_data[gpr_read_if.warp_num];
    assign gpr_read_if.rs2_data = rs2_int_data[gpr_read_if.warp_num];
    assign gpr_read_if.rs3_data = 0;
    assign gpr_read_if.ready = 1;
    
    wire valid = gpr_read_if.valid;
    wire rs1_is_fp = gpr_read_if.rs1_is_fp; 
    wire rs2_is_fp = gpr_read_if.rs2_is_fp;
    wire use_rs3 = gpr_read_if.use_rs3; 
    wire [`NR_BITS-1:0] rs3 = gpr_read_if.rs3;
    `UNUSED_VAR (valid);
    `UNUSED_VAR (rs1_is_fp);
    `UNUSED_VAR (rs2_is_fp);
    `UNUSED_VAR (use_rs3);
    `UNUSED_VAR (rs3);
`endif

    assign writeback_if.ready = 1'b1;

endmodule
