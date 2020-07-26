`include "VX_define.vh"

module VX_gpr_stage #(
    parameter CORE_ID = 0
) (
    input wire      clk,
    input wire      reset,

    // inputs    
    VX_wb_if        writeback_if,
    VX_decode_if    decode_if,    

    // outputs
    VX_gpr_data_if  gpr_data_if,

    input wire      schedule_delay,
    output wire     gpr_delay
);
    `UNUSED_VAR (reset)

    wire [`NUM_THREADS-1:0][31:0] rs1_int_data [`NUM_WARPS-1:0];
    wire [`NUM_THREADS-1:0][31:0] rs2_int_data [`NUM_WARPS-1:0]; 
    wire [`NUM_THREADS-1:0] we [`NUM_WARPS-1:0];

    wire [`NR_BITS-1:0] raddr1;
	wire [`NR_BITS-1:0] raddr2;

    genvar i;

    for (i = 0; i < `NUM_WARPS; i++) begin
        assign we[i] = writeback_if.thread_mask & {`NUM_THREADS{~writeback_if.rd_is_fp && (i == writeback_if.warp_num)}};                
        VX_gpr_ram gpr_int_ram (
            .clk      (clk),
            .we       (we[i]),                
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
        assign we[i] = writeback_if.thread_mask & {`NUM_THREADS{writeback_if.rd_is_fp && (i == writeback_if.warp_num)}};
        VX_gpr_ram gpr_fp_ram (
            .clk      (clk),
            .we       (we[i]),                
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
        .decode_if      (decode_if), 	
        .rs1_int_data   (rs1_int_data[decode_if.warp_num]),
        .rs2_int_data   (rs2_int_data[decode_if.warp_num]),			
        .rs1_fp_data    (rs1_fp_data[decode_if.warp_num]),
        .rs2_fp_data    (rs2_fp_data[decode_if.warp_num]),

        // outputs
        .raddr1         (raddr1),
        .raddr2         (raddr2),
        .gpr_data_if    (gpr_data_if),
        .schedule_delay (schedule_delay),
        .gpr_delay      (gpr_delay)
    );

`else
    assign raddr1 = decode_if.rs1;
    assign raddr2 = decode_if.rs2;
    assign gpr_data_if.rs1_data = rs1_int_data[decode_if.warp_num];
    assign gpr_data_if.rs2_data = rs2_int_data[decode_if.warp_num];
    assign gpr_data_if.rs3_data = 0;
    assign gpr_delay = 0;
    `UNUSED_VAR (schedule_delay)
`endif

    assign writeback_if.ready = 1'b1;

endmodule
