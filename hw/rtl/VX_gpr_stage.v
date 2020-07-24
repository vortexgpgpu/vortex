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

    wire [`NUM_THREADS-1:0][31:0] rs1_int_data [`NUM_WARPS-1:0];
    wire [`NUM_THREADS-1:0][31:0] rs2_int_data [`NUM_WARPS-1:0]; 
    wire [`NUM_THREADS-1:0][31:0] rs1_fp_data [`NUM_WARPS-1:0];
    wire [`NUM_THREADS-1:0][31:0] rs2_fp_data [`NUM_WARPS-1:0]; 
    wire [`NUM_THREADS-1:0] we [`NUM_WARPS-1:0];

    wire [`NR_BITS-1:0] raddr1;
	wire [`NR_BITS-1:0] raddr2;

    genvar i;

    for (i = 0; i < `NUM_WARPS; i++) begin
        assign we[i] = writeback_if.valid & {`NUM_THREADS{(i == writeback_if.warp_num)}};

        // Int GPRs
        VX_gpr_ram gpr_int_ram (
            .clk      (clk),
            .we       (we[i] & {`NUM_THREADS{~writeback_if.rd_is_fp}}),                
            .waddr    (writeback_if.rd),
            .wdata    (writeback_if.data),
            .rs1      (raddr1),
            .rs2      (raddr2),                
            .rs1_data (rs1_int_data[i]),
            .rs2_data (rs2_int_data[i])
        );

        // FP GPRs
        VX_gpr_ram gpr_fp_ram (
            .clk      (clk),
            .we       (we[i] & {`NUM_THREADS{writeback_if.rd_is_fp}}),                
            .waddr    (writeback_if.rd),
            .wdata    (writeback_if.data),
            .rs1      (raddr1),
            .rs2      (raddr2),                
            .rs1_data (rs1_fp_data[i]),
            .rs2_data (rs2_fp_data[i])
        );

        // controller for multi-cycle read
		VX_gpr_fp_ctrl VX_gpr_fp_ctrl (
			.clk           (clk),
			.reset         (reset),

			//inputs
            .decode_if      (decode_if), 	
			.rs1_int_data   (rs1_int_data[i]),
			.rs2_int_data   (rs2_int_data[i]),			
            .rs1_fp_data    (rs1_fp_data[i]),
			.rs2_fp_data    (rs2_fp_data[i]),

			// outputs
			.raddr1         (raddr1),
			.raddr2         (raddr2),
			.gpr_data_if    (gpr_data_if),
            .schedule_delay (schedule_delay),
			.gpr_delay      (gpr_delay)
		);
    end

    assign writeback_if.ready = 1'b1;

endmodule
