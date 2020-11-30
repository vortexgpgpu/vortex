`include "VX_define.vh"

`ifndef SYNTHESIS
`include "float_dpi.vh"
`endif

module VX_fp_addmul #( 
    parameter TAGW = 1,
    parameter LANES = 1
) (
    input wire clk,
    input wire reset,   

    output wire ready_in,
    input wire  valid_in,

    input wire [TAGW-1:0] tag_in,

    input wire  do_sub,
    input wire  do_mul,    

    input wire [LANES-1:0][31:0]  dataa,
    input wire [LANES-1:0][31:0]  datab,
    output wire [LANES-1:0][31:0] result, 

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);    
    
    wire stall = ~ready_out && valid_out;
    wire enable = ~stall;

    reg do_sub_r, do_mul_r;

    for (genvar i = 0; i < LANES; i++) begin
        
        wire [31:0] result_add;
        wire [31:0] result_sub;
        wire [31:0] result_mul;

    `ifdef QUARTUS
        twentynm_fp_mac mac_fp_add (
            // inputs
            .accumulate(),
            .chainin_overflow(),
            .chainin_invalid(),
            .chainin_underflow(),
            .chainin_inexact(),
            .ax(dataa[i]),
            .ay(datab[i]),
            .az(),
            .clk({2'b00, clk}),
            .ena({2'b00, enable}),
            .aclr({reset, reset}),
            .chainin(),
            // outputs
            .overflow(),
            .invalid(),
            .underflow(),
            .inexact(),
            .chainout_overflow(),
            .chainout_invalid(),
            .chainout_underflow(),
            .chainout_inexact(),
            .resulta(result_add),
            .chainout()
        );
        defparam mac_fp_add.operation_mode = "sp_add"; 
        defparam mac_fp_add.use_chainin = "false"; 
        defparam mac_fp_add.adder_subtract = "false"; 
        defparam mac_fp_add.ax_clock = "0"; 
        defparam mac_fp_add.ay_clock = "0"; 
        defparam mac_fp_add.az_clock = "none"; 
        defparam mac_fp_add.output_clock = "0"; 
        defparam mac_fp_add.accumulate_clock = "none"; 
        defparam mac_fp_add.ax_chainin_pl_clock = "none"; 
        defparam mac_fp_add.accum_pipeline_clock = "none"; 
        defparam mac_fp_add.mult_pipeline_clock = "none"; 
        defparam mac_fp_add.adder_input_clock = "0"; 
        defparam mac_fp_add.accum_adder_clock = "none"; 

        twentynm_fp_mac mac_fp_sub (
            // inputs
            .accumulate(),
            .chainin_overflow(),
            .chainin_invalid(),
            .chainin_underflow(),
            .chainin_inexact(),
            .ax(dataa[i]),
            .ay(datab[i]),
            .az(),
            .clk({2'b00, clk}),
            .ena({2'b00, enable}),
            .aclr({reset, reset}),
            .chainin(),
            // outputs
            .overflow(),
            .invalid(),
            .underflow(),
            .inexact(),
            .chainout_overflow(),
            .chainout_invalid(),
            .chainout_underflow(),
            .chainout_inexact(),
            .resulta(result_sub),
            .chainout()
        );
        defparam mac_fp_sub.operation_mode = "sp_add"; 
        defparam mac_fp_sub.use_chainin = "false"; 
        defparam mac_fp_sub.adder_subtract = "true"; 
        defparam mac_fp_sub.ax_clock = "0"; 
        defparam mac_fp_sub.ay_clock = "0"; 
        defparam mac_fp_sub.az_clock = "none"; 
        defparam mac_fp_sub.output_clock = "0"; 
        defparam mac_fp_sub.accumulate_clock = "none"; 
        defparam mac_fp_sub.ax_chainin_pl_clock = "none"; 
        defparam mac_fp_sub.accum_pipeline_clock = "none"; 
        defparam mac_fp_sub.mult_pipeline_clock = "none"; 
        defparam mac_fp_sub.adder_input_clock = "0"; 
        defparam mac_fp_sub.accum_adder_clock = "none";

        twentynm_fp_mac mac_fp_mul (
            // inputs
            .accumulate(),
            .chainin_overflow(),
            .chainin_invalid(),
            .chainin_underflow(),
            .chainin_inexact(),
            .ax(),
            .ay(datab[i]),
            .az(dataa[i]),
            .clk({2'b00, clk}),
            .ena({2'b00, enable}),
            .aclr({reset, reset}),
            .chainin(),
            // outputs
            .overflow(),
            .invalid(),
            .underflow(),
            .inexact(),
            .chainout_overflow(),
            .chainout_invalid(),
            .chainout_underflow(),
            .chainout_inexact(),
            .resulta(result_mul),
            .chainout()
        );
        defparam mac_fp_mul.operation_mode = "sp_mult"; 
        defparam mac_fp_mul.use_chainin = "false"; 
        defparam mac_fp_mul.adder_subtract = "false"; 
        defparam mac_fp_mul.ax_clock = "none"; 
        defparam mac_fp_mul.ay_clock = "0"; 
        defparam mac_fp_mul.az_clock = "0"; 
        defparam mac_fp_mul.output_clock = "0"; 
        defparam mac_fp_mul.accumulate_clock = "none"; 
        defparam mac_fp_mul.ax_chainin_pl_clock = "none"; 
        defparam mac_fp_mul.accum_pipeline_clock = "none"; 
        defparam mac_fp_mul.mult_pipeline_clock = "0"; 
        defparam mac_fp_mul.adder_input_clock = "none"; 
        defparam mac_fp_mul.accum_adder_clock = "none";
    `else
        integer fadd_h, fsub_h, fmul_h;
        initial begin
            fadd_h = dpi_register();
            fsub_h = dpi_register();
            fmul_h = dpi_register();
        end
        always @(posedge clk) begin
           dpi_fadd(fadd_h, enable, dataa[i], datab[i], result_add);
           dpi_fsub(fsub_h, enable, dataa[i], datab[i], result_sub);
           dpi_fmul(fmul_h, enable, dataa[i], datab[i], result_mul);
        end
    `endif

        assign result[i] = do_mul_r ? result_mul : (do_sub_r ? result_sub : result_add);
    end
    
    VX_shift_register #(
        .DATAW(1 + TAGW + 1 + 1),
        .DEPTH(`LATENCY_FADDMUL)
    ) shift_reg (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .in({valid_in,   tag_in,  do_sub,   do_mul}),
        .out({valid_out, tag_out, do_sub_r, do_mul_r})
    );

    assign ready_in = enable;

endmodule
