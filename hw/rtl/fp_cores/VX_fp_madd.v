`include "VX_define.vh"

`ifndef SYNTHESIS
`include "float_dpi.vh"
`endif

module VX_fp_madd #( 
    parameter TAGW = 1,
    parameter LANES = 1
) (
    input wire clk,
    input wire reset,   

    output wire ready_in,
    input wire  valid_in,

    input wire [TAGW-1:0] tag_in,

    input wire  do_add,
    input wire  do_sub,
    input wire  do_mul,    

    input wire [LANES-1:0][31:0]  dataa,
    input wire [LANES-1:0][31:0]  datab,
    input wire [LANES-1:0][31:0]  datac,
    output wire [LANES-1:0][31:0] result, 

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);    
    
    wire stall = ~ready_out && valid_out;

    reg do_add_r, do_sub_r, do_mul_r;

    for (genvar i = 0; i < LANES; i++) begin
        
        wire [31:0] result_add;
        wire [31:0] result_sub;
        wire [31:0] result_mul;
        wire [31:0] result_madd;
        wire [31:0] result_msub;

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
            .clk({2'b00,clk}),
            .ena({2'b11,~stall}),
            .aclr(2'b00),
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
            .clk({2'b00,clk}),
            .ena({2'b11,~stall}),
            .aclr(2'b00),
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
            .clk({2'b00,clk}),
            .ena({2'b11,~stall}),
            .aclr(2'b00),
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

        twentynm_fp_mac mac_fp_madd (
            // inputs
            .accumulate(),
            .chainin_overflow(),
            .chainin_invalid(),
            .chainin_underflow(),
            .chainin_inexact(),
            .ax(datac[i]),
            .ay(datab[i]),
            .az(dataa[i]),
            .clk({2'b00,clk}),
            .ena({2'b11,~stall}),
            .aclr(2'b00),
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
            .resulta(result_madd),
            .chainout()
        );
        defparam mac_fp_madd.operation_mode = "sp_mult_add"; 
        defparam mac_fp_madd.use_chainin = "false"; 
        defparam mac_fp_madd.adder_subtract = "false"; 
        defparam mac_fp_madd.ax_clock = "0"; 
        defparam mac_fp_madd.ay_clock = "0"; 
        defparam mac_fp_madd.az_clock = "0"; 
        defparam mac_fp_madd.output_clock = "0"; 
        defparam mac_fp_madd.accumulate_clock = "none"; 
        defparam mac_fp_madd.ax_chainin_pl_clock = "0"; 
        defparam mac_fp_madd.accum_pipeline_clock = "none"; 
        defparam mac_fp_madd.mult_pipeline_clock = "0"; 
        defparam mac_fp_madd.adder_input_clock = "0"; 
        defparam mac_fp_madd.accum_adder_clock = "none"; 

        twentynm_fp_mac mac_fp_msub (
            // inputs
            .accumulate(),
            .chainin_overflow(),
            .chainin_invalid(),
            .chainin_underflow(),
            .chainin_inexact(),
            .ax(datac[i]),
            .ay(datab[i]),
            .az(dataa[i]),
            .clk({2'b00,clk}),
            .ena({2'b11,~stall}),
            .aclr(2'b00),
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
            .resulta(result_msub),
            .chainout()
        );
        defparam mac_fp_msub.operation_mode = "sp_mult_add"; 
        defparam mac_fp_msub.use_chainin = "false"; 
        defparam mac_fp_msub.adder_subtract = "true"; 
        defparam mac_fp_msub.ax_clock = "0"; 
        defparam mac_fp_msub.ay_clock = "0"; 
        defparam mac_fp_msub.az_clock = "0"; 
        defparam mac_fp_msub.output_clock = "0"; 
        defparam mac_fp_msub.accumulate_clock = "none"; 
        defparam mac_fp_msub.ax_chainin_pl_clock = "0"; 
        defparam mac_fp_msub.accum_pipeline_clock = "none"; 
        defparam mac_fp_msub.mult_pipeline_clock = "0"; 
        defparam mac_fp_msub.adder_input_clock = "0"; 
        defparam mac_fp_msub.accum_adder_clock = "none";
    `else
        always @(posedge clk) begin
           dpi_fadd(0*LANES+i, ~stall, valid_in, dataa[i], datab[i], result_add);
           dpi_fsub(1*LANES+i, ~stall, valid_in, dataa[i], datab[i], result_sub);
           dpi_fmul(2*LANES+i, ~stall, valid_in, dataa[i], datab[i], result_mul);
           dpi_fmadd(3*LANES+i, ~stall, valid_in, dataa[i], datab[i], datac[i], result_madd);
           dpi_fmsub(4*LANES+i, ~stall, valid_in, dataa[i], datab[i], datac[i], result_msub);
        end
    `endif

        reg [31:0] result_r;        

        always @(*) begin
            result_r = 'x;
            if (do_mul_r) begin
                if (do_add_r)
                    result_r = result_madd;
                else if (do_sub_r)
                    result_r = result_msub;
                else
                    result_r = result_mul;
            end else begin
                if (do_add_r)
                    result_r = result_add;
                else if (do_sub_r)
                    result_r = result_sub;
            end            
        end

        assign result[i] = result_r;
    end
    
    VX_shift_register #(
        .DATAW(TAGW + 1 + 1 + 1 + 1),
        .DEPTH(`LATENCY_FMADD)
    ) shift_reg1 (
        .clk(clk),
        .reset(reset),
        .enable(~stall),
        .in({tag_in,   valid_in,  do_add,   do_sub,   do_mul}),
        .out({tag_out, valid_out, do_add_r, do_sub_r, do_mul_r})
    );

    assign ready_in  = ~stall;

endmodule
