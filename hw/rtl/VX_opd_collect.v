`include "VX_platform.vh"

module VX_opd_collect #(
    parameter INSTW    = 1,
    parameter OPDSW    = 1,
    parameter PASSTHRU = 0
) ( 
    input  wire             clk,
    input  wire             reset,
    
    input  wire             valid_in,
    output wire             ready_in,        
    input  wire [INSTW-1:0] inst_in,
    input  wire [OPDSW-1:0] opds_in,

    output wire [INSTW+OPDSW-1:0] data_out,
    output wire             valid_out,    
    input  wire             ready_out
); 
    wire [INSTW-1:0] inst_out;
    wire [OPDSW-1:0] opds_out;
    wire valid_out_tmp, ready_out_tmp;
    
    VX_skid_buffer #(
        .DATAW (INSTW)
    ) skid_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (valid_in),
        .ready_in  (ready_in),
        .data_in   (inst_in),
        .data_out  (inst_out),
        .valid_out (valid_out_tmp),
        .ready_out (ready_out_tmp)
    );   

    VX_gpr_bypass #(
        .DATAW (OPDSW),
        .PASSTHRU (PASSTHRU)
    ) gpr_bypass (
        .clk       (clk),
        .reset     (reset),
        .push      (valid_in && ready_in),        
        .pop       (valid_out_tmp && ready_out_tmp),
        .data_in   (opds_in),
        .data_out  (opds_out)
    );

    wire stall_out = valid_out && ~ready_out;

    VX_generic_register #(
        .N(1 + INSTW + OPDSW),
        .R(1)
    ) pipe_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall_out),
        .flush (1'b0),
        .in    ({valid_out_tmp, inst_out, opds_out}),
        .out   ({valid_out,     data_out})
    );

    assign ready_out_tmp = ~stall_out;

endmodule