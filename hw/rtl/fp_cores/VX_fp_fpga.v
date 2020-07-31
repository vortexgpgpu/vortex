`include "VX_define.vh"

module VX_fp_fpga (
	input wire clk,
	input wire reset,   

    output wire in_ready,
    input wire  in_valid,

    input wire [`ISTAG_BITS-1:0] in_tag,
	
    input wire [`FPU_BITS-1:0] op,
    input wire [`FRM_BITS-1:0] frm,

    input wire [`NUM_THREADS-1:0][31:0]  dataa,
    input wire [`NUM_THREADS-1:0][31:0]  datab,
    input wire [`NUM_THREADS-1:0][31:0]  datac,
    output wire [`NUM_THREADS-1:0][31:0] result, 

    output wire has_fflags,
    output wire [`NUM_THREADS-1:0][`FFG_BITS-1:0] fflags,

    output wire [`ISTAG_BITS-1:0] out_tag,

    input wire  out_ready,
    output wire out_valid
);
    wire fpnew_in_ready;
    wire [`NUM_THREADS-1:0][31:0] fpnew_result;
    wire fpnew_has_fflags;  
    wire [`NUM_THREADS-1:0][`FFG_BITS-1:0] fpnew_fflags;  
    wire [`ISTAG_BITS-1:0] fpnew_out_tag;
    wire fpnew_out_ready;
    wire fpnew_out_valid;

    wire [`NUM_THREADS-1:0][31:0] add_result;
    wire add_out_ready;
    
    VX_fpnew #(
        .FMULADD  (0),
        .FDIVSQRT (1),
        .FNONCOMP (1),
        .FCONV    (1)
    ) fp_core (
        .clk        (clk),
        .reset      (reset),   

        .in_valid   (in_valid),
        .in_ready   (fpnew_in_ready),        

        .in_tag     (in_tag),
        
        .op         (op),
        .frm        (frm),

        .dataa      (dataa),
        .datab      (datab),
        .datac      (datac),
        .result     (fpnew_result), 

        .has_fflags (fpnew_has_fflags),
        .fflags     (fpnew_fflags),

        .out_tag    (fpnew_out_tag),

        .out_ready  (fpnew_out_ready),
        .out_valid  (fpnew_out_valid)
    );

    acl_fp_add fp_add (
        .clock  (clk), 
        .dataa  (dataa), 
        .datab  (datab),         
        .enable (add_out_ready), 
        .result (add_result)
    );

    assign in_reqady  = fpnew_in_ready;
    assign has_fflags = fpnew_has_fflags;  
    assign fflags     = fpnew_fflags;  
    assign out_tag    = fpnew_out_tag;
    assign fpnew_out_ready = out_ready;

    assign result = fpnew_out_valid ? fpnew_result : add_result;
    assign out_valid = fpnew_out_valid; 

endmodule