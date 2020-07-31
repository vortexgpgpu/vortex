`include "VX_define.vh"

module VX_fpu_unit #(
    parameter CORE_ID = 0
) (
	// inputs 
	input wire clk,
	input wire reset,   

	// inputs
	VX_fpu_req_if       fpu_req_if,
    VX_csr_to_fpu_if    csr_to_fpu_if,
    
	// outputs        
    VX_fpu_to_cmt_if    fpu_commit_if
);  
    
    assign csr_to_fpu_if.warp_num = fpu_req_if.warp_num;
    wire [`FRM_BITS-1:0] frm = (fpu_req_if.frm == `FRM_DYN) ? csr_to_fpu_if.frm : fpu_req_if.frm;

`ifdef SYNTHESIS

    VX_fp_fpga fp_core (
        .clk        (clk),
        .reset      (reset),   

        .in_valid   (fpu_req_if.valid),
        .in_ready   (fpu_req_if.ready),        

        .in_tag     (fpu_req_if.issue_tag),
        
        .op         (fpu_req_if.fpu_op),
        .frm        (frm),

        .dataa      (fpu_req_if.rs1_data),
        .datab      (fpu_req_if.rs2_data),
        .datac      (fpu_req_if.rs3_data),
        .result     (fpu_commit_if.data), 

        .has_fflags (fpu_commit_if.has_fflags),
        .fflags     (fpu_commit_if.fflags),

        .out_tag    (fpu_commit_if.issue_tag),

        .out_ready  (fpu_commit_if.ready),
        .out_valid  (fpu_commit_if.valid)
    );   

`else

    VX_fpnew #(
        .FMULADD  (0),
        .FDIVSQRT (1),
        .FNONCOMP (1),
        .FCONV    (1)
    ) fp_core (
        .clk        (clk),
        .reset      (reset),   

        .in_valid   (fpu_req_if.valid),
        .in_ready   (fpu_req_if.ready),        

        .in_tag     (fpu_req_if.issue_tag),
        
        .op         (fpu_req_if.fpu_op),
        .frm        (frm),

        .dataa      (fpu_req_if.rs1_data),
        .datab      (fpu_req_if.rs2_data),
        .datac      (fpu_req_if.rs3_data),
        .result     (fpu_commit_if.data), 

        .has_fflags (fpu_commit_if.has_fflags),
        .fflags     (fpu_commit_if.fflags),

        .out_tag    (fpu_commit_if.issue_tag),

        .out_ready  (fpu_commit_if.ready),
        .out_valid  (fpu_commit_if.valid)
    );
    
`endif

endmodule