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
    VX_fpu_req_if fpu_req_tmp_if();

    // resolve dynamic FRM    
    wire [`FRM_BITS-1:0] frm, frm_tmp;
    assign csr_to_fpu_if.wid = fpu_req_if.wid;
    assign frm = (fpu_req_if.frm == `FRM_DYN) ? csr_to_fpu_if.frm : fpu_req_if.frm;

    // use a skid buffer since fpcore has realtime backpressure
    VX_elastic_buffer #(
        .DATAW (`ISTAG_BITS + `NW_BITS + 32 + `FPU_BITS + `FRM_BITS + (3 * `NUM_THREADS * 32)),
        .SIZE  (0)
    ) input_buffer (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (fpu_req_if.valid),
        .ready_in  (fpu_req_if.ready),
        .data_in   ({fpu_req_if.issue_tag,     fpu_req_if.wid,     fpu_req_if.curr_PC,     fpu_req_if.op,     frm,     fpu_req_if.rs1_data,     fpu_req_if.rs2_data,     fpu_req_if.rs3_data}),
        .data_out  ({fpu_req_tmp_if.issue_tag, fpu_req_tmp_if.wid, fpu_req_tmp_if.curr_PC, fpu_req_tmp_if.op, frm_tmp, fpu_req_tmp_if.rs1_data, fpu_req_tmp_if.rs2_data, fpu_req_tmp_if.rs3_data}),        
        .ready_out (fpu_req_tmp_if.ready),
        .valid_out (fpu_req_tmp_if.valid)
    );

`ifdef SYNTHESIS

    VX_fp_fpga fp_core (
        .clk        (clk),
        .reset      (reset),   

        .valid_in   (fpu_req_tmp_if.valid),
        .ready_in   (fpu_req_tmp_if.ready),        

        .tag_in     (fpu_req_tmp_if.issue_tag),
        
        .op         (fpu_req_tmp_if.op),
        .frm        (frm_tmp),

        .dataa      (fpu_req_tmp_if.rs1_data),
        .datab      (fpu_req_tmp_if.rs2_data),
        .datac      (fpu_req_tmp_if.rs3_data),
        .result     (fpu_commit_if.data), 

        .has_fflags (fpu_commit_if.has_fflags),
        .fflags     (fpu_commit_if.fflags),

        .tag_out    (fpu_commit_if.issue_tag),

        .ready_out  (1'b1),
        .valid_out  (fpu_commit_if.valid)
    );   

`else

    VX_fpnew #(
        .FMULADD  (1),
        .FDIVSQRT (1),
        .FNONCOMP (1),
        .FCONV    (1)
    ) fp_core (
        .clk        (clk),
        .reset      (reset),   

        .valid_in   (fpu_req_tmp_if.valid),
        .ready_in   (fpu_req_tmp_if.ready),        

        .tag_in     (fpu_req_tmp_if.issue_tag),
        
        .op         (fpu_req_tmp_if.op),
        .frm        (frm_tmp),

        .dataa      (fpu_req_tmp_if.rs1_data),
        .datab      (fpu_req_tmp_if.rs2_data),
        .datac      (fpu_req_tmp_if.rs3_data),
        .result     (fpu_commit_if.data), 

        .has_fflags (fpu_commit_if.has_fflags),
        .fflags     (fpu_commit_if.fflags),

        .tag_out    (fpu_commit_if.issue_tag),

        .ready_out  (1'b1),
        .valid_out  (fpu_commit_if.valid)
    );
    
`endif

endmodule