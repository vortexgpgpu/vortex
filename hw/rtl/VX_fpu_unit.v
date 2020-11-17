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
    localparam FPUQ_BITS = `LOG2UP(`FPUQ_SIZE);

    wire ready_in;
    wire valid_out;
    wire ready_out;

    wire [`NW_BITS-1:0] rsp_wid;
    wire [`NUM_THREADS-1:0] rsp_tmask;
    wire [31:0] rsp_PC;
    wire [`NR_BITS-1:0] rsp_rd;
    wire rsp_wb;

    wire has_fflags;
    fflags_t [`NUM_THREADS-1:0] fflags;
    wire [`NUM_THREADS-1:0][31:0] result;

    wire [FPUQ_BITS-1:0] tag_in, tag_out;    
    wire fpuq_full;

    wire fpuq_push = fpu_req_if.valid && fpu_req_if.ready;
    wire fpuq_pop  = valid_out && ready_out;

    VX_cam_buffer #(
        .DATAW (`NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1),
        .SIZE  (`FPUQ_SIZE)
    ) fpu_cam  (
        .clk            (clk),
        .reset          (reset),
        .acquire_slot   (fpuq_push),       
        .write_addr     (tag_in),                
        .read_addr      (tag_out),
        .release_addr   (tag_out),        
        .write_data     ({fpu_req_if.wid, fpu_req_if.tmask, fpu_req_if.PC, fpu_req_if.rd, fpu_req_if.wb}),                    
        .read_data      ({rsp_wid,        rsp_tmask,        rsp_PC,        rsp_rd,        rsp_wb}),        
        .release_slot   (fpuq_pop),     
        .full           (fpuq_full)
    );
    
    // can accept new request?
    assign fpu_req_if.ready = ready_in && ~fpuq_full;

    wire valid_in = fpu_req_if.valid && ~fpuq_full;

    // resolve dynamic FRM    
    assign csr_to_fpu_if.wid = fpu_req_if.wid;
    wire [`FRM_BITS-1:0] fpu_frm = (fpu_req_if.op_mod == `FRM_DYN) ? csr_to_fpu_if.frm : fpu_req_if.op_mod;   

`ifdef FPU_FAST

    VX_fp_fpga #(
        .TAGW (FPUQ_BITS)
    ) fp_core (
        .clk        (clk),
        .reset      (reset),   

        .valid_in   (valid_in),
        .ready_in   (ready_in),        

        .tag_in     (tag_in),
        
        .op_type    (fpu_req_if.op_type),
        .frm        (fpu_frm),

        .dataa      (fpu_req_if.rs1_data),
        .datab      (fpu_req_if.rs2_data),
        .datac      (fpu_req_if.rs3_data),
        .result     (result), 

        .has_fflags (has_fflags),
        .fflags     (fflags),

        .tag_out    (tag_out),

        .ready_out  (ready_out),
        .valid_out  (valid_out)
    );   

`else

    VX_fpnew #(
        .FMULADD  (1),
        .FDIVSQRT (1),
        .FNONCOMP (1),
        .FCONV    (1),
        .TAGW     (FPUQ_BITS)
    ) fp_core (
        .clk        (clk),
        .reset      (reset),   

        .valid_in   (valid_in),
        .ready_in   (ready_in),        

        .tag_in     (tag_in),
        
        .op_type    (fpu_req_if.op_type),
        .frm        (fpu_frm),

        .dataa      (fpu_req_if.rs1_data),
        .datab      (fpu_req_if.rs2_data),
        .datac      (fpu_req_if.rs3_data),
        .result     (result), 

        .has_fflags (has_fflags),
        .fflags     (fflags),

        .tag_out    (tag_out),

        .ready_out  (ready_out),
        .valid_out  (valid_out)
    );
    
`endif

    wire stall_out = ~fpu_commit_if.ready && fpu_commit_if.valid;   

    VX_generic_register #(
        .N(1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32) + 1 + (`NUM_THREADS * `FFG_BITS))
    ) pipe_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall_out),
        .flush (1'b0),
        .in    ({valid_out,           rsp_wid,           rsp_tmask,           rsp_PC,           rsp_rd,           rsp_wb,           result,             has_fflags,               fflags}),
        .out   ({fpu_commit_if.valid, fpu_commit_if.wid, fpu_commit_if.tmask, fpu_commit_if.PC, fpu_commit_if.rd, fpu_commit_if.wb, fpu_commit_if.data, fpu_commit_if.has_fflags, fpu_commit_if.fflags})
    );

    assign ready_out = ~stall_out;

endmodule