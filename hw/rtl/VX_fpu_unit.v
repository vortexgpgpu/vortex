`include "VX_define.vh"
`include "fpnew_pkg.sv"
`include "defs_div_sqrt_mvp.sv"

module VX_fpu_unit #(
    parameter CORE_ID = 0
) (
	// inputs 
	input wire clk,
	input wire reset,   

	// inputs
	VX_fpu_req_if       fpu_req_if,
    VX_fpu_from_csr_if  fpu_from_csr_if,
    
	// outputs        
    VX_commit_fp_if     fpu_commit_if,
    VX_fpu_to_csr_if    fpu_to_csr_if
);    
    localparam FOP_BITS  = fpnew_pkg::OP_BITS;
    localparam FMTF_BITS = $clog2(fpnew_pkg::NUM_FP_FORMATS);
    localparam FMTI_BITS = $clog2(fpnew_pkg::NUM_INT_FORMATS);

    localparam int FPU_DPATHW = `NUM_THREADS * 32;

    localparam fpnew_pkg::fpu_features_t FPU_FEATURES = '{
        Width:         FPU_DPATHW,
        EnableVectors: 1,
        EnableNanBox:  1,
        FpFmtMask:     5'b10000,
        IntFmtMask:    4'b0010
    };

    localparam fpnew_pkg::fpu_implementation_t FPU_IMPLEMENTATION = '{
      PipeRegs:'{'{`LATENCY_FMULADD, 0, 0, 0, 0},   // ADDMUL
                 '{default: `LATENCY_FDIVSQRT},     // DIVSQRT
                 '{default: `LATENCY_FNONCOMP},     // NONCOMP
                 '{default: `LATENCY_FCONV}},       // CONV
      UnitTypes:'{'{default: fpnew_pkg::PARALLEL},  // ADDMUL
                  '{default: fpnew_pkg::MERGED},    // DIVSQRT
                  '{default: fpnew_pkg::PARALLEL},  // NONCOMP
                  '{default: fpnew_pkg::MERGED}},   // CONV
      PipeConfig: fpnew_pkg::DISTRIBUTED
    };
    
    wire fpu_in_ready, fpu_in_valid;    
    wire fpu_out_ready, fpu_out_valid;

    wire [`LOG2UP(`FPURQ_SIZE)-1:0] fpu_in_tag, fpu_out_tag;
    
    wire [2:0][`NUM_THREADS-1:0][31:0] fpu_operands;   

    wire [FMTF_BITS-1:0] fpu_src_fmt = fpnew_pkg::FP32;
    wire [FMTF_BITS-1:0] fpu_dst_fmt = fpnew_pkg::FP32;
    wire [FMTI_BITS-1:0] fpu_int_fmt = fpnew_pkg::INT32;

    wire [`NUM_THREADS-1:0][31:0] fpu_result;
    fpnew_pkg::status_t fpu_status;

    assign fpu_from_csr_if.warp_num = fpu_req_if.warp_num;
    wire is_dyn_rnd =  &(fpu_req_if.frm);
    wire [`FRM_BITS-1:0] real_frm = is_dyn_rnd ? fpu_from_csr_if.frm : fpu_req_if.frm;

    reg [FOP_BITS-1:0] fpu_op;
    reg [`FRM_BITS-1:0] fpu_rnd;
    reg fpu_op_mod;

    always @(*) begin
        fpu_op     = fpnew_pkg::SGNJ;
        fpu_op_mod = 0;
        fpu_rnd    = fpu_req_if.frm;
        case (fpu_req_if.fpu_op)
            `FPU_ADD:   fpu_op = fpnew_pkg::ADD;
            `FPU_SUB:   begin fpu_op = fpnew_pkg::ADD; fpu_op_mod = 1; end
            `FPU_MUL:   fpu_op = fpnew_pkg::MUL;
            `FPU_DIV:   fpu_op = fpnew_pkg::DIV;
            `FPU_SQRT:  fpu_op = fpnew_pkg::SQRT;
            `FPU_MADD:  fpu_op = fpnew_pkg::FMADD;
            `FPU_MSUB:  begin fpu_op = fpnew_pkg::FMADD; fpu_op_mod = 1; end
            `FPU_NMSUB: fpu_op = fpnew_pkg::FNMSUB;
            `FPU_NMADD: begin fpu_op = fpnew_pkg::FNMSUB; fpu_op_mod = 1; end
            `FPU_SGNJ:  begin fpu_op = fpnew_pkg::SGNJ;   fpu_rnd = `FRM_RNE; end
            `FPU_SGNJN: begin fpu_op = fpnew_pkg::SGNJ;   fpu_rnd = `FRM_RTZ; end
            `FPU_SGNJX: begin fpu_op = fpnew_pkg::SGNJ;   fpu_rnd = `FRM_RDN; end
            `FPU_MIN:   begin fpu_op = fpnew_pkg::MINMAX; fpu_rnd = `FRM_RNE; end
            `FPU_MAX:   begin fpu_op = fpnew_pkg::MINMAX; fpu_rnd = `FRM_RTZ; end
            `FPU_CVTWS: fpu_op = fpnew_pkg::F2I;
            `FPU_CVTWUS:begin fpu_op = fpnew_pkg::ADD;  fpu_op_mod = 1; end
            `FPU_CVTSW: fpu_op = fpnew_pkg::I2F;
            `FPU_CVTSWU:begin fpu_op = fpnew_pkg::I2F;  fpu_op_mod = 1; end
            `FPU_MVXW:  begin fpu_op = fpnew_pkg::SGNJ; fpu_rnd = `FRM_RUP; end
            `FPU_MVWX:  begin fpu_op = fpnew_pkg::SGNJ; fpu_rnd = `FRM_RUP; end
            `FPU_CLASS: fpu_op = fpnew_pkg::CLASSIFY;
            `FPU_CMP:   fpu_op = fpnew_pkg::CMP;
            default:;
        endcase
    end

    assign fpu_operands = {fpu_req_if.rs3_data, fpu_req_if.rs2_data, fpu_req_if.rs1_data};

    fpnew_top #( 
        .Features       (FPU_FEATURES),
        .Implementation (FPU_IMPLEMENTATION),
        .TagType        (logic [`LOG2UP(`FPURQ_SIZE)-1:0])
    ) fpnew_core (
        .clk_i          (clk),
        .rst_ni         (1'b1),
        .operands_i     (fpu_operands),
        .rnd_mode_i     (fpu_rnd),
        .op_i           (fpu_op),
        .op_mod_i       (fpu_op_mod),
        .src_fmt_i      (fpu_src_fmt),
        .dst_fmt_i      (fpu_dst_fmt),
        .int_fmt_i      (fpu_int_fmt),
        .vectorial_op_i (1'b1),
        .tag_i          (fpu_in_tag),
        .in_valid_i     (fpu_in_valid),
        .in_ready_o     (fpu_in_ready),
        .flush_i        (reset),
        .result_o       (fpu_result),
        .status_o       (fpu_status),
        .tag_o          (fpu_out_tag),
        .out_valid_o    (fpu_out_valid),
        .out_ready_i    (fpu_out_ready),
        `UNUSED_PIN     (busy_o)
    );

    wire req_push = fpu_req_if.valid && fpu_req_if.ready;    
    wire req_pop  = fpu_out_valid && fpu_out_ready;
    wire req_full;  

    wire [`NUM_THREADS-1:0] rsp_valid;
    wire [`NW_BITS-1:0] rsp_warp_num;
    wire [31:0] rsp_curr_PC;
    wire rsp_wb;
    wire [`NR_BITS-1:0] rsp_rd;
    wire rsp_rd_is_fp;

    VX_index_queue #(
        .DATAW (`NUM_THREADS + `NW_BITS + 32 + 1 + `NR_BITS + 1),
        .SIZE  (`FPURQ_SIZE)
    ) fpu_req_queue (
        .clk        (clk),
        .reset      (reset),
        .write_data ({fpu_req_if.valid, fpu_req_if.warp_num, fpu_req_if.curr_PC, fpu_req_if.wb, fpu_req_if.rd, fpu_req_if.rd_is_fp}),
        .write_addr (fpu_in_tag),        
        .push       (req_push),    
        .full       (req_full),
        .pop        (req_pop),
        .read_addr  (fpu_out_tag),
        .read_data  ({rsp_valid, rsp_warp_num, rsp_curr_PC, rsp_wb, rsp_rd, rsp_rd_is_fp}),
        `UNUSED_PIN (empty)
    );

    assign fpu_in_valid = (| fpu_req_if.valid) && ~req_full;
    assign fpu_req_if.ready = fpu_in_ready && ~req_full;

    assign fpu_commit_if.valid    = rsp_valid & {`NUM_THREADS{fpu_out_valid}};
    assign fpu_commit_if.warp_num = rsp_warp_num;
    assign fpu_commit_if.curr_PC  = rsp_curr_PC;
    assign fpu_commit_if.data     = fpu_result;
    assign fpu_commit_if.wb       = rsp_wb;
    assign fpu_commit_if.rd       = rsp_rd;
    assign fpu_commit_if.rd_is_fp = rsp_rd_is_fp;
    assign fpu_out_ready          = fpu_commit_if.ready;

    assign fpu_to_csr_if.valid     = fpu_out_valid;
    assign fpu_to_csr_if.warp_num  = rsp_warp_num;
    assign fpu_to_csr_if.fflags_NV = fpu_status.NV;
    assign fpu_to_csr_if.fflags_DZ = fpu_status.DZ;
    assign fpu_to_csr_if.fflags_OF = fpu_status.OF;
    assign fpu_to_csr_if.fflags_UF = fpu_status.UF;
    assign fpu_to_csr_if.fflags_NX = fpu_status.NX;

endmodule