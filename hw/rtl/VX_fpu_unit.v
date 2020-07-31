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
    VX_csr_to_fpu_if    csr_to_fpu_if,
    
	// outputs        
    VX_fpu_to_cmt_if    fpu_commit_if
);    
    localparam FOP_BITS  = fpnew_pkg::OP_BITS;
    localparam FMTF_BITS = $clog2(fpnew_pkg::NUM_FP_FORMATS);
    localparam FMTI_BITS = $clog2(fpnew_pkg::NUM_INT_FORMATS);

    localparam FPU_DPATHW = 32'd32;

    localparam fpnew_pkg::fpu_features_t FPU_FEATURES = '{
        Width:         FPU_DPATHW,
        EnableVectors: 1'b0,
        EnableNanBox:  1'b1,
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

    reg [`LOG2UP(`FPURQ_SIZE)-1:0] fpu_in_tag, fpu_out_tag;
    
    reg [2:0][`NUM_THREADS-1:0][31:0] fpu_operands;   
    
    wire [FMTF_BITS-1:0] fpu_src_fmt = fpnew_pkg::FP32;
    wire [FMTF_BITS-1:0] fpu_dst_fmt = fpnew_pkg::FP32;
    wire [FMTI_BITS-1:0] fpu_int_fmt = fpnew_pkg::INT32;

    wire [`NUM_THREADS-1:0][31:0] fpu_result;
    fpnew_pkg::status_t fpu_status [0:`NUM_THREADS-1];

    assign csr_to_fpu_if.warp_num = fpu_req_if.warp_num;
    wire [`FRM_BITS-1:0] real_frm = (fpu_req_if.frm == `FRM_DYN) ? csr_to_fpu_if.frm : fpu_req_if.frm;

    wire is_class_op_i, is_class_op_o;
    assign is_class_op_i = (fpu_req_if.fpu_op == `FPU_CLASS);

    reg [FOP_BITS-1:0] fpu_op;
    reg [`FRM_BITS-1:0] fpu_rnd;
    reg fpu_op_mod;
    reg fflags_en, fflags_en_o;

    always @(*) begin
        fpu_op     = fpnew_pkg::SGNJ;
        fpu_rnd    = real_frm;  
        fpu_op_mod = 0;        
        fflags_en  = 1;
        fpu_operands[0] = fpu_req_if.rs1_data;
        fpu_operands[1] = fpu_req_if.rs2_data;
        fpu_operands[2] = fpu_req_if.rs3_data;
        case (fpu_req_if.fpu_op)
            `FPU_ADD: begin
                    fpu_op = fpnew_pkg::ADD;
                    fpu_operands[1] = fpu_req_if.rs1_data;
                    fpu_operands[2] = fpu_req_if.rs2_data;
                end
            `FPU_SUB: begin 
                    fpu_op = fpnew_pkg::ADD; 
                    fpu_operands[1] = fpu_req_if.rs1_data;
                    fpu_operands[2] = fpu_req_if.rs2_data;
                    fpu_op_mod = 1; 
                end
            `FPU_MUL:   begin fpu_op = fpnew_pkg::MUL; end
            `FPU_DIV:   begin fpu_op = fpnew_pkg::DIV; end
            `FPU_SQRT:  begin fpu_op = fpnew_pkg::SQRT; end
            `FPU_MADD:  begin fpu_op = fpnew_pkg::FMADD; end
            `FPU_MSUB:  begin fpu_op = fpnew_pkg::FMADD;  fpu_op_mod = 1; end
            `FPU_NMSUB: begin fpu_op = fpnew_pkg::FNMSUB; end
            `FPU_NMADD: begin fpu_op = fpnew_pkg::FNMSUB; fpu_op_mod = 1; end
            `FPU_SGNJ:  begin fpu_op = fpnew_pkg::SGNJ;   fpu_rnd = `FRM_RNE; fflags_en = 0; end
            `FPU_SGNJN: begin fpu_op = fpnew_pkg::SGNJ;   fpu_rnd = `FRM_RTZ; fflags_en = 0; end
            `FPU_SGNJX: begin fpu_op = fpnew_pkg::SGNJ;   fpu_rnd = `FRM_RDN; fflags_en = 0; end
            `FPU_MIN:   begin fpu_op = fpnew_pkg::MINMAX; fpu_rnd = `FRM_RNE; end
            `FPU_MAX:   begin fpu_op = fpnew_pkg::MINMAX; fpu_rnd = `FRM_RTZ; end
            `FPU_CVTWS: begin fpu_op = fpnew_pkg::F2I; end
            `FPU_CVTWUS:begin fpu_op = fpnew_pkg::F2I;  fpu_op_mod = 1; end
            `FPU_CVTSW: begin fpu_op = fpnew_pkg::I2F; end
            `FPU_CVTSWU:begin fpu_op = fpnew_pkg::I2F;  fpu_op_mod = 1; end
            `FPU_MVXW:  begin fpu_op = fpnew_pkg::SGNJ; fpu_rnd = `FRM_RUP; fflags_en = 0; end
            `FPU_MVWX:  begin fpu_op = fpnew_pkg::SGNJ; fpu_rnd = `FRM_RUP; fflags_en = 0; end
            `FPU_CLASS: begin fpu_op = fpnew_pkg::CLASSIFY; fflags_en = 0; end
            `FPU_CMP:   begin fpu_op = fpnew_pkg::CMP; end
            default:;
        endcase
    end  

    genvar i;

`DISABLE_TRACING
    
    for (i = 0; i < `NUM_THREADS; i++) begin
        if (0 == i) begin
            fpnew_top #( 
                .Features       (FPU_FEATURES),
                .Implementation (FPU_IMPLEMENTATION),
                .TagType        (logic[`LOG2UP(`FPURQ_SIZE)+1+1-1:0])
            ) fpnew_core (
                .clk_i          (clk),
                .rst_ni         (1'b1),
                .operands_i     ({fpu_operands[2][0], fpu_operands[1][0], fpu_operands[0][0]}),
                .rnd_mode_i     (fpnew_pkg::roundmode_e'(fpu_rnd)),
                .op_i           (fpnew_pkg::operation_e'(fpu_op)),
                .op_mod_i       (fpu_op_mod),
                .src_fmt_i      (fpnew_pkg::fp_format_e'(fpu_src_fmt)),
                .dst_fmt_i      (fpnew_pkg::fp_format_e'(fpu_dst_fmt)),
                .int_fmt_i      (fpnew_pkg::int_format_e'(fpu_int_fmt)),
                .vectorial_op_i (1'b0),
                .tag_i          ({fpu_in_tag, fflags_en, is_class_op_i}),
                .in_valid_i     (fpu_in_valid),
                .in_ready_o     (fpu_in_ready),
                .flush_i        (reset),
                .result_o       (fpu_result),
                .status_o       (fpu_status[0]),
                .tag_o          ({fpu_out_tag, fflags_en_o, is_class_op_o}),
                .out_valid_o    (fpu_out_valid),
                .out_ready_i    (fpu_out_ready),
                `UNUSED_PIN     (busy_o)
            );
        end else begin
            fpnew_top #( 
                .Features       (FPU_FEATURES),
                .Implementation (FPU_IMPLEMENTATION),
                .TagType        (logic)
            ) fpnew_core (
                .clk_i          (clk),
                .rst_ni         (1'b1),
                .operands_i     ({fpu_operands[2][i], fpu_operands[1][i], fpu_operands[0][i]}),
                .rnd_mode_i     (fpnew_pkg::roundmode_e'(fpu_rnd)),
                .op_i           (fpnew_pkg::operation_e'(fpu_op)),
                .op_mod_i       (fpu_op_mod),
                .src_fmt_i      (fpnew_pkg::fp_format_e'(fpu_src_fmt)),
                .dst_fmt_i      (fpnew_pkg::fp_format_e'(fpu_dst_fmt)),
                .int_fmt_i      (fpnew_pkg::int_format_e'(fpu_int_fmt)),
                .vectorial_op_i (1'b0),
                .tag_i          (1'b0),
                .in_valid_i     (fpu_in_valid),
                `UNUSED_PIN     (in_ready_o),
                .flush_i        (reset),
                .result_o       (fpu_result[i]),
                .status_o       (fpu_status[i]),
                `UNUSED_PIN     (tag_o),
                `UNUSED_PIN     (out_valid_o),
                .out_ready_i    (fpu_out_ready),
                `UNUSED_PIN     (busy_o)
            );
        end
    end

`ENABLE_TRACING

    assign fpu_in_valid = fpu_req_if.valid;
    assign fpu_in_tag   = fpu_req_if.issue_tag;

    // can accept new request?
    assign fpu_req_if.ready = fpu_in_ready;

    assign fpu_commit_if.valid     = fpu_out_valid;
    assign fpu_commit_if.issue_tag = fpu_out_tag;
    assign fpu_commit_if.data      = fpu_result;
    
    assign fpu_commit_if.upd_fflags = fflags_en_o;   
    
    for (i = 0; i < `NUM_THREADS; i++) begin
        assign fpu_commit_if.fflags[i][0] = fpu_status[i].NX;
        assign fpu_commit_if.fflags[i][1] = fpu_status[i].UF;
        assign fpu_commit_if.fflags[i][2] = fpu_status[i].OF;
        assign fpu_commit_if.fflags[i][3] = fpu_status[i].DZ;
        assign fpu_commit_if.fflags[i][4] = fpu_status[i].NV;
    end
        
    assign fpu_out_ready = fpu_commit_if.ready;

endmodule