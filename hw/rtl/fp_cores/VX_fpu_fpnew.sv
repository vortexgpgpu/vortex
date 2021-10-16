`include "VX_fpu_define.vh"
`include "fpnew_pkg.sv"
`include "defs_div_sqrt_mvp.sv"

`TRACING_OFF
module VX_fpu_fpnew #(      
    parameter TAGW     = 1,
    parameter FMULADD  = 1,
    parameter FDIVSQRT = 1,
    parameter FNONCOMP = 1,
    parameter FCONV    = 1
) (
    input wire clk,
    input wire reset,

    input wire  valid_in,
    output wire ready_in,

    input wire [TAGW-1:0] tag_in,
    
    input wire [`INST_FPU_BITS-1:0] op_type,
    input wire [`INST_MOD_BITS-1:0] frm,

    input wire [`NUM_THREADS-1:0][31:0]  dataa,
    input wire [`NUM_THREADS-1:0][31:0]  datab,
    input wire [`NUM_THREADS-1:0][31:0]  datac,
    output wire [`NUM_THREADS-1:0][31:0] result, 

    output wire has_fflags,
    output fflags_t [`NUM_THREADS-1:0] fflags,

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);  
    localparam UNIT_FMULADD  = FMULADD  ? fpnew_pkg::PARALLEL : fpnew_pkg::DISABLED;
    localparam UNIT_FDIVSQRT = FDIVSQRT ? fpnew_pkg::MERGED   : fpnew_pkg::DISABLED;
    localparam UNIT_FNONCOMP = FNONCOMP ? fpnew_pkg::PARALLEL : fpnew_pkg::DISABLED;
    localparam UNIT_FCONV    = FCONV    ? fpnew_pkg::MERGED   : fpnew_pkg::DISABLED;

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
      PipeRegs:'{'{`LATENCY_FMA, 0, 0, 0, 0},   // ADDMUL
                 '{default: `LATENCY_FDIVSQRT}, // DIVSQRT
                 '{default: `LATENCY_FNCP},     // NONCOMP
                 '{default: `LATENCY_FCVT}},    // CONV
      UnitTypes:'{'{default: UNIT_FMULADD},     // ADDMUL
                  '{default: UNIT_FDIVSQRT},    // DIVSQRT
                  '{default: UNIT_FNONCOMP},    // NONCOMP
                  '{default: UNIT_FCONV}},      // CONV
      PipeConfig: fpnew_pkg::DISTRIBUTED
    };
    
    wire fpu_ready_in, fpu_valid_in;    
    wire fpu_ready_out, fpu_valid_out;

    reg [TAGW-1:0] fpu_tag_in, fpu_tag_out;
    
    reg [2:0][`NUM_THREADS-1:0][31:0] fpu_operands;   
    
    wire [FMTF_BITS-1:0] fpu_src_fmt = fpnew_pkg::FP32;
    wire [FMTF_BITS-1:0] fpu_dst_fmt = fpnew_pkg::FP32;
    wire [FMTI_BITS-1:0] fpu_int_fmt = fpnew_pkg::INT32;

    wire [`NUM_THREADS-1:0][31:0] fpu_result;
    fpnew_pkg::status_t [`NUM_THREADS-1:0] fpu_status;

    reg [FOP_BITS-1:0] fpu_op;
    reg [`INST_FRM_BITS-1:0] fpu_rnd;
    reg fpu_op_mod;
    reg fpu_has_fflags, fpu_has_fflags_out;

    always @(*) begin
        fpu_op          = fpnew_pkg::SGNJ;
        fpu_rnd         = frm;  
        fpu_op_mod      = 0;        
        fpu_has_fflags  = 1;
        fpu_operands[0] = dataa;
        fpu_operands[1] = datab;
        fpu_operands[2] = datac;

        case (op_type)
            `INST_FPU_ADD: begin
                    fpu_op = fpnew_pkg::ADD;
                    fpu_operands[1] = dataa;
                    fpu_operands[2] = datab;
                end
            `INST_FPU_SUB: begin 
                    fpu_op = fpnew_pkg::ADD; 
                    fpu_operands[1] = dataa;
                    fpu_operands[2] = datab;
                    fpu_op_mod = 1; 
                end
            `INST_FPU_MUL:   begin fpu_op = fpnew_pkg::MUL; end
            `INST_FPU_DIV:   begin fpu_op = fpnew_pkg::DIV; end
            `INST_FPU_SQRT:  begin fpu_op = fpnew_pkg::SQRT; end
            `INST_FPU_MADD:  begin fpu_op = fpnew_pkg::FMADD; end
            `INST_FPU_MSUB:  begin fpu_op = fpnew_pkg::FMADD;  fpu_op_mod = 1; end            
            `INST_FPU_NMADD: begin fpu_op = fpnew_pkg::FNMSUB; fpu_op_mod = 1; end
            `INST_FPU_NMSUB: begin fpu_op = fpnew_pkg::FNMSUB; end
            `INST_FPU_CVTWS: begin fpu_op = fpnew_pkg::F2I; end
            `INST_FPU_CVTWUS:begin fpu_op = fpnew_pkg::F2I; fpu_op_mod = 1; end
            `INST_FPU_CVTSW: begin fpu_op = fpnew_pkg::I2F; end
            `INST_FPU_CVTSWU:begin fpu_op = fpnew_pkg::I2F; fpu_op_mod = 1; end
            `INST_FPU_CLASS: begin fpu_op = fpnew_pkg::CLASSIFY; fpu_has_fflags = 0; end
            `INST_FPU_CMP:   begin fpu_op = fpnew_pkg::CMP; end
            `INST_FPU_MISC:  begin
                case (frm)
                      0: begin fpu_op = fpnew_pkg::SGNJ;   fpu_rnd = `INST_FRM_RNE; fpu_has_fflags = 0; end
                      1: begin fpu_op = fpnew_pkg::SGNJ;   fpu_rnd = `INST_FRM_RTZ; fpu_has_fflags = 0; end
                      2: begin fpu_op = fpnew_pkg::SGNJ;   fpu_rnd = `INST_FRM_RDN; fpu_has_fflags = 0; end
                      3: begin fpu_op = fpnew_pkg::MINMAX; fpu_rnd = `INST_FRM_RNE; end
                      4: begin fpu_op = fpnew_pkg::MINMAX; fpu_rnd = `INST_FRM_RTZ; end    
                default: begin fpu_op = fpnew_pkg::SGNJ;   fpu_rnd = `INST_FRM_RUP; fpu_has_fflags = 0; end
                endcase    
            end
            default:;
        endcase
    end  
    
    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        if (0 == i) begin
            fpnew_top #( 
                .Features       (FPU_FEATURES),
                .Implementation (FPU_IMPLEMENTATION),
                .TagType        (logic[TAGW+1+1-1:0])
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
                .tag_i          ({fpu_tag_in, fpu_has_fflags}),
                .in_valid_i     (fpu_valid_in),
                .in_ready_o     (fpu_ready_in),
                .flush_i        (reset),
                .result_o       (fpu_result[0]),
                .status_o       (fpu_status[0]),
                .tag_o          ({fpu_tag_out, fpu_has_fflags_out}),
                .out_valid_o    (fpu_valid_out),
                .out_ready_i    (fpu_ready_out),
                `UNUSED_PIN (busy_o)
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
                .in_valid_i     (fpu_valid_in),
                `UNUSED_PIN (in_ready_o),
                .flush_i        (reset),
                .result_o       (fpu_result[i]),
                .status_o       (fpu_status[i]),
                `UNUSED_PIN (tag_o),
                `UNUSED_PIN (out_valid_o),
                .out_ready_i    (fpu_ready_out),
                `UNUSED_PIN (busy_o)
            );
        end
    end

    assign fpu_valid_in = valid_in;
    assign ready_in = fpu_ready_in;

    assign fpu_tag_in = tag_in;    
    assign tag_out = fpu_tag_out;

    assign result = fpu_result;
    
    assign has_fflags = fpu_has_fflags_out;   
    assign fflags = fpu_status;

    assign valid_out = fpu_valid_out;    
    assign fpu_ready_out = ready_out;

endmodule 
`TRACING_ON