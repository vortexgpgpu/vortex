`include "VX_fpu_define.vh"

`ifdef FPU_DSP

module VX_fpu_dsp #(
    parameter NUM_LANES = 4, 
    parameter TAGW      = 4
) (
    input wire clk,
    input wire reset,

    input wire  valid_in,
    output wire ready_in,

    input wire [TAGW-1:0] tag_in,
    
    input wire [`INST_FPU_BITS-1:0] op_type,
    input wire [`INST_FMT_BITS-1:0] fmt,
    input wire [`INST_FRM_BITS-1:0] frm,

    input wire [NUM_LANES-1:0][`XLEN-1:0]  dataa,
    input wire [NUM_LANES-1:0][`XLEN-1:0]  datab,
    input wire [NUM_LANES-1:0][`XLEN-1:0]  datac,
    output wire [NUM_LANES-1:0][`XLEN-1:0] result, 

    output wire has_fflags,
    output wire [NUM_LANES-1:0][`FP_FLAGS_BITS-1:0] fflags,

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);
    localparam FPU_FMA  = 0;
    localparam FPU_DIV  = 1;
    localparam FPU_SQRT = 2;
    localparam FPU_CVT  = 3;
    localparam FPU_NCP  = 4;
    localparam NUM_FPC  = 5;
    localparam FPC_BITS = `LOG2UP(NUM_FPC);

    localparam RSP_ARB_DATAW = (NUM_LANES * 32) + 1 + (NUM_LANES * $bits(fflags_t)) + TAGW;

    `UNUSED_VAR (fmt)
    
    wire [NUM_FPC-1:0] per_core_ready_in;
    wire [NUM_FPC-1:0][NUM_LANES-1:0][31:0] per_core_result;
    wire [NUM_FPC-1:0][TAGW-1:0] per_core_tag_out;
    wire [NUM_FPC-1:0] per_core_ready_out;
    wire [NUM_FPC-1:0] per_core_valid_out;
    
    wire [NUM_FPC-1:0] per_core_has_fflags;  
    fflags_t [NUM_FPC-1:0][NUM_LANES-1:0] per_core_fflags;  

    reg [FPC_BITS-1:0] core_select;
    reg do_madd, do_sub, do_neg, is_itof, is_signed;

    always @(*) begin
        do_madd   = 0;
        do_sub    = 0;        
        do_neg    = 0;
        is_itof   = 0;
        is_signed = 0;
        case (op_type)
            `INST_FPU_ADD:    begin core_select = FPU_FMA; end
            `INST_FPU_SUB:    begin core_select = FPU_FMA; do_sub = 1; end
            `INST_FPU_MUL:    begin core_select = FPU_FMA; do_neg = 1; end
            `INST_FPU_MADD:   begin core_select = FPU_FMA; do_madd = 1; end
            `INST_FPU_MSUB:   begin core_select = FPU_FMA; do_madd = 1; do_sub = 1; end
            `INST_FPU_NMADD:  begin core_select = FPU_FMA; do_madd = 1; do_neg = 1; end
            `INST_FPU_NMSUB:  begin core_select = FPU_FMA; do_madd = 1; do_sub = 1; do_neg = 1; end
            `INST_FPU_DIV:    begin core_select = FPU_DIV; end
            `INST_FPU_SQRT:   begin core_select = FPU_SQRT; end
            `INST_FPU_F2I:    begin core_select = FPU_CVT; is_signed = 1; end
            `INST_FPU_F2U:    begin core_select = FPU_CVT; end
            `INST_FPU_I2F:    begin core_select = FPU_CVT; is_itof = 1; is_signed = 1; end
            `INST_FPU_U2F:    begin core_select = FPU_CVT; is_itof = 1; end
            default:          begin core_select = FPU_NCP; end
        endcase
    end

    `RESET_RELAY (fma_reset, reset);
    `RESET_RELAY (div_reset, reset);
    `RESET_RELAY (sqrt_reset, reset);
    `RESET_RELAY (cvt_reset, reset);
    `RESET_RELAY (ncp_reset, reset);

    wire [NUM_LANES-1:0][31:0] dataa_s;
    wire [NUM_LANES-1:0][31:0] datab_s;
    wire [NUM_LANES-1:0][31:0] datac_s;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign dataa_s[i] = dataa[i][31:0];
        assign datab_s[i] = datab[i][31:0];
        assign datac_s[i] = datac[i][31:0];
    end

    `UNUSED_VAR (dataa)
    `UNUSED_VAR (datab)
    `UNUSED_VAR (datac)

    VX_fpu_fma #(
        .NUM_LANES (NUM_LANES),
        .TAGW      (TAGW)
    ) fp_fma (
        .clk        (clk), 
        .reset      (fma_reset),   
        .valid_in   (valid_in && (core_select == FPU_FMA)),
        .ready_in   (per_core_ready_in[FPU_FMA]),    
        .tag_in     (tag_in),  
        .frm        (frm),
        .do_madd    (do_madd),
        .do_sub     (do_sub),
        .do_neg     (do_neg),
        .dataa      (dataa_s), 
        .datab      (datab_s),    
        .datac      (datac_s),   
        .has_fflags (per_core_has_fflags[FPU_FMA]),
        .fflags     (per_core_fflags[FPU_FMA]),
        .result     (per_core_result[FPU_FMA]),
        .tag_out    (per_core_tag_out[FPU_FMA]),
        .ready_out  (per_core_ready_out[FPU_FMA]),
        .valid_out  (per_core_valid_out[FPU_FMA])
    );

    VX_fpu_div #(
        .NUM_LANES (NUM_LANES),
        .TAGW      (TAGW)
    ) fp_div (
        .clk        (clk), 
        .reset      (div_reset),   
        .valid_in   (valid_in && (core_select == FPU_DIV)),
        .ready_in   (per_core_ready_in[FPU_DIV]),    
        .tag_in     (tag_in),
        .frm        (frm),  
        .dataa      (dataa_s), 
        .datab      (datab_s),   
        .has_fflags (per_core_has_fflags[FPU_DIV]),
        .fflags     (per_core_fflags[FPU_DIV]),   
        .result     (per_core_result[FPU_DIV]),
        .tag_out    (per_core_tag_out[FPU_DIV]),
        .valid_out  (per_core_valid_out[FPU_DIV]),
        .ready_out  (per_core_ready_out[FPU_DIV])
    );

    VX_fpu_sqrt #(
        .NUM_LANES (NUM_LANES),
        .TAGW      (TAGW)
    ) fp_sqrt (
        .clk        (clk), 
        .reset      (sqrt_reset),   
        .valid_in   (valid_in && (core_select == FPU_SQRT)),
        .ready_in   (per_core_ready_in[FPU_SQRT]),    
        .tag_in     (tag_in),
        .frm        (frm),    
        .dataa      (dataa_s), 
        .has_fflags (per_core_has_fflags[FPU_SQRT]),
        .fflags     (per_core_fflags[FPU_SQRT]),
        .result     (per_core_result[FPU_SQRT]),
        .tag_out    (per_core_tag_out[FPU_SQRT]),
        .valid_out  (per_core_valid_out[FPU_SQRT]),
        .ready_out  (per_core_ready_out[FPU_SQRT])
    );

    wire cvt_rt_int_in = ~is_itof;
    wire cvt_rt_int_out;

    VX_fpu_cvt #(
        .NUM_LANES (NUM_LANES),
        .TAGW      (TAGW+1)
    ) fp_cvt (
        .clk        (clk), 
        .reset      (cvt_reset),   
        .valid_in   (valid_in && (core_select == FPU_CVT)),
        .ready_in   (per_core_ready_in[FPU_CVT]),    
        .tag_in     ({cvt_rt_int_in, tag_in}), 
        .frm        (frm),
        .is_itof    (is_itof),   
        .is_signed  (is_signed),        
        .dataa      (dataa_s),  
        .has_fflags (per_core_has_fflags[FPU_CVT]),
        .fflags     (per_core_fflags[FPU_CVT]),
        .result     (per_core_result[FPU_CVT]),
        .tag_out    ({cvt_rt_int_out, per_core_tag_out[FPU_CVT]}),
        .valid_out  (per_core_valid_out[FPU_CVT]),
        .ready_out  (per_core_ready_out[FPU_CVT])
    );

    wire ncp_rt_int_in = (op_type == `INST_FPU_CMP)
                      || `INST_FPU_IS_CLASS(op_type, frm) 
                      || `INST_FPU_IS_MVXW(op_type, frm);
    wire ncp_rt_int_out;

    wire ncp_rt_sext_in = `INST_FPU_IS_MVXW(op_type, frm);
    wire ncp_rt_sext_out;
    
    VX_fpu_ncomp #(
        .NUM_LANES (NUM_LANES),
        .TAGW      (TAGW+2)
    ) fp_ncomp (
        .clk        (clk),
        .reset      (ncp_reset),   
        .valid_in   (valid_in && (core_select == FPU_NCP)),
        .ready_in   (per_core_ready_in[FPU_NCP]),        
        .tag_in     ({ncp_rt_sext_in, ncp_rt_int_in, tag_in}),
        .op_type    (op_type),
        .frm        (frm),
        .dataa      (dataa_s),
        .datab      (datab_s),        
        .result     (per_core_result[FPU_NCP]), 
        .has_fflags (per_core_has_fflags[FPU_NCP]),
        .fflags     (per_core_fflags[FPU_NCP]),
        .tag_out    ({ncp_rt_sext_out, ncp_rt_int_out, per_core_tag_out[FPU_NCP]}),
        .valid_out  (per_core_valid_out[FPU_NCP]),
        .ready_out  (per_core_ready_out[FPU_NCP])
    );

    ///////////////////////////////////////////////////////////////////////////

    reg [NUM_FPC-1:0][RSP_ARB_DATAW+2-1:0] per_core_data_out;
    
    always @(*) begin
        for (integer i = 0; i < NUM_FPC; ++i) begin
            per_core_data_out[i][RSP_ARB_DATAW+1:2] = {per_core_result[i], per_core_has_fflags[i], per_core_fflags[i], per_core_tag_out[i]};
            per_core_data_out[i][1:0] = '0;
        end        
        per_core_data_out[FPU_CVT][1:0] = {1'b1, cvt_rt_int_out};
        per_core_data_out[FPU_NCP][1:0] = {ncp_rt_sext_out, ncp_rt_int_out};
    end

    wire [NUM_LANES-1:0][31:0] result_s;
    wire [1:0] op_rt_int_out;

    VX_stream_arb #(
        .NUM_INPUTS (NUM_FPC),
        .DATAW      (RSP_ARB_DATAW + 2),        
        .ARBITER    ("R"),
        .BUFFERED   (2)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (per_core_valid_out),        
        .ready_in  (per_core_ready_out),
        .data_in   (per_core_data_out),
        .data_out  ({result_s, has_fflags, fflags, tag_out, op_rt_int_out}),
        .valid_out (valid_out),
        .ready_out (ready_out)
    );

`ifndef FPU_RV64_F
    `UNUSED_VAR (op_rt_int_out)
`endif

    for (genvar i = 0; i < NUM_LANES; ++i) begin        
    `ifdef FPU_RV64_F
        reg [`XLEN-1:0] result_r;
        always @(*) begin
            case (op_rt_int_out)
            2'b11:   result_r = `XLEN'($signed(result_s[i]));
            2'b01:   result_r = {32'h00000000, result_s[i]};
            default: result_r = {32'hffffffff, result_s[i]};
            endcase
        end
        assign result[i] = result_r;
    `else
        assign result[i] = result_s[i];
    `endif
    end

    // can accept new request?
    assign ready_in = per_core_ready_in[core_select];

endmodule
`endif 
