`include "VX_define.vh"

module VX_tcu_fedp_drl import VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter USE_VALID = 0,
    parameter LATENCY = 0,
    parameter N = 2,
    parameter W = 25
) (
    input  wire clk,
    input  wire reset,
    input  wire enable,
    input  wire [TCU_MAX_INPUTS-1:0] vld_mask,
    input  wire [3:0] fmt_s,
    input  wire [3:0] fmt_d,
    input  wire [N-1:0][31:0] a_row,
    input  wire [N-1:0][31:0] b_col,
    input  wire [31:0]        c_val,
    output wire [31:0]        d_val
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (fmt_d)

    localparam TCK     = 2 * N;
    localparam EXP_W   = 10;
    localparam SHIFT_W = 8;
    localparam EXC_W   = $bits(fedp_excep_t);
    localparam C_HI_W  = 7;
    localparam HR = $clog2(TCK+1);

    localparam ALN_SIG_W = W + 2;
    localparam ACC_SIG_W = W + 1 + HR;

    // Latency Configuration
    localparam MUL_LATENCY = 1;
    localparam ALN_LATENCY = 1;
    localparam ACC_LATENCY = 1;
    localparam NRM_LATENCY = 1;
    localparam TOTAL_LATENCY = MUL_LATENCY + ALN_LATENCY + ACC_LATENCY + NRM_LATENCY;

    `STATIC_ASSERT (LATENCY == 0 || LATENCY == TOTAL_LATENCY,
        ("invalid latency! expected=%0d, actual=%0d", TOTAL_LATENCY, LATENCY))

    localparam S0_IDX = 0;
    localparam S1_IDX = S0_IDX + MUL_LATENCY;
    localparam S2_IDX = S1_IDX + ALN_LATENCY;
    localparam S3_IDX = S2_IDX + ACC_LATENCY;
    localparam S4_IDX = S3_IDX + NRM_LATENCY;
    `UNUSED_PARAM(S4_IDX)

    reg [TOTAL_LATENCY-1:0] vld_pipe_r;
    reg [TOTAL_LATENCY-1:0][31:0] req_pipe_r;
    reg [31:0] req_id;
    wire vld_any = (|vld_mask) && USE_VALID;

    always_ff @(posedge clk) begin
        if (reset) begin
            vld_pipe_r <= '0;
            req_pipe_r <= '0;
            req_id     <= 0;
        end else if (enable) begin
            vld_pipe_r <= {vld_pipe_r[TOTAL_LATENCY-2:0], vld_any};
            req_pipe_r <= {req_pipe_r[TOTAL_LATENCY-2:0], req_id};
            req_id     <= req_id + 32'(vld_any);
        end
    end

    wire [TOTAL_LATENCY:0] vld_pipe = {vld_pipe_r, (~reset && enable && vld_any)};
    wire [TOTAL_LATENCY:0][31:0] req_pipe = {req_pipe_r, req_id};

    // Stage 1: Multiply & Max Exponent
    wire [EXP_W-1:0]          max_exp;
    wire [TCK:0][SHIFT_W-1:0] shift_amt;
    wire [TCK:0][W-1:0]   raw_sigs;
    fedp_excep_t              exceptions;
    wire [TCK-1:0]            lane_mask;

    wire is_int = fmt_s[3];

    // Integer C-term extraction + correction
    wire [7:0] cval_top = c_val[31:24];
    wire [6:0] cval_hi = cval_top[7:1] + 7'(cval_top[0]);

    VX_tcu_drl_mul_exp #(
        .N (N),
        .W (W),
        .WA(ACC_SIG_W),
        .EXP_W (EXP_W)
    ) mul_exp (
        .clk(clk),
        .valid_in(vld_pipe[S0_IDX]),
        .req_id(req_pipe[S0_IDX]),
        .fmt_s(fmt_s),
        .a_row(a_row),
        .b_col(b_col),
        .c_val(c_val),
        .vld_mask(vld_mask | TCU_MAX_INPUTS'(!USE_VALID)),
        .max_exp(max_exp),
        .shift_amt(shift_amt),
        .raw_sigs(raw_sigs),
        .exceptions(exceptions),
        .lane_mask(lane_mask)
    );

    wire [EXP_W-1:0]          s1_max_exp;
    fedp_excep_t              s1_exceptions;
    wire [TCK-1:0]            s1_lane_mask;
    wire [TCK:0][SHIFT_W-1:0] s1_shift_amt;
    wire [TCK:0][W-1:0]   s1_raw_sig;
    wire                      s1_is_int;
    wire [C_HI_W-1:0]         s1_cval_hi;

    VX_pipe_register #(
        .DATAW (EXP_W + EXC_W + TCK + SHIFT_W + W + C_HI_W + 1),
        .DEPTH (MUL_LATENCY)
    ) pipe_fmul_ctrl (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .data_in ({max_exp,    exceptions,    lane_mask,    shift_amt[TCK],    raw_sigs[TCK],   cval_hi,    is_int}),
        .data_out({s1_max_exp, s1_exceptions, s1_lane_mask, s1_shift_amt[TCK], s1_raw_sig[TCK], s1_cval_hi, s1_is_int})
    );
    for (genvar i = 0; i < TCK; i++) begin : g_fmul_lane
        VX_pipe_register #(
            .DATAW (SHIFT_W + W),
            .DEPTH (MUL_LATENCY))
        pipe_fmul_lane (
            .clk(clk),
            .reset(reset),
            .enable(enable & (lane_mask[i] || !USE_VALID)),
            .data_in ({shift_amt[i],    raw_sigs[i]}),
            .data_out({s1_shift_amt[i], s1_raw_sig[i]})
        );
    end

    // Stage 2: Alignment
    wire [TCK:0][ALN_SIG_W-1:0] s1_aln_sigs;
    wire [TCK:0]                s1_aln_sticky;

    VX_tcu_drl_align #(
        .N (TCK+1),
        .WI(W),
        .WO(ALN_SIG_W)
    ) sigs_aln (
        .clk(clk),
        .valid_in(vld_pipe[S1_IDX]),
        .req_id(req_pipe[S1_IDX]),
        .shift_amt(s1_shift_amt),
        .sigs_in(s1_raw_sig),
        .is_int(s1_is_int),
        .sigs_out(s1_aln_sigs),
        .sticky_bits(s1_aln_sticky)
    );

    wire [EXP_W-1:0]            s2_max_exp;
    fedp_excep_t                s2_exceptions;
    wire [TCK-1:0]              s2_lane_mask;
    wire [TCK:0][ALN_SIG_W-1:0] s2_aln_sigs;
    wire [TCK:0]                s2_aln_sticky;
    wire                        s2_is_int;
    wire [C_HI_W-1:0]           s2_cval_hi;

    VX_pipe_register #(
        .DATAW (EXP_W + EXC_W + TCK + ALN_SIG_W + 1 + C_HI_W + 1),
        .DEPTH (ALN_LATENCY)
    ) pipe_aln_ctrl (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .data_in ({s1_max_exp, s1_exceptions, s1_lane_mask, s1_aln_sigs[TCK], s1_aln_sticky[TCK], s1_cval_hi, s1_is_int}),
        .data_out({s2_max_exp, s2_exceptions, s2_lane_mask, s2_aln_sigs[TCK], s2_aln_sticky[TCK], s2_cval_hi, s2_is_int})
    );
    for (genvar i = 0; i < TCK; i++) begin : g_aln_lane
        VX_pipe_register #(
            .DATAW (ALN_SIG_W + 1),
            .DEPTH (ALN_LATENCY))
        pipe_aln_lane (
            .clk(clk),
            .reset(reset),
            .enable(enable & (s1_lane_mask[i] || !USE_VALID)),
            .data_in ({s1_aln_sigs[i], s1_aln_sticky[i]}),
            .data_out({s2_aln_sigs[i], s2_aln_sticky[i]})
        );
    end

    // Stage 3: Accumulation
    wire [ACC_SIG_W-1:0] s2_acc_sum;
    wire                 s2_acc_sticky;

    VX_tcu_drl_acc #(
        .N (TCK+1),
        .WI(ALN_SIG_W),
        .WO(ACC_SIG_W)
    ) csa_acc (
        .clk(clk),
        .valid_in(vld_pipe[S2_IDX]),
        .req_id(req_pipe[S2_IDX]),
        .lane_mask(s2_lane_mask),
        .sigs_in(s2_aln_sigs),
        .sticky_in(s2_aln_sticky),
        .sig_out(s2_acc_sum),
        .sticky_out(s2_acc_sticky)
    );

    wire [EXP_W-1:0]      s3_max_exp;
    wire [ACC_SIG_W-1:0]  s3_acc_sum;
    fedp_excep_t          s3_exceptions;
    wire                  s3_acc_sticky;
    wire                  s3_is_int;
    wire [C_HI_W-1:0]     s3_cval_hi;

    VX_pipe_register #(
        .DATAW (EXP_W + ACC_SIG_W + EXC_W + 1 + C_HI_W + 1),
        .DEPTH (ACC_LATENCY)
    ) pipe_acc (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .data_in ({s2_max_exp, s2_acc_sum, s2_exceptions, s2_acc_sticky, s2_cval_hi, s2_is_int}),
        .data_out({s3_max_exp, s3_acc_sum, s3_exceptions, s3_acc_sticky, s3_cval_hi, s3_is_int})
    );

    // Stage 4: Normalization and rounding
    wire [31:0] final_result;

    VX_tcu_drl_norm_round #(
        .EXP_W (EXP_W),
        .C_HI_W(C_HI_W),
        .WA (ACC_SIG_W)
    ) norm_round (
        .clk(clk),
        .valid_in(vld_pipe[S3_IDX]),
        .req_id(req_pipe[S3_IDX]),
        .max_exp(s3_max_exp),
        .acc_sig(s3_acc_sum),
        .sticky_in(s3_acc_sticky),
        .exceptions(s3_exceptions),
        .cval_hi(s3_cval_hi),
        .is_int(s3_is_int),
        .result(final_result)
    );

    VX_pipe_register #(
        .DATAW (32),
        .DEPTH (NRM_LATENCY)
    ) pipe_norm_round (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .data_in (final_result),
        .data_out(d_val)
    );

`ifdef DBG_TRACE_TCU
    // Stage 0: Setup
    always_ff @(posedge clk) begin
        if (vld_pipe[S0_IDX]) begin
            `TRACE(4, ("%t: %s FEDP-S0(%0d): fmt_s=%0d, a_row=",
                $time, INSTANCE_ID, req_pipe[S0_IDX], fmt_s));
            `TRACE_ARRAY1D(4, "0x%0h", a_row, N)
            `TRACE(4, (", b_col="))
            `TRACE_ARRAY1D(4, "0x%0h", b_col, N)
            `TRACE(4, (", c_val=0x%0h, vld_mask=%b\n", c_val, vld_mask))
        end
    end

    // Stage 1: Mul/Exp
    always_ff @(posedge clk) begin
        if (vld_pipe[S1_IDX]) begin
            `TRACE(4, ("%t: %s FEDP-S1(%0d): is_int=%b, cval_hi=0x%0h, max_exp=0x%0h, shift_amt=",
                $time, INSTANCE_ID, s1_is_int, s1_cval_hi, req_pipe[S1_IDX], s1_max_exp));
            `TRACE_ARRAY1D(4, "0x%0h", s1_shift_amt, (TCK+1))
            `TRACE(4, (", raw_sig="))
            `TRACE_ARRAY1D(4, "0x%0h", s1_raw_sig, (TCK+1))
            `TRACE(4, (", exceptions=%0b, lane_mask=%b\n", s1_exceptions, s1_lane_mask))
        end
    end

    // Stage 2: Alignment
    always_ff @(posedge clk) begin
        if (vld_pipe[S2_IDX]) begin
            `TRACE(4, ("%t: %s FEDP-S2(%0d): is_int=%b, cval_hi=0x%0h, max_exp=0x%0h, aln_sig=",
                $time, INSTANCE_ID, req_pipe[S2_IDX], s2_is_int, s2_cval_hi, s2_max_exp));
            `TRACE_ARRAY1D(4, "0x%0h", s2_aln_sigs, (TCK+1))
            `TRACE(4, (", sticky_bits="))
            `TRACE_ARRAY1D(4, "0b%b", s2_aln_sticky, (TCK+1))
            `TRACE(4, (", exceptions=%0b\n", s2_exceptions))
        end
    end

    // Stage 3: Accumulation
    always_ff @(posedge clk) begin
        if (vld_pipe[S3_IDX]) begin
            `TRACE(4, ("%t: %s FEDP-S3(%0d): is_int=%b, cval_hi=0x%0h, acc_sig=0x%0h, max_exp=0x%0h, sticky=%b, exceptions=%0b\n",
                $time, INSTANCE_ID, req_pipe[S3_IDX], s3_is_int, s3_cval_hi, s3_acc_sum, s3_max_exp, s3_acc_sticky, s3_exceptions));
        end
    end

    // Stage 4: Norm/Round
    always_ff @(posedge clk) begin
        if (vld_pipe[S4_IDX]) begin
            `TRACE(4, ("%t: %s FEDP-S4(%0d): result=0x%0h\n", $time, INSTANCE_ID, req_pipe[S4_IDX], d_val));
        end
    end
`endif // DBG_TRACE_TCU

endmodule
