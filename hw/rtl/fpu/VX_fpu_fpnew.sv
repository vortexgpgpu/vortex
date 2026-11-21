// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_fpu_define.vh"

`ifdef VX_CFG_FPU_TYPE_FPNEW

module VX_fpu_fpnew
    import VX_gpu_pkg::*;
    import VX_fpu_pkg::*;
    import fpnew_pkg::*;
    import cf_math_pkg::*;
    import defs_div_sqrt_mvp::*;
#(
    parameter NUM_LANES = 1,
    parameter TAG_WIDTH = 1,
    parameter OUT_BUF   = 0
) (
    input wire clk,
    input wire reset,

    input wire  valid_in,
    output wire ready_in,

    input wire [NUM_LANES-1:0] mask_in,

    input wire [TAG_WIDTH-1:0] tag_in,

    input wire [INST_FPU_BITS-1:0] op_type,
    input wire [INST_FMT_BITS-1:0] dst_fmt,
    input wire [INST_FMT_BITS-1:0] src_fmt,
    input wire                     op_mod,
    input wire [INST_FRM_BITS-1:0] frm,

    input wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0]  dataa,
    input wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0]  datab,
    input wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0]  datac,
    output wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] result,

    output wire has_fflags,
    output wire [`FP_FLAGS_BITS-1:0] fflags,

    output wire [TAG_WIDTH-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);
    // FPnew instantiates a full lane of every unit; it has no PE serializer.
    `STATIC_ASSERT((`VX_CFG_FMA_PE_RATIO == 1) && (`VX_CFG_FMA64_PE_RATIO == 1)
                && (`VX_CFG_FDIV_PE_RATIO == 1) && (`VX_CFG_FSQRT_PE_RATIO == 1)
                && (`VX_CFG_FCVT_PE_RATIO == 1) && (`VX_CFG_FNCP_PE_RATIO == 1),
        ("FPU_TYPE_FPNEW does not support FPU PE ratios other than 1"))

    localparam LATENCY_FDIVSQRT = `MAX(`VX_CFG_FDIV_LATENCY, `VX_CFG_FSQRT_LATENCY);
    localparam RSP_DATAW = (NUM_LANES * `VX_CFG_XLEN) + 1 + $bits(fflags_t) + TAG_WIDTH;

`ifdef VX_CFG_EXT_ZFHMIN_ENABLE
    localparam [4:0] FP16_FMT_MASK = 5'b00100;
`else
    localparam [4:0] FP16_FMT_MASK = 5'b00000;
`endif

    localparam fpnew_pkg::fpu_features_t FPU_FEATURES = '{
        Width:         unsigned'(`VX_CFG_XLEN),
        EnableVectors: 1'b0,
    `ifdef VX_CFG_XLEN_64
        EnableNanBox:  1'b1,
    `ifdef VX_CFG_FLEN_64
        FpFmtMask:     5'b11000 | FP16_FMT_MASK,
    `else
        FpFmtMask:     5'b11000 | FP16_FMT_MASK, // FP32+FP64 enabled to avoid CVT bug in FpNew
    `endif
        IntFmtMask:    4'b0011
    `else
        EnableNanBox:  (FP16_FMT_MASK != 0),
        FpFmtMask:     5'b10000 | FP16_FMT_MASK,
        IntFmtMask:    4'b0010
    `endif
    };

    // Div/sqrt has operand-dependent latency, so it runs on its own per-lane
    // instance; the remaining groups are fixed-latency and stay in lockstep.
    localparam fpnew_pkg::fpu_implementation_t FPU_IMPL_LOCKSTEP = '{
      PipeRegs:'{'{`VX_CFG_FMA_LATENCY, 0, 0, 0, 0}, // ADDMUL
                 '{default: 0}, // DIVSQRT
                 '{default: `VX_CFG_FNCP_LATENCY}, // NONCOMP
                 '{default: `VX_CFG_FCVT_LATENCY}}, // CONV
      UnitTypes:'{'{default: fpnew_pkg::PARALLEL}, // ADDMUL
                  '{default: fpnew_pkg::DISABLED}, // DIVSQRT
                  '{default: fpnew_pkg::PARALLEL}, // NONCOMP
                  '{default: fpnew_pkg::MERGED}}, // CONV
      PipeConfig: fpnew_pkg::DISTRIBUTED
    };

    localparam fpnew_pkg::fpu_implementation_t FPU_IMPL_DIVSQRT = '{
      PipeRegs:'{'{default: 0}, // ADDMUL
                 '{default: unsigned'(LATENCY_FDIVSQRT)}, // DIVSQRT
                 '{default: 0}, // NONCOMP
                 '{default: 0}}, // CONV
      UnitTypes:'{'{default: fpnew_pkg::DISABLED}, // ADDMUL
                  '{default: fpnew_pkg::MERGED}, // DIVSQRT
                  '{default: fpnew_pkg::DISABLED}, // NONCOMP
                  '{default: fpnew_pkg::DISABLED}}, // CONV
      PipeConfig: fpnew_pkg::DISTRIBUTED
    };

    localparam LS_TAGW  = TAG_WIDTH + 1 + NUM_LANES; // {tag, has_fflags, mask}
    localparam DS_TAGW  = TAG_WIDTH + NUM_LANES;     // {tag, mask}
    localparam DS_DATAW = DS_TAGW + $bits(fflags_t) + `VX_CFG_XLEN;

    logic [2:0][NUM_LANES-1:0][`VX_CFG_XLEN-1:0] fpu_operands;

    fpnew_pkg::operation_e fpu_op;
    reg [INST_FRM_BITS-1:0] fpu_rnd;
    reg fpu_op_mod;
    reg fpu_has_fflags;
    fpnew_pkg::fp_format_e fpu_src_fmt, fpu_dst_fmt;
    fpnew_pkg::int_format_e fpu_int_fmt;

    function automatic fpnew_pkg::fp_format_e to_fpnew_fmt(input logic [INST_FMT_BITS-1:0] f);
        case (f)
        `ifdef VX_CFG_FLEN_64
            INST_FMT_D: to_fpnew_fmt = fpnew_pkg::FP64;
        `endif
        `ifdef VX_CFG_EXT_ZFHMIN_ENABLE
            INST_FMT_H: to_fpnew_fmt = fpnew_pkg::FP16;
        `endif
            default:    to_fpnew_fmt = fpnew_pkg::FP32;
        endcase
    endfunction

    always @(*) begin
        fpu_op          = fpnew_pkg::operation_e'('x);
        fpu_rnd         = frm;
        fpu_op_mod      = 0;
        fpu_has_fflags  = 1;
        fpu_operands[0] = dataa;
        fpu_operands[1] = datab;
        fpu_operands[2] = datac;
        fpu_dst_fmt     = to_fpnew_fmt(dst_fmt);
        fpu_src_fmt     = fpu_dst_fmt;
        fpu_int_fmt     = fpnew_pkg::INT32;

        case (op_type)
            INST_FPU_ADD: begin
                fpu_op = fpnew_pkg::ADD;
                fpu_operands[1] = dataa;
                fpu_operands[2] = datab;
                fpu_op_mod = op_mod; // FADD or FSUB
            end
            INST_FPU_MUL:   begin fpu_op = fpnew_pkg::MUL; end
            INST_FPU_MADD:  begin fpu_op = fpnew_pkg::FMADD; fpu_op_mod = op_mod; end
            INST_FPU_NMADD: begin fpu_op = fpnew_pkg::FNMSUB; fpu_op_mod = ~op_mod; end
            INST_FPU_DIV:   begin fpu_op = fpnew_pkg::DIV; end
            INST_FPU_SQRT:  begin fpu_op = fpnew_pkg::SQRT; end
            INST_FPU_F2F: begin
                fpu_op      = fpnew_pkg::F2F;
                fpu_src_fmt = to_fpnew_fmt(src_fmt);
            end
            INST_FPU_F2I,
            INST_FPU_F2U: begin
                fpu_op      = fpnew_pkg::F2I;
                fpu_op_mod  = op_type[0];
                fpu_src_fmt = to_fpnew_fmt(src_fmt);
                fpu_dst_fmt = fpu_src_fmt;
            `ifdef VX_CFG_XLEN_64
                if (dst_fmt == INST_FMT_L) begin
                    fpu_int_fmt = fpnew_pkg::INT64;
                end
            `endif
            end
            INST_FPU_I2F,
            INST_FPU_U2F: begin
                fpu_op      = fpnew_pkg::I2F;
                fpu_op_mod  = op_type[0];
            `ifdef VX_CFG_XLEN_64
                if (src_fmt == INST_FMT_L) begin
                    fpu_int_fmt = fpnew_pkg::INT64;
                end
            `endif
            end
            INST_FPU_CMP: begin fpu_op = fpnew_pkg::CMP; end
            INST_FPU_MISC:begin
                case (frm)
                    0,1,2: begin fpu_op = fpnew_pkg::SGNJ; fpu_rnd = {1'b0, frm[1:0]}; fpu_has_fflags = 0; end // FSGNJ
                    3:     begin fpu_op = fpnew_pkg::CLASSIFY; fpu_has_fflags = 0; end // CLASS
                    4,5:   begin fpu_op = fpnew_pkg::SGNJ; fpu_rnd = 3'b011; fpu_op_mod = ~frm[0]; fpu_has_fflags = 0; end // MVXW, MVWX
                    6,7:   begin fpu_op = fpnew_pkg::MINMAX; fpu_rnd = {2'b00, frm[0]}; end // MIN, MAX
                endcase
            end
            default:;
        endcase
    end

    wire is_divsqrt = (op_type == INST_FPU_DIV) || (op_type == INST_FPU_SQRT);

    wire ls_valid_in = valid_in && ~is_divsqrt;
    wire [NUM_LANES-1:0] ls_ready_in;
    wire [NUM_LANES-1:0] ls_valid_out;
    wire ls_ready_out;
    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] ls_result;
    fflags_t [NUM_LANES-1:0] ls_status;
    wire [NUM_LANES-1:0][LS_TAGW-1:0] ls_tag;

    wire ds_valid_in, ds_ready_in;
    wire [NUM_LANES-1:0] ds_lane_valid_in, ds_lane_ready_in;
    wire [NUM_LANES-1:0] ds_lane_valid_out, ds_lane_ready_out;
    wire [NUM_LANES-1:0][DS_DATAW-1:0] ds_lane_data_out;
    wire ds_valid_out, ds_ready_out;
    wire [NUM_LANES-1:0][DS_DATAW-1:0] ds_data_out;

    // Each lane's div/sqrt unit is in order and sees every instruction once, so
    // the k-th result of every lane belongs to the same instruction.
    VX_stream_fork #(
        .NUM_OUTPUTS (NUM_LANES),
        .DATAW       (1),
        .EAGER       (1)
    ) ds_fork (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (ds_valid_in),
        .data_in   (1'b0),
        .ready_in  (ds_ready_in),
        .valid_out (ds_lane_valid_in),
        `UNUSED_PIN (data_out),
        .ready_out (ds_lane_ready_in)
    );

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_lanes
        wire [`VX_CFG_XLEN-1:0] lane_ds_result;
        fflags_t lane_ds_status;
        wire [DS_TAGW-1:0] lane_ds_tag;
        wire [1:0][`VX_CFG_XLEN-1:0] ds_operands;

        for (genvar j = 0; j < 2; ++j) begin : g_ds_operands
        `ifdef VX_CFG_XLEN_64
            // fpnew's div/sqrt ignores NaN-boxing: an improperly boxed single
            // operand must read as the canonical NaN. Sqrt has no operand B,
            // but the unit still tests it for NaN.
            wire is_used = (j == 0) || (fpu_op == fpnew_pkg::DIV);
            wire unboxed = is_used && (fpu_dst_fmt == fpnew_pkg::FP32) && ~(& fpu_operands[j][i][63:32]);
            assign ds_operands[j] = unboxed ? 64'hFFFFFFFF7FC00000 : fpu_operands[j][i];
        `else
            assign ds_operands[j] = fpu_operands[j][i];
        `endif
        end

        fpnew_top #(
            .Features       (FPU_FEATURES),
            .Implementation (FPU_IMPL_LOCKSTEP),
            .TagType        (logic[LS_TAGW-1:0])
        ) fpnew_lockstep (
            .clk_i          (clk),
            .rst_ni         (~reset),
            .operands_i     ({fpu_operands[2][i], fpu_operands[1][i], fpu_operands[0][i]}),
            .rnd_mode_i     (fpnew_pkg::roundmode_e'(fpu_rnd)),
            .op_i           (fpu_op),
            .op_mod_i       (fpu_op_mod),
            .src_fmt_i      (fpu_src_fmt),
            .dst_fmt_i      (fpu_dst_fmt),
            .int_fmt_i      (fpu_int_fmt),
            .vectorial_op_i (1'b0),
            .simd_mask_i    (1'b1),
            .tag_i          ({tag_in, fpu_has_fflags, mask_in}),
            .in_valid_i     (ls_valid_in),
            .in_ready_o     (ls_ready_in[i]),
            .flush_i        (1'b0),
            .result_o       (ls_result[i]),
            .status_o       (ls_status[i]),
            .tag_o          (ls_tag[i]),
            .out_valid_o    (ls_valid_out[i]),
            .out_ready_i    (ls_ready_out),
            `UNUSED_PIN (busy_o)
        );

        fpnew_top #(
            .Features       (FPU_FEATURES),
            .Implementation (FPU_IMPL_DIVSQRT),
            .TagType        (logic[DS_TAGW-1:0])
        ) fpnew_divsqrt (
            .clk_i          (clk),
            .rst_ni         (~reset),
            .operands_i     ({fpu_operands[2][i], ds_operands}),
            .rnd_mode_i     (fpnew_pkg::roundmode_e'(fpu_rnd)),
            .op_i           (fpu_op),
            .op_mod_i       (fpu_op_mod),
            .src_fmt_i      (fpu_src_fmt),
            .dst_fmt_i      (fpu_dst_fmt),
            .int_fmt_i      (fpu_int_fmt),
            .vectorial_op_i (1'b0),
            .simd_mask_i    (1'b1),
            .tag_i          ({tag_in, mask_in}),
            .in_valid_i     (ds_lane_valid_in[i]),
            .in_ready_o     (ds_lane_ready_in[i]),
            .flush_i        (1'b0),
            .result_o       (lane_ds_result),
            .status_o       (lane_ds_status),
            .tag_o          (lane_ds_tag),
            .out_valid_o    (ds_lane_valid_out[i]),
            .out_ready_i    (ds_lane_ready_out[i]),
            `UNUSED_PIN (busy_o)
        );

        // The div/sqrt unit also raises DZ on 0/0, which is only invalid.
        fflags_t lane_ds_fflags;
        always @(*) begin
            lane_ds_fflags    = lane_ds_status;
            lane_ds_fflags.DZ = lane_ds_status.DZ && ~lane_ds_status.NV;
        end

        assign ds_lane_data_out[i] = {lane_ds_tag, lane_ds_fflags, lane_ds_result};

        if (i != 0) begin : g_unused_tags
            `UNUSED_VAR (ls_tag[i])
        end
    end

    `RUNTIME_ASSERT(((ls_valid_out == '0) || (ls_valid_out == '1)) && ((ls_ready_in == '0) || (ls_ready_in == '1)),
        ("%t: fpnew lanes out of lockstep: valid_out=%b, ready_in=%b", $time, ls_valid_out, ls_ready_in))

    VX_stream_join #(
        .NUM_INPUTS (NUM_LANES),
        .DATAW      (DS_DATAW)
    ) ds_join (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (ds_lane_valid_out),
        .data_in   (ds_lane_data_out),
        .ready_in  (ds_lane_ready_out),
        .valid_out (ds_valid_out),
        .data_out  (ds_data_out),
        .ready_out (ds_ready_out)
    );

    assign ds_valid_in = valid_in && is_divsqrt;
    assign ready_in = is_divsqrt ? ds_ready_in : ls_ready_in[0];

    // Lockstep response
    wire [TAG_WIDTH-1:0] ls_tag_out;
    wire ls_has_fflags;
    wire [NUM_LANES-1:0] ls_mask;
    fflags_t ls_fflags;
    assign {ls_tag_out, ls_has_fflags, ls_mask} = ls_tag[0];

    `FPU_MERGE_FFLAGS(ls_fflags, ls_status, ls_mask, NUM_LANES);

    // Div/sqrt response
    wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] ds_result;
    fflags_t [NUM_LANES-1:0] ds_status;
    wire [NUM_LANES-1:0][DS_TAGW-1:0] ds_tag;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_ds_unpack
        assign {ds_tag[i], ds_status[i], ds_result[i]} = ds_data_out[i];
        if (i != 0) begin : g_unused_tags
            `UNUSED_VAR (ds_tag[i])
        end
    end

    wire [TAG_WIDTH-1:0] ds_tag_out;
    wire [NUM_LANES-1:0] ds_mask;
    fflags_t ds_fflags;
    assign {ds_tag_out, ds_mask} = ds_tag[0];

    `FPU_MERGE_FFLAGS(ds_fflags, ds_status, ds_mask, NUM_LANES);

    VX_stream_arb #(
        .NUM_INPUTS (2),
        .DATAW      (RSP_DATAW),
        .ARBITER    ("P"),
        .OUT_BUF    (OUT_BUF)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  ({ls_valid_out[0], ds_valid_out}),
        .ready_in  ({ls_ready_out, ds_ready_out}),
        .data_in   ({{ls_result, ls_has_fflags, ls_fflags, ls_tag_out},
                     {ds_result, 1'b1, ds_fflags, ds_tag_out}}),
        .data_out  ({result, has_fflags, fflags, tag_out}),
        .valid_out (valid_out),
        .ready_out (ready_out),
        `UNUSED_PIN (sel_out)
    );

endmodule

`endif
