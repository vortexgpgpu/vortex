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

`include "VX_define.vh"

// Aligns the significands and emits ready-to-add accumulator operands:
// lane masking, the fp ones-complement negate and the int sign-extension all
// fold into the output select here, so the accumulate stage is a bare
// compressor tree. fp_negs flags the negated lanes (their +1 completion is a
// single popcount operand downstream).
module VX_tcu_tfr_align import VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter N     = 5,
    parameter WI    = 25,
    parameter WO    = WI + 5
) (
    input  wire                 clk,
    input  wire                 valid_in,
    input  wire [31:0]          req_id,

    input  wire [N-1:0][TCU_EXP_BITS-1:0] exponents,
    input  wire [N-1:0]         sel_exp,
    input  wire [N-2:0]         lane_mask,

    input  wire [N-1:0][WI-1:0] sigs_in,
    input  wire                 is_int,
    output wire [TCU_EXP_BITS-1:0] max_exp,
    output wire [N-1:0][WO-1:0] sigs_out,
    output wire [N-1:0]         sticky_bits,
    output wire [N-1:0]         fp_negs
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({clk, valid_in, req_id})

    localparam MAX_PRE_SHIFT = WI - 23;
    localparam SHIFT_MAG_W   = (WI - 1) + MAX_PRE_SHIFT;
    localparam SHIFT_W       = `CLOG2(SHIFT_MAG_W + 1);
    localparam TZ_W          = `LOG2UP(SHIFT_MAG_W);

    `STATIC_ASSERT (TCU_EXP_BITS > SHIFT_W, ("invalid shift width: %0d", SHIFT_W))

    VX_onehot_mux #(
        .DATAW (TCU_EXP_BITS),
        .N     (N)
    ) max_exp_mux (
        .data_in  (exponents),
        .sel_in   (sel_exp),
        .data_out (max_exp)
    );

    // Per-lane right-shift distance from the maximum exponent (max_exp is the
    // largest input, so the difference is non-negative).
    wire [N-1:0][TCU_EXP_BITS-1:0] shift_amts;
    for (genvar i = 0; i < N; ++i) begin : g_shift_amts
        assign shift_amts[i] = max_exp - exponents[i];
    end

    for (genvar i = 0; i < N; ++i) begin : g_align_lanes
        wire shift_hi = (| shift_amts[i][TCU_EXP_BITS-1:SHIFT_W]);
        wire [SHIFT_W-1:0] shift_lo = shift_amts[i][SHIFT_W-1:0];

        wire lane_en;
        if (i == N-1) begin : g_c_en
            assign lane_en = 1'b1; // C-term is never masked
        end else begin : g_lane_en
            assign lane_en = lane_mask[i];
        end

        // 1. Unpack Sign and Magnitude
        wire in_sign = sigs_in[i][WI-1];
        wire [WI-2:0] in_mag = sigs_in[i][WI-2:0];

        // 2. Pre-Shift Magnitude
        wire [SHIFT_MAG_W-1:0] mag_shifted;
        if (i == N-1) begin : g_c_term
            assign mag_shifted = { {(MAX_PRE_SHIFT - (WI - 24)){1'b0}}, in_mag, {(WI - 24){1'b0}} };
        end else begin : g_prod_term
            assign mag_shifted = { in_mag, {(WI - 23){1'b0}} };
        end

        // 3. Shift adjustment
        wire is_overshift = shift_hi || (shift_lo >= SHIFT_W'(SHIFT_MAG_W));
        wire [SHIFT_MAG_W-1:0] shift_res_full = mag_shifted >> shift_lo;
        wire [WI:0] adj_mag = is_overshift ? '0 : shift_res_full[WI:0];

        // 4. Sticky: a set bit shifts out iff the shift exceeds the trailing
        // zero count, which comes straight from the registered magnitude and
        // so resolves in parallel with the exponent difference.
        wire [TZ_W-1:0] mag_tz;
        wire            mag_nz;
        VX_lzc #(
            .N       (SHIFT_MAG_W),
            .REVERSE (1)
        ) mag_tzc (
            .data_in   (mag_shifted),
            .data_out  (mag_tz),
            .valid_out (mag_nz)
        );
        assign sticky_bits[i] = lane_en && mag_nz && (shift_hi || (shift_lo > SHIFT_W'(mag_tz)));

        // 5. Operand formation: masked, negated (fp) or sign-extended (int).
        wire fp_neg = in_sign & lane_en & ~is_int;
        wire [WO-1:0] fp_mag  = WO'(adj_mag & {(WI+1){lane_en}});
        wire [WO-1:0] fp_form = fp_neg ? ~fp_mag : fp_mag;
        wire [WO-1:0] int_form = WO'($signed(sigs_in[i])) & {WO{lane_en}};
        assign sigs_out[i] = is_int ? int_form : fp_form;
        assign fp_negs[i]  = fp_neg;
    end

`ifdef DBG_TRACE_TCU
    always_ff @(posedge clk) begin
        if (valid_in) begin
            `TRACE(4, ("%t: %s FEDP-ALIGN(%0d): is_int=%0d", $time, INSTANCE_ID, req_id, is_int));
            `TRACE(4, (", max_exp=0x%0h, shift_amts=", max_exp));
            `TRACE_ARRAY1D(4, "0x%0h", shift_amts, N)
            `TRACE(4, (", sigs_in="));
            `TRACE_ARRAY1D(4, "0x%0h", sigs_in, N)
            `TRACE(4, (", sigs_out="));
            `TRACE_ARRAY1D(4, "0x%0h", sigs_out, N)
            `TRACE(4, (", sticky="));
            `TRACE_ARRAY1D(4, "%0d", sticky_bits, N)
            `TRACE(4, ("\n"));
        end
    end
`endif

endmodule
