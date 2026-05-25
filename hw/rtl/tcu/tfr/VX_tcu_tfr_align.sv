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

module VX_tcu_tfr_align import VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter N     = 5,
    parameter WI    = 25,
    parameter WO    = WI + 2
) (
    input  wire                 clk,
    input  wire                 valid_in,
    input  wire [31:0]          req_id,

    input  wire [N-1:0][TCU_EXP_BITS-1:0] exponents,
    input  wire [N-1:0]         sel_exp,
    input  wire [N-2:0][N-2:0][TCU_EXP_BITS:0] diff_mat,

    input  wire [N-1:0][WI-1:0] sigs_in,
    input  wire                 is_int,
    output logic [TCU_EXP_BITS-1:0] max_exp,
    output wire [N-1:0][WO-1:0] sigs_out,
    output wire [N-1:0]         sticky_bits
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({clk, valid_in, req_id})

    localparam MAX_PRE_SHIFT = WI - 23;
    localparam SHIFT_MAG_W   = (WI - 1) + MAX_PRE_SHIFT;

    wire [TCU_EXP_BITS-1:0] or_red[N:0] /* verilator split_var */;
    wire [N-1:0][7:0] shift_amts;

    assign or_red[0] = {TCU_EXP_BITS{1'b0}};
    for (genvar i = 0; i < N; i++) begin : g_or_red
        assign or_red[i+1] = or_red[i] | (sel_exp[i] ? exponents[i] : {TCU_EXP_BITS{1'b0}});
    end
    assign max_exp = or_red[N];

    for (genvar i = 0; i < N; i++) begin : g_shift_amts
        wire [TCU_EXP_BITS-1:0] sh_or [N:0] /* verilator split_var */;

        assign sh_or[0] = {TCU_EXP_BITS{1'b0}};
        for (genvar k = 0; k < N; k++) begin : g_sh_mux
            if (k == i) begin : g_self
                assign sh_or[k+1] = sh_or[k];
            end else if (k < i) begin : g_direct
                wire [TCU_EXP_BITS-1:0] diff_lane = diff_mat[k][i-1][TCU_EXP_BITS-1:0];
                assign sh_or[k+1] = sh_or[k] | (sel_exp[k] ? diff_lane : {TCU_EXP_BITS{1'b0}});
            end else begin : g_invert
                wire [TCU_EXP_BITS-1:0] diff_lane = diff_mat[i][k-1][TCU_EXP_BITS-1:0];
                assign sh_or[k+1] = sh_or[k] | (sel_exp[k] ? ~diff_lane : {TCU_EXP_BITS{1'b0}});
            end
        end

        wire needs_inc;
        if (i == N-1) begin : g_no_inc
            assign needs_inc = 1'b0;
        end else begin : g_calc_inc
            wire [N-2-i:0] inc_sel;
            for (genvar k = i+1; k < N; k++) begin : g_inc_sel
                assign inc_sel[k-i-1] = sel_exp[k];
            end
            assign needs_inc = |inc_sel;
        end

        if (TCU_EXP_BITS > 8) begin : g_sat
            wire [7:0] shift_lo = sh_or[N][7:0] + 8'(needs_inc);
            wire shift_hi = (|sh_or[N][TCU_EXP_BITS-1:8]) || (needs_inc && (&sh_or[N][7:0]));
            assign shift_amts[i] = shift_hi ? 8'hFF : shift_lo;
        end else begin : g_no_sat
            wire [TCU_EXP_BITS-1:0] shift_full = sh_or[N] + TCU_EXP_BITS'(needs_inc);
            assign shift_amts[i] = 8'(shift_full);
        end
    end

    for (genvar i = 0; i < N; ++i) begin : g_align_lanes
        wire [7:0] shift_amt = shift_amts[i];

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
        wire is_overshift = (shift_amt >= 8'(SHIFT_MAG_W));
        wire [SHIFT_MAG_W-1:0] shift_res_full = mag_shifted >> shift_amt;
        wire [WO-2:0] adj_mag = is_overshift ? '0 : shift_res_full[WO-2:0];

        // 4. Sticky Calculation
        wire [SHIFT_MAG_W-1:0] sticky_check_shift = mag_shifted << (8'(SHIFT_MAG_W) - shift_amt);
        assign sticky_bits[i] = is_overshift ? (|mag_shifted) : (|sticky_check_shift);

        // 5. Output select
        assign sigs_out[i] = is_int ? WO'($signed(sigs_in[i])) : {in_sign, adj_mag};
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
