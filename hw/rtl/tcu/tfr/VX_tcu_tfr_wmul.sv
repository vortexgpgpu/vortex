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

// Generic unsigned mantissa multiplier for the TFR FEDP.
//   p[i] = a[i] * b[SHARED_B ? 0 : i]   (i = 0 .. LANES-1)
//
// USE_DSP=0: LUT fabric — a Wallace tree per lane when square (N==M), else an
//            inferred multiply. Area-optimal / ASIC-portable.
// USE_DSP=1: pack all LANES products into ONE DSP48 by placing the per-lane
//            operands in non-overlapping fields of the 27x18 product:
//   LANES==1            : plain DSP multiply.
//   LANES==2, SHARED_B  : pa = a1<<P | a0, prod = pa*b0  -> no cross-terms
//                         (single b), stride K=P.
//   LANES==2, !SHARED_B : pa = a1<<K | a0, pb = b1<<K | b0, K=P+1; the two
//                         diagonal products land clear of the middle cross-term
//                         field, which is discarded.
// All paths are bit-identical; USE_DSP only changes the target primitive.
// (Independent packing is limited to LANES<=2: at LANES>2 the off-diagonal
//  partial products collide with the wanted diagonal terms.)
module VX_tcu_tfr_wmul #(
    parameter N        = 4,       // operand A width
    parameter M        = N,       // operand B width
    parameter LANES    = 1,       // products packed into one DSP (1 or 2)
    parameter SHARED_B = 0,       // 1: every lane multiplies the single b[0]
    parameter P        = N + M,   // per-product width
    parameter USE_DSP  = 0
) (
    input  wire [LANES-1:0][N-1:0] a,
    input  wire [LANES-1:0][M-1:0] b,
    output wire [LANES-1:0][P-1:0] p
);
    `STATIC_ASSERT (LANES == 1 || LANES == 2, ("VX_tcu_tfr_wmul: LANES must be 1 or 2"))

    if (USE_DSP == 0) begin : g_lut
        for (genvar i = 0; i < LANES; ++i) begin : g_mul
            localparam BI = (SHARED_B != 0) ? 0 : i;
            if (N == M) begin : g_wal
                VX_wallace_mul #(.N(N), .P(P), .CPA_KS(!`FORCE_BUILTIN_ADDER(N+M))) u_mul (
                    .a(a[i]), .b(b[BI]), .p(p[i]));
            end else begin : g_inf
                assign p[i] = a[i] * b[BI];
            end
        end
        if (SHARED_B != 0 && LANES > 1) begin : g_unused
            `UNUSED_VAR (b[1])
        end
    end else if (LANES == 1) begin : g_dsp1
        (* use_dsp = "yes" *) wire [P-1:0] prod = a[0] * b[0];
        assign p[0] = prod;
    end else if (SHARED_B != 0) begin : g_dsp_shared
        localparam K  = P;                  // single b: no cross-term to clear
        localparam AW = K + N;
        wire [AW-1:0]      pa   = {a[1], {(K-N){1'b0}}, a[0]};
        (* use_dsp = "yes" *) wire [AW+M-1:0] prod = pa * b[0];
        assign p[0] = prod[P-1:0];
        assign p[1] = prod[K +: P];
        `UNUSED_VAR (b[1])
        `UNUSED_VAR (prod)
    end else begin : g_dsp_indep
        localparam K  = P + 1;              // clear product (P) + cross carry (P+1)
        localparam AW = K + N;
        localparam BW = K + M;
        wire [AW-1:0]      pa   = {a[1], {(K-N){1'b0}}, a[0]};
        wire [BW-1:0]      pb   = {b[1], {(K-M){1'b0}}, b[0]};
        (* use_dsp = "yes" *) wire [AW+BW-1:0] prod = pa * pb;
        assign p[0] = prod[P-1:0];
        assign p[1] = prod[2*K +: P];
        `UNUSED_VAR (prod)
    end

endmodule
