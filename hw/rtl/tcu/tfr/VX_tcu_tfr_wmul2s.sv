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

// Two unsigned products that SHARE one operand, packed into ONE DSP48:
//   p0 = a0*b,  p1 = a1*b
// Because b is a single value there are no cross-terms, so the two products
// land cleanly in non-overlapping fields at stride K=P:
//   pa = a1<<K | a0,  pa*b = a0*b + (a1*b)<<K
// This is the Xilinx weight-reuse packing; here the shared operand exists
// naturally (e.g. nvfp4's per-lane scale-factor mantissa multiplied into every
// term). Bit-identical to two separate multiplies (USE_DSP=0).
module VX_tcu_tfr_wmul2s #(
    parameter NA      = 4,
    parameter NB      = 8,
    parameter P       = NA + NB,
    parameter USE_DSP = 0
) (
    input  wire [NA-1:0] a0,
    input  wire [NA-1:0] a1,
    input  wire [NB-1:0] b,
    output wire [P-1:0]  p0,
    output wire [P-1:0]  p1
);
    if (USE_DSP != 0) begin : g_dsp
        localparam K  = P;                  // no cross-term: stride = product width
        localparam AW = K + NA;
        wire [AW-1:0]      pa   = {a1, {(K-NA){1'b0}}, a0};
        (* use_dsp = "yes" *) wire [AW+NB-1:0] prod = pa * b;
        assign p0 = prod[P-1:0];
        assign p1 = prod[K +: P];
        `UNUSED_VAR (prod)
    end else begin : g_lut
        assign p0 = a0 * b;
        assign p1 = a1 * b;
    end

endmodule
