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

// Two independent unsigned NxN -> 2N products packed into ONE DSP48 (USE_DSP=1).
//
// Operands are packed at a stride K = 2N+1 so the two wanted ("diagonal")
// products land in non-overlapping fields of the single 27x18 product, while
// the cross-terms collect in the middle field (K..2K-1) and are discarded:
//
//   pa = a1<<K | a0,  pb = b1<<K | b0
//   pa*pb = a0*b0            (bits [0 : 2N-1])
//         + (a0*b1+a1*b0)<<K (bits [K : K+2N]  -- discarded)
//         + a1*b1<<2K        (bits [2K: 2K+2N-1])
//
// K=2N+1 guarantees a0*b0 (2N bits) clears before the cross field, and the
// cross field (2N+1 bits) clears before a1*b1 — so both extractions are exact.
// Only valid for unsigned operands small enough that 2*(K+N) fits a DSP slice
// (N<=5 here: fp8/fp4 mantissas). Bit-identical to two Wallace muls (USE_DSP=0).
module VX_tcu_tfr_wmul2 #(
    parameter N       = 4,
    parameter P       = 2 * N,
    parameter USE_DSP = 0
) (
    input  wire [N-1:0] a0,
    input  wire [N-1:0] b0,
    input  wire [N-1:0] a1,
    input  wire [N-1:0] b1,
    output wire [P-1:0] p0,
    output wire [P-1:0] p1
);
    if (USE_DSP != 0) begin : g_dsp
        localparam K  = 2 * N + 1;          // field stride
        localparam AW = K + N;              // packed operand width
        wire [AW-1:0] pa = {a1, {(K-N){1'b0}}, a0};
        wire [AW-1:0] pb = {b1, {(K-N){1'b0}}, b0};
        (* use_dsp = "yes" *) wire [2*AW-1:0] prod = pa * pb;
        assign p0 = prod[P-1:0];
        assign p1 = prod[2*K +: P];
        `UNUSED_VAR (prod)
    end else begin : g_wallace
        VX_wallace_mul #(.N(N), .P(P), .CPA_KS(!`FORCE_BUILTIN_ADDER(N*2))) m0 (.a(a0), .b(b0), .p(p0));
        VX_wallace_mul #(.N(N), .P(P), .CPA_KS(!`FORCE_BUILTIN_ADDER(N*2))) m1 (.a(a1), .b(b1), .p(p1));
    end

endmodule
