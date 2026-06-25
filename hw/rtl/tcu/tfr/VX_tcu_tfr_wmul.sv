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

// Unsigned NxN -> 2N mantissa multiplier primitive for the TFR FEDP.
//   USE_DSP=0: LUT-fabric Wallace tree (ASIC / DSP-off), area-optimal.
//   USE_DSP=1: inferred multiply with a use_dsp hint, mapped to an FPGA DSP48.
// The two paths are bit-identical; USE_DSP only changes the target primitive.
module VX_tcu_tfr_wmul #(
    parameter N       = 4,
    parameter P       = 2 * N,
    parameter USE_DSP = 0
) (
    input  wire [N-1:0] a,
    input  wire [N-1:0] b,
    output wire [P-1:0] p
);
    if (USE_DSP != 0) begin : g_dsp
        (* use_dsp = "yes" *) wire [P-1:0] prod = P'(a) * P'(b);
        assign p = prod;
    end else begin : g_wallace
        VX_wallace_mul #(
            .N (N),
            .P (P),
            .CPA_KS (!`FORCE_BUILTIN_ADDER(N*2))
        ) u_mul (
            .a (a),
            .b (b),
            .p (p)
        );
    end

endmodule
