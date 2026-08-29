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

// VX_clockgate — Inference-based integrated clock gating (ICG)
//
// Portable across ASIC and FPGA tool flows:
//   • Synopsys DC  : infers a library ICG cell (e.g. CKLNQD1)
//   • Vivado       : pushes enable into downstream flop CE pins
//   • Quartus      : same CE-push-down optimisation
//   • Yosys        : keeps behavioural model for simulation
//
// The latch is transparent while clk_in is low, so en_latched can
// only transition during the low phase. This guarantees clk_out is
// glitch-free on the rising edge of clk_in.

`TRACING_OFF

module VX_clockgate (
    input  wire clk_in,   // free-running source clock
    input  wire en,        // active-high gate enable
    output wire clk_out    // gated clock output
);

    // Negative-level latch: capture enable while clock is low
    // to prevent glitches on the gated clock.

    /* verilator lint_off LATCH */
    reg en_latched;
    always @(*) begin
        if (~clk_in)
            en_latched = en;
    end
    /* verilator lint_on LATCH */

    // Gated clock: rises only when the latched enable is asserted.
    assign clk_out = clk_in & en_latched;

endmodule

`TRACING_ON
