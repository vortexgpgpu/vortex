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

// Toy device under test for the FireSim platform bring-up.
//
// Deliberately trivial: its purpose is to exercise the FireSim flow end to end
// (Chisel BlackBox -> Golden Gate -> bitstream -> card) with a design small
// enough that any failure is attributable to the flow rather than the design.
//
// The port shape mirrors the contract Golden Gate imposes on a blackboxed
// target, which Vortex also satisfies: a single clock input, a single reset,
// and no internally generated or gated clocks.

module VX_adder4 (
    input  wire       clk,
    input  wire       reset,
    input  wire [3:0] a,
    input  wire [3:0] b,
    output reg  [4:0] sum
);
    always @(posedge clk) begin
        if (reset) begin
            sum <= 5'd0;
        end else begin
            sum <= a + b;
        end
    end
endmodule
