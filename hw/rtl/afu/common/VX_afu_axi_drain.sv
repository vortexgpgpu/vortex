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

`include "vortex_afu.vh"

// ============================================================================
// VX_afu_axi_drain — outstanding-transaction tracker for one AXI master port.
//
// Reports `idle` when the port owes the interconnect nothing: every accepted
// AW has had its write burst sent and its BRESP returned, and every accepted
// AR has had its final R beat returned. That is the only safe moment to
// assert a reset across the master, because resetting it with a transaction
// in flight leaves the interconnect waiting for beats that never arrive.
//
// Implemented as three pairs of free-running wrapping counters compared for
// equality rather than up/down counters. Equality is correct in both
// directions, so it stays right when AXI4 allows write data to be presented
// before its address (which would drive an up/down counter negative).
//
// COUNT_WIDTH must exceed the port's maximum outstanding transactions; the
// comparison is exact only while fewer than 2**COUNT_WIDTH are in flight.
// ============================================================================

module VX_afu_axi_drain #(
    parameter COUNT_WIDTH = 10
) (
    input  wire clk,
    input  wire reset,

    // Handshakes observed on the shell side of the request gate.
    input  wire aw_fire,      // awvalid && awready
    input  wire w_fire_last,  // wvalid && wready && wlast
    input  wire b_fire,       // bvalid && bready
    input  wire ar_fire,      // arvalid && arready
    input  wire r_fire_last,  // rvalid && rready && rlast

    output wire idle,

    // Raw counters, for the debug block in VX_afu_ctrl. Comparing these from
    // the host is how "did the write ever leave the AFU" gets answered
    // without a JTAG ILA.
    output wire [COUNT_WIDTH-1:0] dbg_aw_count,
    output wire [COUNT_WIDTH-1:0] dbg_w_count,
    output wire [COUNT_WIDTH-1:0] dbg_b_count,
    output wire [COUNT_WIDTH-1:0] dbg_ar_count,
    output wire [COUNT_WIDTH-1:0] dbg_r_count
);
    reg [COUNT_WIDTH-1:0] aw_count;
    reg [COUNT_WIDTH-1:0] w_count;
    reg [COUNT_WIDTH-1:0] b_count;
    reg [COUNT_WIDTH-1:0] ar_count;
    reg [COUNT_WIDTH-1:0] r_count;

    always @(posedge clk) begin
        if (reset) begin
            aw_count <= '0;
            w_count  <= '0;
            b_count  <= '0;
            ar_count <= '0;
            r_count  <= '0;
        end else begin
            if (aw_fire) begin
                aw_count <= aw_count + 1'b1;
            end
            if (w_fire_last) begin
                w_count <= w_count + 1'b1;
            end
            if (b_fire) begin
                b_count <= b_count + 1'b1;
            end
            if (ar_fire) begin
                ar_count <= ar_count + 1'b1;
            end
            if (r_fire_last) begin
                r_count <= r_count + 1'b1;
            end
        end
    end

    // Every burst launched has been fully written, answered, and read back.
    assign dbg_aw_count = aw_count;
    assign dbg_w_count  = w_count;
    assign dbg_b_count  = b_count;
    assign dbg_ar_count = ar_count;
    assign dbg_r_count  = r_count;

    assign idle = (aw_count == w_count)
               && (aw_count == b_count)
               && (ar_count == r_count);

endmodule
