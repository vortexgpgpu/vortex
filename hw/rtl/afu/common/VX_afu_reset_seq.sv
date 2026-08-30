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
// VX_afu_reset_seq — quiesce-before-reset sequencer for the AFU soft reset.
//
//   IDLE    ap_reset request               → QUIESCE
//   QUIESCE stop_req asserted: no new AW/AR leaves the AFU. Bursts already
//           accepted finish and their responses return.
//             all masters idle             → ASSERT
//             timeout                      → ERROR   (reset is NOT asserted)
//   ASSERT  one cycle: reload the Vortex reset-delay shift register
//                                          → RELEASE
//   RELEASE hold stop_req until the delayed reset has drained
//                                          → IDLE
//   ERROR   raise a sticky status bit and give up
//                                          → IDLE
//
// Refusing to reset a master that will not drain is deliberate. Resetting it
// anyway is what produces a protocol violation on the interconnect, and a
// device that reports "I could not reset" is strictly more useful than one
// that silently corrupts the bus. `busy` drives ap_idle, so software's
// existing poll observes the whole sequence.
// ============================================================================

module VX_afu_reset_seq #(
    // Cycles allowed for the masters to drain. At 200 MHz the default is
    // ~5 ms, orders of magnitude beyond any legitimate memory transaction and
    // still far below the host's PCIe completion timeout.
    parameter TIMEOUT_CYCLES = 1048576
) (
    input  wire clk,
    input  wire reset,

    input  wire ap_reset_req,     // one-cycle request from VX_afu_ctrl
    input  wire masters_idle,     // AND of every VX_afu_axi_drain.idle
    input  wire vx_reset_active,  // the delayed reset is still asserted

    output wire stop_req,         // gate new AW/AR out of the AFU
    output wire rst_assert,       // one-cycle reload of the reset-delay shift register
    output wire busy,             // sequence in flight (drives ~ap_idle)
    output wire timeout_error     // sticky: last request could not be honoured
);
    localparam TIMEOUT_WIDTH = $clog2(TIMEOUT_CYCLES + 1);

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_QUIESCE,
        ST_ASSERT,
        ST_RELEASE,
        ST_ERROR
    } state_e;

    state_e                     state;
    reg [TIMEOUT_WIDTH-1:0]     timer;
    reg                         error_r;

    always @(posedge clk) begin
        if (reset) begin
            state   <= ST_IDLE;
            timer   <= '0;
            error_r <= 1'b0;
        end else begin
            case (state)
                ST_IDLE: begin
                    if (ap_reset_req) begin
                        // A new request clears the previous verdict.
                        error_r <= 1'b0;
                        timer   <= '0;
                        state   <= ST_QUIESCE;
                    end
                end
                ST_QUIESCE: begin
                    if (masters_idle) begin
                        state <= ST_ASSERT;
                    end else if (timer == TIMEOUT_WIDTH'(TIMEOUT_CYCLES)) begin
                        state <= ST_ERROR;
                    end else begin
                        timer <= timer + 1'b1;
                    end
                end
                ST_ASSERT: begin
                    state <= ST_RELEASE;
                end
                ST_RELEASE: begin
                    // Hold new requests off until the shift register has
                    // finished driving the subsystem reset.
                    if (!vx_reset_active) begin
                        state <= ST_IDLE;
                    end
                end
                ST_ERROR: begin
                    error_r <= 1'b1;
                    state   <= ST_IDLE;
                end
                default: begin
                    state <= ST_IDLE;
                end
            endcase
        end
    end

    assign stop_req      = (state == ST_QUIESCE)
                        || (state == ST_ASSERT)
                        || (state == ST_RELEASE);
    assign rst_assert    = (state == ST_ASSERT);
    assign busy          = (state != ST_IDLE);
    assign timeout_error = error_r;

endmodule
