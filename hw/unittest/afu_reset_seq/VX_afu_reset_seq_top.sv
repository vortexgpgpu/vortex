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

// Harness for the AFU soft-reset path: the real VX_afu_reset_seq and the real
// VX_afu_axi_drain, plus the same reset-delay shift register VX_afu_wrap uses,
// so the RELEASE handshake is exercised exactly as it is in the AFU.
//
// TIMEOUT_CYCLES is small here so the refusal path is reachable in a test.

module VX_afu_reset_seq_top #(
    parameter TIMEOUT_CYCLES = 64,
    parameter RESET_DELAY    = 4
) (
    input  wire clk,
    input  wire reset,

    input  wire ap_reset_req,

    // AXI handshakes, driven directly by the testbench.
    input  wire aw_fire,
    input  wire w_fire,
    input  wire w_fire_last,
    input  wire b_fire,
    input  wire ar_fire,
    input  wire r_fire_last,

    // A real request gate driven by the sequencer's stop_req, with the
    // testbench playing the AFU-internal master (gate_in_*) and the shell
    // (gate_out_*). This is how VX_afu_wrap wires every AW/AR channel.
    input  wire gate_in_valid,
    input  wire gate_out_ready,
    output wire gate_in_ready,
    output wire gate_out_valid,

    // A real write gate on the same stop_req, the testbench again playing
    // both the AFU-internal master (wg_in_*) and the shell (wg_out_*).
    input  wire wg_in_awvalid,
    input  wire wg_out_awready,
    output wire wg_in_awready,
    output wire wg_out_awvalid,
    input  wire wg_in_wvalid,
    input  wire wg_in_wlast,
    input  wire wg_out_wready,
    output wire wg_in_wready,
    output wire wg_out_wvalid,

    output wire stop_req,
    output wire rst_assert,
    output wire busy,
    output wire timeout_error,
    output wire masters_idle,
    output wire vx_reset
);
    VX_afu_req_gate gate (
        .clk       (clk),
        .reset     (reset),
        .stop_req  (stop_req),
        .in_valid  (gate_in_valid),
        .in_ready  (gate_in_ready),
        .out_valid (gate_out_valid),
        .out_ready (gate_out_ready)
    );

    VX_afu_wr_gate wr_gate (
        .clk         (clk),
        .reset       (reset),
        .stop_req    (stop_req),
        .in_awvalid  (wg_in_awvalid),
        .in_awready  (wg_in_awready),
        .out_awvalid (wg_out_awvalid),
        .out_awready (wg_out_awready),
        .in_wvalid   (wg_in_wvalid),
        .in_wlast    (wg_in_wlast),
        .in_wready   (wg_in_wready),
        .out_wvalid  (wg_out_wvalid),
        .out_wready  (wg_out_wready)
    );

    VX_afu_axi_drain drain (
        .clk          (clk),
        .reset        (reset),
        .aw_fire      (aw_fire),
        .w_fire       (w_fire),
        .w_fire_last  (w_fire_last),
        .b_fire       (b_fire),
        .ar_fire      (ar_fire),
        .r_fire_last  (r_fire_last),
        .idle         (masters_idle),
        // debug counters unused by this test
        `UNUSED_PIN (dbg_aw_count),
        `UNUSED_PIN (dbg_w_count),
        `UNUSED_PIN (dbg_b_count),
        `UNUSED_PIN (dbg_ar_count),
        `UNUSED_PIN (dbg_r_count)
    );

    // Mirrors VX_afu_wrap: the platform reset reloads the shift register
    // directly, a soft reset only through the sequencer.
    reg [RESET_DELAY-1:0] vx_reset_shift_r;
    always @(posedge clk) begin
        if (reset || rst_assert) begin
            vx_reset_shift_r <= {RESET_DELAY{1'b1}};
        end else begin
            vx_reset_shift_r <= {vx_reset_shift_r[RESET_DELAY-2:0], 1'b0};
        end
    end
    assign vx_reset = vx_reset_shift_r[RESET_DELAY-1];

    VX_afu_reset_seq #(
        .TIMEOUT_CYCLES (TIMEOUT_CYCLES)
    ) seq (
        .clk             (clk),
        .reset           (reset),
        .ap_reset_req    (ap_reset_req),
        .masters_idle    (masters_idle),
        .vx_reset_active (vx_reset),
        .stop_req        (stop_req),
        .rst_assert      (rst_assert),
        .busy            (busy),
        .timeout_error   (timeout_error)
    );

endmodule
