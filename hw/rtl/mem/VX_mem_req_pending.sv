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

// Tracks the requests held between the input and output handshakes of a
// section that neither creates nor drops requests (arbiters, buffers).
module VX_mem_req_pending #(
    parameter NUM_INPUTS  = 1,
    parameter NUM_OUTPUTS = 1,
    parameter SIZE        = 8 * NUM_OUTPUTS + 2 * NUM_INPUTS // bound on the requests the section can hold
) (
    input wire                   clk,
    input wire                   reset,
    input wire [NUM_INPUTS-1:0]  in_fire,
    input wire [NUM_OUTPUTS-1:0] out_fire,
    output wire                  empty
);
    localparam INCRW = `CLOG2(NUM_INPUTS+1);
    localparam DECRW = `CLOG2(NUM_OUTPUTS+1);
    localparam SIZEW = `CLOG2(SIZE+1);

    // The handshakes usually end long ready chains, so they are registered
    // before counting, and over many inputs the count of them is registered
    // too: the counter then lags two cycles. empty is conservative: it is the
    // exact empty of the section when no input fired the previous cycle and
    // false otherwise, so it never reports empty while a request is held and
    // reports it at most one cycle after the section drained. A cycle in which
    // an input fires and the section drains the same cycle is the only one it
    // misses.
    reg [NUM_INPUTS-1:0]  in_fire_r;
    reg [NUM_OUTPUTS-1:0] out_fire_r;
    always @(posedge clk) begin
        if (reset) begin
            in_fire_r  <= '0;
            out_fire_r <= '0;
        end else begin
            in_fire_r  <= in_fire;
            out_fire_r <= out_fire;
        end
    end

    wire [INCRW-1:0] in_cnt_w;
    wire [DECRW-1:0] out_cnt_w;
    `POP_COUNT(in_cnt_w, in_fire_r);
    `POP_COUNT(out_cnt_w, out_fire_r);

    reg [INCRW-1:0] in_cnt_r;
    reg [DECRW-1:0] out_cnt_r;
    always @(posedge clk) begin
        if (reset) begin
            in_cnt_r  <= '0;
            out_cnt_r <= '0;
        end else begin
            in_cnt_r  <= in_cnt_w;
            out_cnt_r <= out_cnt_w;
        end
    end

    // requests through the handshakes of two cycles ago
    wire [SIZEW-1:0] size_lag;
    VX_pending_size #(
        .SIZE  (SIZE),
        .INCRW (INCRW),
        .DECRW (DECRW)
    ) pending (
        .clk       (clk),
        .reset     (reset),
        .incr      (in_cnt_r),
        .decr      (out_cnt_r),
        `UNUSED_PIN (empty),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_full),
        .size      (size_lag)
    );

    // held after last cycle's handshakes, given no input fired last cycle
    wire [SIZEW:0] held_in  = (SIZEW+1)'(size_lag) + (SIZEW+1)'(in_cnt_r);
    wire [SIZEW:0] held_out = (SIZEW+1)'(out_cnt_r) + (SIZEW+1)'(out_cnt_w);
    assign empty = ~(| in_fire_r) && (held_in == held_out);

endmodule
