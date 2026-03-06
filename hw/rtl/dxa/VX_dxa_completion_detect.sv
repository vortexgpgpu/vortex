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

module VX_dxa_completion_detect import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter NUM_BANKS = 1,
    parameter TAG_WIDTH = DXA_BANK_WR_TAG_WIDTH
) (
    input  wire                                clk,
    input  wire                                reset,

    // DXA bank writes (from dedicated bank_wr port, all are DXA by definition)
    input  wire [NUM_BANKS-1:0]                bank_wr_fire,
    input  wire [TAG_WIDTH-1:0]                bank_wr_tag,    // shared tag: {last_pkt, bar_addr}

    output wire                                done_valid,
    input  wire                                done_ready,
    output wire [BAR_ADDR_W-1:0]               done_bar_addr
);
`ifdef EXT_DXA_ENABLE
    // Tag format: {last_pkt, bar_addr}
    // With shared tag across banks, at most 1 done event per DXA write cycle.
    wire is_last = bank_wr_tag[TAG_WIDTH-1];
    wire any_dxa_wr = |bank_wr_fire;

    // Valid/ready handshake: hold done_valid high until downstream accepts.
    // Use a single pending slot to handle backpressure and back-to-back events.

    reg pending_valid_r;
    reg [BAR_ADDR_W-1:0] pending_bar_r;

    wire done_fire = any_dxa_wr && is_last;
    wire [BAR_ADDR_W-1:0] done_fire_bar = bank_wr_tag[BAR_ADDR_W-1:0];
    wire done_accepted = done_valid && done_ready;

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (any_dxa_wr && !reset) begin
            `TRACE(2, ("%t: dxa-completion-detect: bank_wr_fire=%b tag=0x%0h is_last=%b done_fire=%b pending=%b done_valid=%b done_ready=%b\n",
                $time, bank_wr_fire, bank_wr_tag, is_last, done_fire, pending_valid_r, done_valid, done_ready))
        end
        if (done_accepted && !reset) begin
            `TRACE(2, ("%t: dxa-completion-detect: ACCEPTED bar_addr=0x%0h\n",
                $time, done_bar_addr))
        end
    end
`endif

    // Priority: emit pending first, then same-cycle fire
    assign done_valid = pending_valid_r || (done_fire && ~pending_valid_r);
    assign done_bar_addr = pending_valid_r ? pending_bar_r : done_fire_bar;

    always @(posedge clk) begin
        if (reset) begin
            pending_valid_r <= 1'b0;
            pending_bar_r <= '0;
        end else begin
            if (pending_valid_r) begin
                if (done_accepted) begin
                    // Pending was accepted by downstream
                    if (done_fire) begin
                        // New event same cycle — refill pending
                        pending_valid_r <= 1'b1;
                        pending_bar_r <= done_fire_bar;
                    end else begin
                        pending_valid_r <= 1'b0;
                        pending_bar_r <= '0;
                    end
                end
                // If !done_accepted: keep pending as-is (hold valid high)
            end else if (done_fire) begin
                if (!done_ready) begin
                    // Not accepted — queue it in pending
                    pending_valid_r <= 1'b1;
                    pending_bar_r <= done_fire_bar;
                end
                // If done_ready=1: accepted immediately, no pending needed
            end
        end
    end
`else
    assign done_valid = 1'b0;
    assign done_bar_addr = '0;
    `UNUSED_VAR ({clk, reset})
    `UNUSED_VAR ({bank_wr_fire, bank_wr_tag})
`endif
endmodule
