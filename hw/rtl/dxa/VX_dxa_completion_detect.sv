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
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_BANKS = 1,
    parameter TAG_WIDTH = DXA_BANK_WR_TAG_WIDTH
) (
    input  wire                                clk,
    input  wire                                reset,

    // DXA bank writes (from dedicated bank_wr port, all are DXA by definition)
    input  wire [NUM_BANKS-1:0]                bank_wr_fire,
    input  wire [TAG_WIDTH-1:0]                bank_wr_tag,    // shared tag: {last_pkt, bar_addr}

    VX_txbar_bus_if.master                     txbar_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // Tag format: {last_pkt, bar_addr}
    // With shared tag across banks, at most 1 done event per DXA write cycle.
    wire is_last = bank_wr_tag[TAG_WIDTH-1];
    wire any_dxa_wr = |bank_wr_fire;

    // Valid/ready handshake: hold valid high until downstream accepts.
    // Use a single pending slot to handle backpressure and back-to-back events.

    reg pending_valid_r;
    reg [BAR_ADDR_W-1:0] pending_bar_r;

    wire done_fire = any_dxa_wr && is_last;
    wire [BAR_ADDR_W-1:0] done_fire_bar = bank_wr_tag[BAR_ADDR_W-1:0];
    wire done_accepted = txbar_bus_if.valid && txbar_bus_if.ready;

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (any_dxa_wr && !reset) begin
            `TRACE(2, ("%t: %s: bank_wr_fire=%b tag=0x%0h is_last=%b done_fire=%b pending=%b valid=%b ready=%b\n",
                $time, INSTANCE_ID, bank_wr_fire, bank_wr_tag, is_last, done_fire, pending_valid_r, txbar_bus_if.valid, txbar_bus_if.ready))
        end
        if (done_accepted && !reset) begin
            `TRACE(2, ("%t: %s: ACCEPTED bar_addr=0x%0h\n",
                $time, INSTANCE_ID, txbar_bus_if.data.addr))
        end
        if (~reset && done_accepted) begin
            $write("DXA_TL,%0d,DONE_DETECT,%s,bar=%0d\n",
                $time, INSTANCE_ID, txbar_bus_if.data.addr);
        end
    end
`endif

    // Priority: emit pending first, then same-cycle fire
    assign txbar_bus_if.valid        = pending_valid_r || (done_fire && ~pending_valid_r);
    assign txbar_bus_if.data.addr    = pending_valid_r ? pending_bar_r : done_fire_bar;
    assign txbar_bus_if.data.is_done = 1'b1;

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
                if (!txbar_bus_if.ready) begin
                    // Not accepted — queue it in pending
                    pending_valid_r <= 1'b1;
                    pending_bar_r <= done_fire_bar;
                end
                // If ready=1: accepted immediately, no pending needed
            end
        end
    end
endmodule
