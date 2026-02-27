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
    output wire [BAR_ADDR_W-1:0]               done_bar_addr
);
`ifdef EXT_DXA_ENABLE
    // Tag format: {last_pkt, bar_addr}
    // With shared tag across banks, at most 1 done event per DXA write cycle.
    wire is_last = bank_wr_tag[TAG_WIDTH-1];
    wire any_dxa_wr = |bank_wr_fire;

    // Simple combinational output — at most 1 done event per cycle
    // (each worker issues 1 DXA write/cycle, shared tag across all banks).
    // Use a single pending slot to handle back-to-back done events.

    reg pending_valid_r;
    reg [BAR_ADDR_W-1:0] pending_bar_r;

    wire done_fire = any_dxa_wr && is_last;
    wire [BAR_ADDR_W-1:0] done_fire_bar = bank_wr_tag[BAR_ADDR_W-1:0];

    // Priority: emit pending first, then same-cycle fire
    wire emit_pending = pending_valid_r;
    wire emit_new = done_fire && ~pending_valid_r;

    assign done_valid = emit_pending || emit_new;
    assign done_bar_addr = emit_pending ? pending_bar_r : done_fire_bar;

    always @(posedge clk) begin
        if (reset) begin
            pending_valid_r <= 1'b0;
            pending_bar_r <= '0;
        end else begin
            if (pending_valid_r) begin
                // Pending was emitted this cycle
                if (done_fire) begin
                    // New event arrived while emitting pending — queue it
                    pending_valid_r <= 1'b1;
                    pending_bar_r <= done_fire_bar;
                end else begin
                    pending_valid_r <= 1'b0;
                    pending_bar_r <= '0;
                end
            end else if (done_fire) begin
                // No pending, new event emitted directly — no queue needed
                pending_valid_r <= 1'b0;
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
