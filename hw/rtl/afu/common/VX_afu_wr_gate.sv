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
// VX_afu_wr_gate — withhold new AXI writes without splitting one.
//
// The write-channel counterpart of VX_afu_req_gate. Gating AW alone is not
// enough: AXI4 lets a master present a write's data before or with its
// address, so a master that does would push W beats of a write whose AW the
// gate then withholds. The slave is left holding data no address will ever
// claim, the drain counters never balance, and the reset sequencer refuses.
//
// A write is the k-th AW paired with the k-th W burst. While stopped, the gate
// lets through only what completes a write already begun on the shell side:
// the AW owed to a W burst that has started or is being offered, the W burst
// owed to an AW that has been accepted or is being offered, the rest of a
// burst in progress, and any offer already presented (VALID is never
// withdrawn, AXI4 §A3.2.1). A slave may hold AWREADY until it sees the data,
// or WREADY until it sees the address, so an offer alone obliges the other
// channel. Everything else waits, so the shell sees whole writes only and the
// drain reaches zero.
//
// LEAD_WIDTH must exceed the number of AWs a master may run ahead of its data.
// ============================================================================

module VX_afu_wr_gate #(
    parameter LEAD_WIDTH = 16
) (
    input  wire clk,
    input  wire reset,

    input  wire stop_req,     // withhold new writes

    input  wire in_awvalid,   // from the AFU-internal master
    output wire in_awready,
    output wire out_awvalid,  // to the shell
    input  wire out_awready,

    input  wire in_wvalid,
    input  wire in_wlast,
    output wire in_wready,
    output wire out_wvalid,
    input  wire out_wready
);
    reg                  blocked;
    reg                  aw_offered;
    reg                  w_offered;
    reg                  w_in_burst;
    reg [LEAD_WIDTH-1:0] aw_lead;    // AWs accepted minus W bursts started

    wire aw_fire  = out_awvalid && out_awready;
    wire w_fire   = out_wvalid && out_wready;
    wire w_start  = w_fire && ~w_in_burst;

    always @(posedge clk) begin
        if (reset) begin
            blocked    <= 1'b0;
            aw_offered <= 1'b0;
            w_offered  <= 1'b0;
            w_in_burst <= 1'b0;
            aw_lead    <= '0;
        end else begin
            blocked    <= stop_req;
            aw_offered <= out_awvalid && ~out_awready;
            w_offered  <= out_wvalid && ~out_wready;
            if (w_fire) begin
                w_in_burst <= ~in_wlast;
            end
            aw_lead <= aw_lead + LEAD_WIDTH'(aw_fire) - LEAD_WIDTH'(w_start);
        end
    end

    wire lead_neg  = aw_lead[LEAD_WIDTH-1];
    wire lead_zero = (aw_lead == '0);
    wire aw_owed   = lead_neg || (lead_zero && w_offered && ~w_in_burst);
    wire w_owed    = (~lead_neg && ~lead_zero) || (lead_zero && aw_offered);

    wire aw_pass = ~blocked || aw_offered || aw_owed;
    wire w_pass  = ~blocked || w_offered || w_in_burst || w_owed;

    assign out_awvalid = in_awvalid && aw_pass;
    assign in_awready  = out_awready && aw_pass;
    assign out_wvalid  = in_wvalid && w_pass;
    assign in_wready   = out_wready && w_pass;

endmodule
