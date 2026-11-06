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

`include "VX_platform.vh"

// ============================================================================
// VX_afu_axi_limit — holds an AXI master to one read and one write in flight.
//
// A new AR is withheld until the previous read's last beat, and a new AW until
// the previous write's B. The limits only ever tighten on a handshake, so an
// offer already made to the slave is never withdrawn.
//
// AW and W of one write may fire in either order. W is let through whenever
// its AW has already gone, and AW whenever its W has, whatever the limit says
// by then: withholding half of an accepted write would hang it.
// ============================================================================

`TRACING_OFF
module VX_afu_axi_limit (
    input  wire clk,
    input  wire reset,

    input  wire in_awvalid,
    output wire in_awready,
    input  wire in_wvalid,
    output wire in_wready,
    input  wire in_wlast,
    input  wire in_arvalid,
    output wire in_arready,

    output wire out_awvalid,
    input  wire out_awready,
    output wire out_wvalid,
    input  wire out_wready,
    output wire out_arvalid,
    input  wire out_arready,

    input  wire b_fire,
    input  wire r_fire_last
);
    reg rd_busy;
    reg wr_busy;
    reg w_owed;   // AW sent, its W still to come
    reg aw_owed;  // W sent, its AW still to come

    wire aw_allow = aw_owed || ~wr_busy;
    wire w_allow  = w_owed || (~wr_busy && ~aw_owed);
    wire ar_allow = ~rd_busy;

    assign out_awvalid = in_awvalid && aw_allow;
    assign in_awready  = out_awready && aw_allow;
    assign out_wvalid  = in_wvalid && w_allow;
    assign in_wready   = out_wready && w_allow;
    assign out_arvalid = in_arvalid && ar_allow;
    assign in_arready  = out_arready && ar_allow;

    wire aw_fire     = out_awvalid && out_awready;
    wire w_fire_last = out_wvalid && out_wready && in_wlast;
    wire ar_fire     = out_arvalid && out_arready;

    always @(posedge clk) begin
        if (reset) begin
            rd_busy <= 1'b0;
            wr_busy <= 1'b0;
            w_owed  <= 1'b0;
            aw_owed <= 1'b0;
        end else begin
            if (ar_fire) begin
                rd_busy <= 1'b1;
            end else if (r_fire_last) begin
                rd_busy <= 1'b0;
            end
            if (aw_fire) begin
                wr_busy <= 1'b1;
            end else if (b_fire) begin
                wr_busy <= 1'b0;
            end
            if (aw_fire && ~w_fire_last) begin
                aw_owed <= 1'b0;
                w_owed  <= ~aw_owed;
            end else if (w_fire_last && ~aw_fire) begin
                w_owed  <= 1'b0;
                aw_owed <= ~w_owed;
            end
        end
    end

endmodule
`TRACING_ON
