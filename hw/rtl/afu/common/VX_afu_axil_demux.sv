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
// VX_afu_axil_demux — one-hot AXI4-Lite demux splitting the AFU control
// space by address bit `SEL_BIT`:
//
//   addr[SEL_BIT] == 0 → port 0 (legacy AFU_ctrl, host 0x0000..0x0FFF)
//   addr[SEL_BIT] == 1 → port 1 (CP regfile,      host 0x1000..0x1FFF)
//
// One outstanding write and one outstanding read. AW/AR are stalled while a
// transaction of the same direction is pending, so a route is never reused
// for a transaction that belongs to the other slave.
//
// The W channel is the subtle part. AXI4 permits — and the spec forbids
// masters from avoiding — write data arriving BEFORE or IN THE SAME CYCLE as
// its write address, so W routing must not depend on a register that is only
// updated at the AW handshake:
//
//   * a write is already pending  → use its latched route;
//   * else AW is firing this cycle → fall through and use the incoming
//     address combinationally, so a same-cycle AW+W pair routes together;
//   * else the route is unknown   → hold wready low until AW arrives.
//
// An earlier version routed W by the latched route alone. A legacy-window
// write whose predecessor went to the CP window therefore sent AW to
// AFU_ctrl and W to the CP regfile: AFU_ctrl stalled forever waiting for a
// W beat the CP had already swallowed, no BRESP was ever produced, and every
// later host access died of a PCIe completion timeout. See
// docs/proposals/afu_reset_architecture_proposal.md.
// ============================================================================

module VX_afu_axil_demux #(
    parameter ADDR_WIDTH = 16,
    parameter DATA_WIDTH = 32,
    parameter SEL_BIT    = 12
) (
    input  wire                         clk,
    input  wire                         reset,

    // ---- upstream (host) ----
    input  wire                         s_awvalid,
    output wire                         s_awready,
    input  wire [ADDR_WIDTH-1:0]        s_awaddr,

    input  wire                         s_wvalid,
    output wire                         s_wready,
    input  wire [DATA_WIDTH-1:0]        s_wdata,
    input  wire [DATA_WIDTH/8-1:0]      s_wstrb,

    output wire                         s_bvalid,
    input  wire                         s_bready,
    output wire [1:0]                   s_bresp,

    input  wire                         s_arvalid,
    output wire                         s_arready,
    input  wire [ADDR_WIDTH-1:0]        s_araddr,

    output wire                         s_rvalid,
    input  wire                         s_rready,
    output wire [DATA_WIDTH-1:0]        s_rdata,
    output wire [1:0]                   s_rresp,

    // ---- downstream port 0: legacy AFU_ctrl ----
    output wire                         m0_awvalid,
    input  wire                         m0_awready,
    output wire [ADDR_WIDTH-1:0]        m0_awaddr,

    output wire                         m0_wvalid,
    input  wire                         m0_wready,
    output wire [DATA_WIDTH-1:0]        m0_wdata,
    output wire [DATA_WIDTH/8-1:0]      m0_wstrb,

    input  wire                         m0_bvalid,
    output wire                         m0_bready,
    input  wire [1:0]                   m0_bresp,

    output wire                         m0_arvalid,
    input  wire                         m0_arready,
    output wire [ADDR_WIDTH-1:0]        m0_araddr,

    input  wire                         m0_rvalid,
    output wire                         m0_rready,
    input  wire [DATA_WIDTH-1:0]        m0_rdata,
    input  wire [1:0]                   m0_rresp,

    // ---- downstream port 1: CP regfile ----
    output wire                         m1_awvalid,
    input  wire                         m1_awready,
    output wire [ADDR_WIDTH-1:0]        m1_awaddr,

    output wire                         m1_wvalid,
    input  wire                         m1_wready,
    output wire [DATA_WIDTH-1:0]        m1_wdata,
    output wire [DATA_WIDTH/8-1:0]      m1_wstrb,

    input  wire                         m1_bvalid,
    output wire                         m1_bready,
    input  wire [1:0]                   m1_bresp,

    output wire                         m1_arvalid,
    input  wire                         m1_arready,
    output wire [ADDR_WIDTH-1:0]        m1_araddr,

    input  wire                         m1_rvalid,
    output wire                         m1_rready,
    input  wire [DATA_WIDTH-1:0]        m1_rdata,
    input  wire [1:0]                   m1_rresp
);
    // Address presented to a slave with the select bit (and everything above
    // it) cleared, so each slave sees its own 0-based window.
    wire [ADDR_WIDTH-1:0] awaddr_local = {{(ADDR_WIDTH-SEL_BIT){1'b0}}, s_awaddr[SEL_BIT-1:0]};
    wire [ADDR_WIDTH-1:0] araddr_local = {{(ADDR_WIDTH-SEL_BIT){1'b0}}, s_araddr[SEL_BIT-1:0]};

    wire sel_aw = s_awaddr[SEL_BIT];
    wire sel_ar = s_araddr[SEL_BIT];

    // ------------------------------------------------------------------
    // Write channel
    // ------------------------------------------------------------------
    reg  wr_pending;    // a write has been accepted and has not returned B
    reg  wr_route_r;    // which slave that write went to

    wire aw_fire = s_awvalid && s_awready;
    wire b_fire  = s_bvalid && s_bready;

    // AW is blocked while a write is pending, so wr_route_r can never be
    // overwritten while it is still needed for W and B.
    assign m0_awvalid = s_awvalid && !wr_pending && !sel_aw;
    assign m1_awvalid = s_awvalid && !wr_pending &&  sel_aw;
    assign m0_awaddr  = awaddr_local;
    assign m1_awaddr  = awaddr_local;
    assign s_awready  = !wr_pending && (sel_aw ? m1_awready : m0_awready);

    // W route: latched once the write is pending, otherwise falling through
    // from the AW being accepted this very cycle. Unknown until one of those
    // holds, which is the only case where wready must stay low.
    wire wr_route_known = wr_pending || aw_fire;
    wire wr_route       = wr_pending ? wr_route_r : sel_aw;

    assign m0_wvalid = s_wvalid && wr_route_known && !wr_route;
    assign m1_wvalid = s_wvalid && wr_route_known &&  wr_route;
    assign m0_wdata  = s_wdata;
    assign m1_wdata  = s_wdata;
    assign m0_wstrb  = s_wstrb;
    assign m1_wstrb  = s_wstrb;
    assign s_wready  = wr_route_known && (wr_route ? m1_wready : m0_wready);

    assign s_bvalid  = wr_pending && (wr_route_r ? m1_bvalid : m0_bvalid);
    assign s_bresp   = wr_route_r ? m1_bresp : m0_bresp;
    assign m0_bready = s_bready && wr_pending && !wr_route_r;
    assign m1_bready = s_bready && wr_pending &&  wr_route_r;

    always @(posedge clk) begin
        if (reset) begin
            wr_pending <= 1'b0;
            wr_route_r <= 1'b0;
        end else begin
            if (aw_fire) begin
                wr_pending <= 1'b1;
                wr_route_r <= sel_aw;
            end else if (b_fire) begin
                wr_pending <= 1'b0;
            end
        end
    end

    // ------------------------------------------------------------------
    // Read channel
    // ------------------------------------------------------------------
    reg  rd_pending;
    reg  rd_route_r;

    wire ar_fire = s_arvalid && s_arready;
    wire r_fire  = s_rvalid && s_rready;

    assign m0_arvalid = s_arvalid && !rd_pending && !sel_ar;
    assign m1_arvalid = s_arvalid && !rd_pending &&  sel_ar;
    assign m0_araddr  = araddr_local;
    assign m1_araddr  = araddr_local;
    assign s_arready  = !rd_pending && (sel_ar ? m1_arready : m0_arready);

    assign s_rvalid  = rd_pending && (rd_route_r ? m1_rvalid : m0_rvalid);
    assign s_rdata   = rd_route_r ? m1_rdata : m0_rdata;
    assign s_rresp   = rd_route_r ? m1_rresp : m0_rresp;
    assign m0_rready = s_rready && rd_pending && !rd_route_r;
    assign m1_rready = s_rready && rd_pending &&  rd_route_r;

    always @(posedge clk) begin
        if (reset) begin
            rd_pending <= 1'b0;
            rd_route_r <= 1'b0;
        end else begin
            if (ar_fire) begin
                rd_pending <= 1'b1;
                rd_route_r <= sel_ar;
            end else if (r_fire) begin
                rd_pending <= 1'b0;
            end
        end
    end

endmodule
