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
// VX_mm_axi_arb — N-master to M-slave AXI4 arbiter (reduced view), flat packed
// ports. The AXI analog of VX_stream_arb: same conventions (packed vectors,
// VX_generic_arbiter reuse, ARBITER/STICKY parameters, TRACING wrappers,
// optional registered output stage) and the same N->M capability surface.
//
// Reduced AXI4 channels (no size/burst/cache/prot sideband):
//   AW: valid/ready/addr/id/len   W : valid/ready/data/strb/last
//   B : valid/ready/id/resp       AR: valid/ready/addr/id/len
//   R : valid/ready/data/last/id/resp
//
// Topologies (mirroring VX_stream_arb):
//   * NUM_OUTPUTS == 1  : merge N masters onto 1 slave (the common case).
//   * NUM_OUTPUTS  > 1  : concentrate N masters onto M slaves by a fixed
//                         partition (master i is bound to slave i % M); each
//                         slave runs its own N_i->1 merge. Requires N >= M.
//
// N < M (fan one/few masters across MORE slaves) has no address-free AXI
// meaning — a write/read is bound to a slave by its address. Use VX_mm_axi_xbar
// (address-`sel` routed) for master -> multi-slave distribution.
//
// Outstanding-transaction contract (MULTI_OUT, NUM_OUTPUTS==1 only):
//   * MULTI_OUT == 0 (default): single global outstanding per direction — once
//     AW (or AR) is accepted the slave sticks to that master until the matching
//     response completes; other masters stall. W follows the granted master
//     until WLAST. B/R route back to the owner. IDs pass through untouched.
//   * MULTI_OUT == 1: ID-routed, multi-outstanding fan-in. The top
//     LOG2UP(NUM_INPUTS) bits of the AXI ID carry the source index (masters
//     must leave them free); reads are fully concurrent, write bursts serialize
//     on the shared W channel (AW gated until WLAST) while B stays concurrent.
//     Requires ID_WIDTH > LOG2UP(NUM_INPUTS) and NUM_OUTPUTS == 1.
//
// OUT_REG (0/1): registered AXI stage (VX_mm_axi_slice) on each slave port.
// ============================================================================

`TRACING_OFF
module VX_mm_axi_arb #(
    parameter NUM_INPUTS  = 2,
    parameter NUM_OUTPUTS = 1,
    parameter ADDR_WIDTH  = 64,
    parameter DATA_WIDTH  = 512,
    parameter ID_WIDTH    = 32,
    parameter `STRING ARBITER = "P",
    parameter STICKY      = 0,
    parameter MULTI_OUT   = 0,   // 0: single-outstanding (ownership); 1: multi-outstanding (ID-routed)
    parameter OUT_REG     = 0,
    parameter STRB_WIDTH  = DATA_WIDTH/8,
    parameter SEL_WIDTH   = `LOG2UP(NUM_INPUTS)
) (
    input  wire clk,
    input  wire reset,

    // ---- Upstream masters (slave-side), packed ----
    input  wire [NUM_INPUTS-1:0]                  s_awvalid,
    output wire [NUM_INPUTS-1:0]                  s_awready,
    input  wire [NUM_INPUTS-1:0][ADDR_WIDTH-1:0]  s_awaddr,
    input  wire [NUM_INPUTS-1:0][ID_WIDTH-1:0]    s_awid,
    input  wire [NUM_INPUTS-1:0][7:0]             s_awlen,

    input  wire [NUM_INPUTS-1:0]                  s_wvalid,
    output wire [NUM_INPUTS-1:0]                  s_wready,
    input  wire [NUM_INPUTS-1:0][DATA_WIDTH-1:0]  s_wdata,
    input  wire [NUM_INPUTS-1:0][STRB_WIDTH-1:0]  s_wstrb,
    input  wire [NUM_INPUTS-1:0]                  s_wlast,

    output wire [NUM_INPUTS-1:0]                  s_bvalid,
    input  wire [NUM_INPUTS-1:0]                  s_bready,
    output wire [NUM_INPUTS-1:0][ID_WIDTH-1:0]    s_bid,
    output wire [NUM_INPUTS-1:0][1:0]             s_bresp,

    input  wire [NUM_INPUTS-1:0]                  s_arvalid,
    output wire [NUM_INPUTS-1:0]                  s_arready,
    input  wire [NUM_INPUTS-1:0][ADDR_WIDTH-1:0]  s_araddr,
    input  wire [NUM_INPUTS-1:0][ID_WIDTH-1:0]    s_arid,
    input  wire [NUM_INPUTS-1:0][7:0]             s_arlen,

    output wire [NUM_INPUTS-1:0]                  s_rvalid,
    input  wire [NUM_INPUTS-1:0]                  s_rready,
    output wire [NUM_INPUTS-1:0][DATA_WIDTH-1:0]  s_rdata,
    output wire [NUM_INPUTS-1:0]                  s_rlast,
    output wire [NUM_INPUTS-1:0][ID_WIDTH-1:0]    s_rid,
    output wire [NUM_INPUTS-1:0][1:0]             s_rresp,

    // ---- Downstream slaves (master-side), packed ----
    output wire [NUM_OUTPUTS-1:0]                 m_awvalid,
    input  wire [NUM_OUTPUTS-1:0]                 m_awready,
    output wire [NUM_OUTPUTS-1:0][ADDR_WIDTH-1:0] m_awaddr,
    output wire [NUM_OUTPUTS-1:0][ID_WIDTH-1:0]   m_awid,
    output wire [NUM_OUTPUTS-1:0][7:0]            m_awlen,

    output wire [NUM_OUTPUTS-1:0]                 m_wvalid,
    input  wire [NUM_OUTPUTS-1:0]                 m_wready,
    output wire [NUM_OUTPUTS-1:0][DATA_WIDTH-1:0] m_wdata,
    output wire [NUM_OUTPUTS-1:0][STRB_WIDTH-1:0] m_wstrb,
    output wire [NUM_OUTPUTS-1:0]                 m_wlast,

    input  wire [NUM_OUTPUTS-1:0]                 m_bvalid,
    output wire [NUM_OUTPUTS-1:0]                 m_bready,
    input  wire [NUM_OUTPUTS-1:0][ID_WIDTH-1:0]   m_bid,
    input  wire [NUM_OUTPUTS-1:0][1:0]            m_bresp,

    output wire [NUM_OUTPUTS-1:0]                 m_arvalid,
    input  wire [NUM_OUTPUTS-1:0]                 m_arready,
    output wire [NUM_OUTPUTS-1:0][ADDR_WIDTH-1:0] m_araddr,
    output wire [NUM_OUTPUTS-1:0][ID_WIDTH-1:0]   m_arid,
    output wire [NUM_OUTPUTS-1:0][7:0]            m_arlen,

    input  wire [NUM_OUTPUTS-1:0]                 m_rvalid,
    output wire [NUM_OUTPUTS-1:0]                 m_rready,
    input  wire [NUM_OUTPUTS-1:0][DATA_WIDTH-1:0] m_rdata,
    input  wire [NUM_OUTPUTS-1:0]                 m_rlast,
    input  wire [NUM_OUTPUTS-1:0][ID_WIDTH-1:0]   m_rid,
    input  wire [NUM_OUTPUTS-1:0][1:0]            m_rresp
);
    if (NUM_OUTPUTS == 1) begin : g_merge

        // ================= N -> 1 single-outstanding merge core =================
        // Arbiter core drives an internal slave-side view (c_*); an optional
        // register slice sits between the core and the true slave port.
        wire                  c_awvalid, c_awready;
        wire [ADDR_WIDTH-1:0] c_awaddr;
        wire [ID_WIDTH-1:0]   c_awid;
        wire [7:0]            c_awlen;

        wire                  c_wvalid, c_wready;
        wire [DATA_WIDTH-1:0] c_wdata;
        wire [STRB_WIDTH-1:0] c_wstrb;
        wire                  c_wlast;

        wire                  c_bvalid, c_bready;
        wire [ID_WIDTH-1:0]   c_bid;
        wire [1:0]            c_bresp;

        wire                  c_arvalid, c_arready;
        wire [ADDR_WIDTH-1:0] c_araddr;
        wire [ID_WIDTH-1:0]   c_arid;
        wire [7:0]            c_arlen;

        wire                  c_rvalid, c_rready;
        wire [DATA_WIDTH-1:0] c_rdata;
        wire                  c_rlast;
        wire [ID_WIDTH-1:0]   c_rid;
        wire [1:0]            c_rresp;

      if (!MULTI_OUT) begin : g_single

        // ---- Write channel: sticky single-outstanding owner ----
        reg                 owner_w_valid;
        reg [SEL_WIDTH-1:0] owner_w;
        reg                 w_in_progress;

        wire [SEL_WIDTH-1:0] aw_pick;
        wire                 aw_any;
        wire                 aw_fire     = c_awvalid && c_awready;
        wire                 w_last_fire = c_wvalid && c_wready && c_wlast;
        wire                 b_fire      = c_bvalid && c_bready;

        // Single master: no arbitration (avoids a degenerate NUM_REQS==1 arbiter).
        if (NUM_INPUTS == 1) begin : g_aw_single
            assign aw_pick = '0;
            assign aw_any  = s_awvalid[0];
        end else begin : g_aw_arb
            VX_generic_arbiter #(
                .NUM_REQS (NUM_INPUTS),
                .TYPE     (ARBITER),
                .STICKY   (STICKY)
            ) aw_arb (
                .clk          (clk),
                .reset        (reset),
                .requests     (s_awvalid),
                .grant_index  (aw_pick),
                `UNUSED_PIN   (grant_onehot),
                .grant_valid  (aw_any),
                .grant_ready  (aw_fire)
            );
        end

        always @(posedge clk) begin
            if (reset) begin
                owner_w_valid <= 1'b0;
                owner_w       <= '0;
                w_in_progress <= 1'b0;
            end else begin
                if (aw_fire && !owner_w_valid) begin
                    owner_w_valid <= 1'b1;
                    owner_w       <= aw_pick;
                    w_in_progress <= 1'b1;
                end
                if (w_in_progress && w_last_fire) begin
                    w_in_progress <= 1'b0;
                end
                if (b_fire) begin
                    owner_w_valid <= 1'b0;
                end
            end
        end

        assign c_awvalid = !owner_w_valid && aw_any;
        assign c_awaddr  = s_awaddr[aw_pick];
        assign c_awid    = s_awid  [aw_pick];
        assign c_awlen   = s_awlen [aw_pick];
        assign s_awready = {NUM_INPUTS{!owner_w_valid && c_awready}}
                         & ({{(NUM_INPUTS-1){1'b0}}, aw_any} << aw_pick);

        assign c_wvalid = w_in_progress && s_wvalid[owner_w];
        assign c_wdata  = s_wdata[owner_w];
        assign c_wstrb  = s_wstrb[owner_w];
        assign c_wlast  = s_wlast[owner_w];
        assign s_wready = {NUM_INPUTS{w_in_progress && c_wready}}
                        & ({{(NUM_INPUTS-1){1'b0}}, 1'b1} << owner_w);

        assign s_bvalid = {NUM_INPUTS{owner_w_valid && c_bvalid}}
                        & ({{(NUM_INPUTS-1){1'b0}}, 1'b1} << owner_w);
        for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_bresp
            assign s_bid[i]   = c_bid;
            assign s_bresp[i] = c_bresp;
        end
        assign c_bready = s_bready[owner_w];

        // ---- Read channel: sticky single-outstanding owner ----
        reg                 owner_r_valid;
        reg [SEL_WIDTH-1:0] owner_r;

        wire [SEL_WIDTH-1:0] ar_pick;
        wire                 ar_any;
        wire                 ar_fire     = c_arvalid && c_arready;
        wire                 r_last_fire = c_rvalid && c_rready && c_rlast;

        if (NUM_INPUTS == 1) begin : g_ar_single
            assign ar_pick = '0;
            assign ar_any  = s_arvalid[0];
        end else begin : g_ar_arb
            VX_generic_arbiter #(
                .NUM_REQS (NUM_INPUTS),
                .TYPE     (ARBITER),
                .STICKY   (STICKY)
            ) ar_arb (
                .clk          (clk),
                .reset        (reset),
                .requests     (s_arvalid),
                .grant_index  (ar_pick),
                `UNUSED_PIN   (grant_onehot),
                .grant_valid  (ar_any),
                .grant_ready  (ar_fire)
            );
        end

        always @(posedge clk) begin
            if (reset) begin
                owner_r_valid <= 1'b0;
                owner_r       <= '0;
            end else begin
                if (ar_fire && !owner_r_valid) begin
                    owner_r_valid <= 1'b1;
                    owner_r       <= ar_pick;
                end
                if (r_last_fire) begin
                    owner_r_valid <= 1'b0;
                end
            end
        end

        assign c_arvalid = !owner_r_valid && ar_any;
        assign c_araddr  = s_araddr[ar_pick];
        assign c_arid    = s_arid  [ar_pick];
        assign c_arlen   = s_arlen [ar_pick];
        assign s_arready = {NUM_INPUTS{!owner_r_valid && c_arready}}
                         & ({{(NUM_INPUTS-1){1'b0}}, ar_any} << ar_pick);

        assign s_rvalid = {NUM_INPUTS{owner_r_valid && c_rvalid}}
                        & ({{(NUM_INPUTS-1){1'b0}}, 1'b1} << owner_r);
        for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_rresp
            assign s_rdata[i] = c_rdata;
            assign s_rlast[i] = c_rlast;
            assign s_rid[i]   = c_rid;
            assign s_rresp[i] = c_rresp;
        end
        assign c_rready = s_rready[owner_r];

      end else begin : g_multi

        // ===== Multi-outstanding, ID-routed (source index in ID high bits) =====
        // The top SEL_WIDTH bits of the AXI ID carry the source index; masters
        // must leave them free. Reads are fully concurrent; write bursts serialize
        // on the shared W channel (AW gated until the prior burst's WLAST) while B
        // responses stay concurrent. Sticky arbitration holds the granted addr/id
        // stable until the handshake completes (AXI stability requirement).
        localparam LOW_ID = ID_WIDTH - SEL_WIDTH;   // source's own low ID bits
        `STATIC_ASSERT((ID_WIDTH > SEL_WIDTH),
            ("VX_mm_axi_arb MULTI_OUT: ID_WIDTH must exceed LOG2UP(NUM_INPUTS) to carry the source index"))

        // ---- Read: sticky RR grant, tag arid, demux R by rid ----
        wire [SEL_WIDTH-1:0] ar_pick;
        wire                 ar_any;
        if (NUM_INPUTS == 1) begin : g_ar_single
            assign ar_pick = '0;
            assign ar_any  = s_arvalid[0];
        end else begin : g_ar_arb
            wire ar_fire = c_arvalid && c_arready;
            VX_generic_arbiter #(
                .NUM_REQS (NUM_INPUTS),
                .TYPE     (ARBITER),
                .STICKY   (1)   // hold grant stable until AR fires (AXI stability)
            ) ar_arb (
                .clk (clk), .reset (reset), .requests (s_arvalid),
                .grant_index (ar_pick), `UNUSED_PIN (grant_onehot),
                .grant_valid (ar_any), .grant_ready (ar_fire)
            );
        end
        assign c_arvalid = ar_any;
        assign c_araddr  = s_araddr[ar_pick];
        assign c_arlen   = s_arlen [ar_pick];
        assign c_arid    = {ar_pick, s_arid[ar_pick][LOW_ID-1:0]};
        assign s_arready = {NUM_INPUTS{c_arready}}
                         & ({{(NUM_INPUTS-1){1'b0}}, ar_any} << ar_pick);

        wire [SEL_WIDTH-1:0] r_src = c_rid[ID_WIDTH-1 -: SEL_WIDTH];
        assign s_rvalid = {NUM_INPUTS{c_rvalid}}
                        & ({{(NUM_INPUTS-1){1'b0}}, 1'b1} << r_src);
        for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_rresp
            assign s_rdata[i] = c_rdata;
            assign s_rlast[i] = c_rlast;
            assign s_rid[i]   = {{SEL_WIDTH{1'b0}}, c_rid[LOW_ID-1:0]};
            assign s_rresp[i] = c_rresp;
        end
        assign c_rready = s_rready[r_src];

        // ---- Write: sticky RR grant gated on W-drain, tag awid, demux B by bid ----
        reg                 w_active;
        reg [SEL_WIDTH-1:0] w_route;
        wire [SEL_WIDTH-1:0] aw_pick;
        wire                 aw_any;
        wire                 aw_fire = c_awvalid && c_awready;
        if (NUM_INPUTS == 1) begin : g_aw_single
            assign aw_pick = '0;
            assign aw_any  = s_awvalid[0];
        end else begin : g_aw_arb
            VX_generic_arbiter #(
                .NUM_REQS (NUM_INPUTS),
                .TYPE     (ARBITER),
                .STICKY   (1)
            ) aw_arb (
                .clk (clk), .reset (reset), .requests (s_awvalid),
                .grant_index (aw_pick), `UNUSED_PIN (grant_onehot),
                .grant_valid (aw_any), .grant_ready (aw_fire)
            );
        end
        // Gate a new AW until the previous write's W burst drains.
        assign c_awvalid = aw_any && !w_active;
        assign c_awaddr  = s_awaddr[aw_pick];
        assign c_awlen   = s_awlen [aw_pick];
        assign c_awid    = {aw_pick, s_awid[aw_pick][LOW_ID-1:0]};
        assign s_awready = {NUM_INPUTS{!w_active && c_awready}}
                         & ({{(NUM_INPUTS-1){1'b0}}, aw_any} << aw_pick);

        always @(posedge clk) begin
            if (reset) begin
                w_active <= 1'b0;
                w_route  <= '0;
            end else begin
                if (aw_fire) begin
                    w_active <= 1'b1;
                    w_route  <= aw_pick;
                end else if (w_active && c_wvalid && c_wready && c_wlast) begin
                    w_active <= 1'b0;
                end
            end
        end

        assign c_wvalid = w_active && s_wvalid[w_route];
        assign c_wdata  = s_wdata[w_route];
        assign c_wstrb  = s_wstrb[w_route];
        assign c_wlast  = s_wlast[w_route];
        assign s_wready = {NUM_INPUTS{w_active && c_wready}}
                        & ({{(NUM_INPUTS-1){1'b0}}, 1'b1} << w_route);

        wire [SEL_WIDTH-1:0] b_src = c_bid[ID_WIDTH-1 -: SEL_WIDTH];
        assign s_bvalid = {NUM_INPUTS{c_bvalid}}
                        & ({{(NUM_INPUTS-1){1'b0}}, 1'b1} << b_src);
        for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_bresp
            assign s_bid[i]   = {{SEL_WIDTH{1'b0}}, c_bid[LOW_ID-1:0]};
            assign s_bresp[i] = c_bresp;
        end
        assign c_bready = s_bready[b_src];

      end

        // ---- Optional registered slave-side boundary ----
        if (OUT_REG != 0) begin : g_out_slice
            VX_mm_axi_slice #(
                .ADDR_WIDTH (ADDR_WIDTH),
                .DATA_WIDTH (DATA_WIDTH),
                .ID_WIDTH   (ID_WIDTH),
                .OUT_REG    (OUT_REG),
                .STRB_WIDTH (STRB_WIDTH)
            ) out_slice (
                .clk (clk), .reset (reset),
                .s_awvalid (c_awvalid), .s_awready (c_awready),
                .s_awaddr  (c_awaddr),  .s_awid (c_awid), .s_awlen (c_awlen),
                .s_wvalid  (c_wvalid),  .s_wready (c_wready),
                .s_wdata   (c_wdata),   .s_wstrb (c_wstrb), .s_wlast (c_wlast),
                .s_bvalid  (c_bvalid),  .s_bready (c_bready),
                .s_bid     (c_bid),     .s_bresp (c_bresp),
                .s_arvalid (c_arvalid), .s_arready (c_arready),
                .s_araddr  (c_araddr),  .s_arid (c_arid), .s_arlen (c_arlen),
                .s_rvalid  (c_rvalid),  .s_rready (c_rready),
                .s_rdata   (c_rdata),   .s_rlast (c_rlast),
                .s_rid     (c_rid),     .s_rresp (c_rresp),
                .m_awvalid (m_awvalid[0]), .m_awready (m_awready[0]),
                .m_awaddr  (m_awaddr[0]),  .m_awid (m_awid[0]), .m_awlen (m_awlen[0]),
                .m_wvalid  (m_wvalid[0]),  .m_wready (m_wready[0]),
                .m_wdata   (m_wdata[0]),   .m_wstrb (m_wstrb[0]), .m_wlast (m_wlast[0]),
                .m_bvalid  (m_bvalid[0]),  .m_bready (m_bready[0]),
                .m_bid     (m_bid[0]),     .m_bresp (m_bresp[0]),
                .m_arvalid (m_arvalid[0]), .m_arready (m_arready[0]),
                .m_araddr  (m_araddr[0]),  .m_arid (m_arid[0]), .m_arlen (m_arlen[0]),
                .m_rvalid  (m_rvalid[0]),  .m_rready (m_rready[0]),
                .m_rdata   (m_rdata[0]),   .m_rlast (m_rlast[0]),
                .m_rid     (m_rid[0]),     .m_rresp (m_rresp[0])
            );
        end else begin : g_passthru
            assign m_awvalid[0] = c_awvalid; assign c_awready = m_awready[0];
            assign m_awaddr[0]  = c_awaddr;  assign m_awid[0] = c_awid; assign m_awlen[0] = c_awlen;
            assign m_wvalid[0]  = c_wvalid;  assign c_wready = m_wready[0];
            assign m_wdata[0]   = c_wdata;   assign m_wstrb[0] = c_wstrb; assign m_wlast[0] = c_wlast;
            assign c_bvalid     = m_bvalid[0]; assign m_bready[0] = c_bready;
            assign c_bid        = m_bid[0];  assign c_bresp = m_bresp[0];
            assign m_arvalid[0] = c_arvalid; assign c_arready = m_arready[0];
            assign m_araddr[0]  = c_araddr;  assign m_arid[0] = c_arid; assign m_arlen[0] = c_arlen;
            assign c_rvalid     = m_rvalid[0]; assign m_rready[0] = c_rready;
            assign c_rdata      = m_rdata[0]; assign c_rlast = m_rlast[0];
            assign c_rid        = m_rid[0];  assign c_rresp = m_rresp[0];
        end

    end else begin : g_concentrate

        // ============ N -> M : master i bound to slave (i % NUM_OUTPUTS) ============
        `STATIC_ASSERT((NUM_INPUTS >= NUM_OUTPUTS),
            ("VX_mm_axi_arb: NUM_INPUTS must be >= NUM_OUTPUTS; use VX_mm_axi_xbar for master->multi-slave distribution"))
        `STATIC_ASSERT((MULTI_OUT == 0),
            ("VX_mm_axi_arb: MULTI_OUT is only supported for NUM_OUTPUTS==1 (ID-routed fan-in)"))

        for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_slave
            // Masters assigned to slave o: indices {o, o+M, o+2M, ...}.
            localparam GRP = (NUM_INPUTS - o - 1) / NUM_OUTPUTS + 1;

            wire [GRP-1:0]                  o_awvalid, o_awready;
            wire [GRP-1:0][ADDR_WIDTH-1:0]  o_awaddr;
            wire [GRP-1:0][ID_WIDTH-1:0]    o_awid;
            wire [GRP-1:0][7:0]             o_awlen;
            wire [GRP-1:0]                  o_wvalid, o_wready;
            wire [GRP-1:0][DATA_WIDTH-1:0]  o_wdata;
            wire [GRP-1:0][STRB_WIDTH-1:0]  o_wstrb;
            wire [GRP-1:0]                  o_wlast;
            wire [GRP-1:0]                  o_bvalid, o_bready;
            wire [GRP-1:0][ID_WIDTH-1:0]    o_bid;
            wire [GRP-1:0][1:0]             o_bresp;
            wire [GRP-1:0]                  o_arvalid, o_arready;
            wire [GRP-1:0][ADDR_WIDTH-1:0]  o_araddr;
            wire [GRP-1:0][ID_WIDTH-1:0]    o_arid;
            wire [GRP-1:0][7:0]             o_arlen;
            wire [GRP-1:0]                  o_rvalid, o_rready;
            wire [GRP-1:0][DATA_WIDTH-1:0]  o_rdata;
            wire [GRP-1:0]                  o_rlast;
            wire [GRP-1:0][ID_WIDTH-1:0]    o_rid;
            wire [GRP-1:0][1:0]             o_rresp;

            for (genvar k = 0; k < GRP; ++k) begin : g_map
                localparam gi = k * NUM_OUTPUTS + o;
                // master -> group (requests / write data)
                assign o_awvalid[k] = s_awvalid[gi];
                assign o_awaddr[k]  = s_awaddr[gi];
                assign o_awid[k]    = s_awid[gi];
                assign o_awlen[k]   = s_awlen[gi];
                assign o_wvalid[k]  = s_wvalid[gi];
                assign o_wdata[k]   = s_wdata[gi];
                assign o_wstrb[k]   = s_wstrb[gi];
                assign o_wlast[k]   = s_wlast[gi];
                assign o_bready[k]  = s_bready[gi];
                assign o_arvalid[k] = s_arvalid[gi];
                assign o_araddr[k]  = s_araddr[gi];
                assign o_arid[k]    = s_arid[gi];
                assign o_arlen[k]   = s_arlen[gi];
                assign o_rready[k]  = s_rready[gi];
                // group -> master (grants / responses)
                assign s_awready[gi] = o_awready[k];
                assign s_wready[gi]  = o_wready[k];
                assign s_bvalid[gi]  = o_bvalid[k];
                assign s_bid[gi]     = o_bid[k];
                assign s_bresp[gi]   = o_bresp[k];
                assign s_arready[gi] = o_arready[k];
                assign s_rvalid[gi]  = o_rvalid[k];
                assign s_rdata[gi]   = o_rdata[k];
                assign s_rlast[gi]   = o_rlast[k];
                assign s_rid[gi]     = o_rid[k];
                assign s_rresp[gi]   = o_rresp[k];
            end

            VX_mm_axi_arb #(
                .NUM_INPUTS  (GRP),
                .NUM_OUTPUTS (1),
                .ADDR_WIDTH  (ADDR_WIDTH),
                .DATA_WIDTH  (DATA_WIDTH),
                .ID_WIDTH    (ID_WIDTH),
                .ARBITER     (ARBITER),
                .STICKY      (STICKY),
                .OUT_REG     (OUT_REG),
                .STRB_WIDTH  (STRB_WIDTH)
            ) slave_arb (
                .clk (clk), .reset (reset),
                .s_awvalid (o_awvalid), .s_awready (o_awready),
                .s_awaddr  (o_awaddr),  .s_awid (o_awid), .s_awlen (o_awlen),
                .s_wvalid  (o_wvalid),  .s_wready (o_wready),
                .s_wdata   (o_wdata),   .s_wstrb (o_wstrb), .s_wlast (o_wlast),
                .s_bvalid  (o_bvalid),  .s_bready (o_bready),
                .s_bid     (o_bid),     .s_bresp (o_bresp),
                .s_arvalid (o_arvalid), .s_arready (o_arready),
                .s_araddr  (o_araddr),  .s_arid (o_arid), .s_arlen (o_arlen),
                .s_rvalid  (o_rvalid),  .s_rready (o_rready),
                .s_rdata   (o_rdata),   .s_rlast (o_rlast),
                .s_rid     (o_rid),     .s_rresp (o_rresp),
                .m_awvalid (m_awvalid[o]), .m_awready (m_awready[o]),
                .m_awaddr  (m_awaddr[o]),  .m_awid (m_awid[o]), .m_awlen (m_awlen[o]),
                .m_wvalid  (m_wvalid[o]),  .m_wready (m_wready[o]),
                .m_wdata   (m_wdata[o]),   .m_wstrb (m_wstrb[o]), .m_wlast (m_wlast[o]),
                .m_bvalid  (m_bvalid[o]),  .m_bready (m_bready[o]),
                .m_bid     (m_bid[o]),     .m_bresp (m_bresp[o]),
                .m_arvalid (m_arvalid[o]), .m_arready (m_arready[o]),
                .m_araddr  (m_araddr[o]),  .m_arid (m_arid[o]), .m_arlen (m_arlen[o]),
                .m_rvalid  (m_rvalid[o]),  .m_rready (m_rready[o]),
                .m_rdata   (m_rdata[o]),   .m_rlast (m_rlast[o]),
                .m_rid     (m_rid[o]),     .m_rresp (m_rresp[o])
            );
        end
    end

endmodule
`TRACING_ON
