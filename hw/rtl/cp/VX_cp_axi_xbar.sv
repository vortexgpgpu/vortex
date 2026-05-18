// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_axi_xbar — fans N_SOURCES internal AXI4 sub-masters into the
// single upstream AXI master exposed by VX_cp_core.
//
// Sources: per-CPE fetches + DMA + completion (and, optionally, event_unit
// + profiling). Each source gets a unique TID prefix in the high bits of
// arid / awid; responses are routed back by inspecting the same bits on
// rid / bid.
//
// Arbitration:
//   - AR channel: per-cycle round-robin among sources asserting arvalid.
//     Single grant per cycle.
//   - AW channel: same.
//   - W channel: must follow the AW grant in lockstep — AXI4 requires W
//     beats arrive in AW issue order. We track the most-recent AW grant
//     and route W from that source until wlast.
//   - R channel: routed by rid[ID_W-1:SUB_ID_W] back to the source.
//   - B channel: routed by bid[ID_W-1:SUB_ID_W] back to the source.
//
// TID layout:
//   [ID_W-1 : SUB_ID_W]    = source index (managed by the xbar)
//   [SUB_ID_W-1 : 0]       = sub-tag (each source uses these as it sees
//                            fit — fetch ignores; DMA uses for multi-burst
//                            tracking; etc.)
// ============================================================================

module VX_cp_axi_xbar
  import VX_cp_pkg::*;
#(
  parameter int N_SOURCES = 1,
  parameter int ADDR_W    = 64,
  parameter int DATA_W    = 512,
  parameter int ID_W      = VX_CP_AXI_TID_WIDTH_C
)(
  input  wire                       clk,
  input  wire                       reset,

  // Per-source sub-master ports (slave side here — we receive their
  // requests).
  VX_cp_axi_m_if.slave              src   [N_SOURCES],

  // Upstream master port (we drive this).
  VX_cp_axi_m_if.master             axi_m
);

  localparam int SRC_W = (N_SOURCES > 1) ? $clog2(N_SOURCES) : 1;

  // ---- Unpack interface arrays into plain arrays for indexing ----
  // (verilator can't directly index unpacked-array interfaces inside
  // an always_comb that uses non-genvar indices.)
  wire                       s_awvalid [N_SOURCES];
  wire [ADDR_W-1:0]          s_awaddr  [N_SOURCES];
  wire [ID_W-1:0]            s_awid    [N_SOURCES];
  wire [7:0]                 s_awlen   [N_SOURCES];
  wire [2:0]                 s_awsize  [N_SOURCES];
  wire [1:0]                 s_awburst [N_SOURCES];
  logic                      s_awready [N_SOURCES];

  wire                       s_wvalid  [N_SOURCES];
  wire [DATA_W-1:0]          s_wdata   [N_SOURCES];
  wire [DATA_W/8-1:0]        s_wstrb   [N_SOURCES];
  wire                       s_wlast   [N_SOURCES];
  logic                      s_wready  [N_SOURCES];

  logic                      s_bvalid  [N_SOURCES];
  logic [ID_W-1:0]           s_bid     [N_SOURCES];
  logic [1:0]                s_bresp   [N_SOURCES];
  wire                       s_bready  [N_SOURCES];

  wire                       s_arvalid [N_SOURCES];
  wire [ADDR_W-1:0]          s_araddr  [N_SOURCES];
  wire [ID_W-1:0]            s_arid    [N_SOURCES];
  wire [7:0]                 s_arlen   [N_SOURCES];
  wire [2:0]                 s_arsize  [N_SOURCES];
  wire [1:0]                 s_arburst [N_SOURCES];
  logic                      s_arready [N_SOURCES];

  logic                      s_rvalid  [N_SOURCES];
  logic [DATA_W-1:0]         s_rdata   [N_SOURCES];
  logic [ID_W-1:0]           s_rid     [N_SOURCES];
  logic                      s_rlast   [N_SOURCES];
  logic [1:0]                s_rresp   [N_SOURCES];
  wire                       s_rready  [N_SOURCES];

  generate
    for (genvar i = 0; i < N_SOURCES; ++i) begin : g_unpack
      assign s_awvalid[i]   = src[i].awvalid;
      assign s_awaddr[i]    = src[i].awaddr;
      assign s_awid[i]      = src[i].awid;
      assign s_awlen[i]     = src[i].awlen;
      assign s_awsize[i]    = src[i].awsize;
      assign s_awburst[i]   = src[i].awburst;
      assign src[i].awready = s_awready[i];

      assign s_wvalid[i]    = src[i].wvalid;
      assign s_wdata[i]     = src[i].wdata;
      assign s_wstrb[i]     = src[i].wstrb;
      assign s_wlast[i]     = src[i].wlast;
      assign src[i].wready  = s_wready[i];

      assign src[i].bvalid  = s_bvalid[i];
      assign src[i].bid     = s_bid[i];
      assign src[i].bresp   = s_bresp[i];
      assign s_bready[i]    = src[i].bready;

      assign s_arvalid[i]   = src[i].arvalid;
      assign s_araddr[i]    = src[i].araddr;
      assign s_arid[i]      = src[i].arid;
      assign s_arlen[i]     = src[i].arlen;
      assign s_arsize[i]    = src[i].arsize;
      assign s_arburst[i]   = src[i].arburst;
      assign src[i].arready = s_arready[i];

      assign src[i].rvalid  = s_rvalid[i];
      assign src[i].rdata   = s_rdata[i];
      assign src[i].rid     = s_rid[i];
      assign src[i].rlast   = s_rlast[i];
      assign src[i].rresp   = s_rresp[i];
      assign s_rready[i]    = src[i].rready;
    end
  endgenerate

  // ============================================================================
  // AR channel — round-robin grant; tag the issued arid with the source
  // index in the high bits.
  // ============================================================================

  logic [SRC_W-1:0] ar_rr_ptr;
  logic [SRC_W-1:0] ar_winner;
  logic             ar_any;

  always_comb begin
    ar_winner = '0;
    ar_any    = 1'b0;
    for (int unsigned i = 0; i < N_SOURCES; ++i) begin
      logic [SRC_W:0] sum;
      logic [SRC_W-1:0] idx;
      sum = {1'b0, ar_rr_ptr} + (SRC_W+1)'(i);
      idx = (sum >= (SRC_W+1)'(N_SOURCES))
              ? SRC_W'(sum - (SRC_W+1)'(N_SOURCES))
              : SRC_W'(sum);
      if (!ar_any && s_arvalid[idx]) begin
        ar_any    = 1'b1;
        ar_winner = idx;
      end
    end
  end

  // Drive grants to the winner only.
  always_comb begin
    for (int i = 0; i < N_SOURCES; ++i) begin
      s_arready[i] = 1'b0;
    end
    if (ar_any) s_arready[ar_winner] = axi_m.arready;
  end

  // Drive upstream AR from the winner; arid high bits = winner index.
  always_comb begin
    axi_m.arvalid = ar_any && s_arvalid[ar_winner];
    axi_m.araddr  = s_araddr [ar_winner];
    axi_m.arlen   = s_arlen  [ar_winner];
    axi_m.arsize  = s_arsize [ar_winner];
    axi_m.arburst = s_arburst[ar_winner];
    axi_m.arid    = '0;
    axi_m.arid[ID_W-1 -: SRC_W] = ar_winner;
    // Pass the source's sub-tag through unchanged in the low bits.
    axi_m.arid[ID_W-SRC_W-1:0]  = s_arid[ar_winner][ID_W-SRC_W-1:0];
  end

  always_ff @(posedge clk) begin
    if (reset) begin
      ar_rr_ptr <= '0;
    end else if (axi_m.arvalid && axi_m.arready) begin
      // Advance rr_ptr past the winner.
      logic [SRC_W:0] nxt;
      nxt = {1'b0, ar_winner} + (SRC_W+1)'(1);
      ar_rr_ptr <= (nxt >= (SRC_W+1)'(N_SOURCES))
                     ? SRC_W'(nxt - (SRC_W+1)'(N_SOURCES))
                     : SRC_W'(nxt);
    end
  end

  // ============================================================================
  // R channel — route by high bits of rid.
  // ============================================================================

  wire [SRC_W-1:0] r_route = axi_m.rid[ID_W-1 -: SRC_W];
  always_comb begin
    for (int i = 0; i < N_SOURCES; ++i) begin
      s_rvalid[i] = 1'b0;
      s_rdata[i]  = '0;
      s_rid[i]    = '0;
      s_rlast[i]  = 1'b0;
      s_rresp[i]  = 2'b00;
    end
    if (axi_m.rvalid) begin
      s_rvalid[r_route] = 1'b1;
      s_rdata[r_route]  = axi_m.rdata;
      s_rid[r_route]    = {{SRC_W{1'b0}}, axi_m.rid[ID_W-SRC_W-1:0]};
      s_rlast[r_route]  = axi_m.rlast;
      s_rresp[r_route]  = axi_m.rresp;
    end
    axi_m.rready = s_rready[r_route];
  end

  // ============================================================================
  // AW + W channels — similar round-robin, but W follows the AW grant.
  // ============================================================================

  logic [SRC_W-1:0] aw_rr_ptr;
  logic [SRC_W-1:0] aw_winner;
  logic             aw_any;

  always_comb begin
    aw_winner = '0;
    aw_any    = 1'b0;
    for (int unsigned i = 0; i < N_SOURCES; ++i) begin
      logic [SRC_W:0] sum;
      logic [SRC_W-1:0] idx;
      sum = {1'b0, aw_rr_ptr} + (SRC_W+1)'(i);
      idx = (sum >= (SRC_W+1)'(N_SOURCES))
              ? SRC_W'(sum - (SRC_W+1)'(N_SOURCES))
              : SRC_W'(sum);
      if (!aw_any && s_awvalid[idx]) begin
        aw_any    = 1'b1;
        aw_winner = idx;
      end
    end
  end

  always_comb begin
    for (int i = 0; i < N_SOURCES; ++i) s_awready[i] = 1'b0;
    if (aw_any) s_awready[aw_winner] = axi_m.awready;
  end

  always_comb begin
    axi_m.awvalid = aw_any && s_awvalid[aw_winner];
    axi_m.awaddr  = s_awaddr [aw_winner];
    axi_m.awlen   = s_awlen  [aw_winner];
    axi_m.awsize  = s_awsize [aw_winner];
    axi_m.awburst = s_awburst[aw_winner];
    axi_m.awid    = '0;
    axi_m.awid[ID_W-1 -: SRC_W] = aw_winner;
    axi_m.awid[ID_W-SRC_W-1:0]  = s_awid[aw_winner][ID_W-SRC_W-1:0];
  end

  // W routing follows the most recent AW grant until wlast.
  logic             w_active;
  logic [SRC_W-1:0] w_route;

  always_ff @(posedge clk) begin
    if (reset) begin
      aw_rr_ptr <= '0;
      w_active  <= 1'b0;
      w_route   <= '0;
    end else begin
      if (axi_m.awvalid && axi_m.awready) begin
        logic [SRC_W:0] nxt;
        nxt = {1'b0, aw_winner} + (SRC_W+1)'(1);
        aw_rr_ptr <= (nxt >= (SRC_W+1)'(N_SOURCES))
                       ? SRC_W'(nxt - (SRC_W+1)'(N_SOURCES))
                       : SRC_W'(nxt);
        // Start routing W from the granted source.
        w_active <= 1'b1;
        w_route  <= aw_winner;
      end
      if (w_active && axi_m.wvalid && axi_m.wready && axi_m.wlast) begin
        w_active <= 1'b0;
      end
    end
  end

  // Drive W from the routed source.
  always_comb begin
    for (int i = 0; i < N_SOURCES; ++i) s_wready[i] = 1'b0;
    axi_m.wvalid = 1'b0;
    axi_m.wdata  = '0;
    axi_m.wstrb  = '0;
    axi_m.wlast  = 1'b0;
    if (w_active) begin
      axi_m.wvalid = s_wvalid[w_route];
      axi_m.wdata  = s_wdata [w_route];
      axi_m.wstrb  = s_wstrb [w_route];
      axi_m.wlast  = s_wlast [w_route];
      s_wready[w_route] = axi_m.wready;
    end
  end

  // ============================================================================
  // B channel — route by high bits of bid.
  // ============================================================================

  wire [SRC_W-1:0] b_route = axi_m.bid[ID_W-1 -: SRC_W];
  always_comb begin
    for (int i = 0; i < N_SOURCES; ++i) begin
      s_bvalid[i] = 1'b0;
      s_bid[i]    = '0;
      s_bresp[i]  = 2'b00;
    end
    if (axi_m.bvalid) begin
      s_bvalid[b_route] = 1'b1;
      s_bid[b_route]    = {{SRC_W{1'b0}}, axi_m.bid[ID_W-SRC_W-1:0]};
      s_bresp[b_route]  = axi_m.bresp;
    end
    axi_m.bready = s_bready[b_route];
  end

endmodule : VX_cp_axi_xbar
