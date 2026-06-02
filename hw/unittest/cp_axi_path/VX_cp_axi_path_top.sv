// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_axi_path_top — instantiates fetch + completion through the xbar
// against the single upstream AXI master, with all signals exposed as
// flat scalar ports for the C++ harness to act as the upstream slave
// (a synthetic AXI4 memory) and the per-CPE driver (cpe_state +
// retire_evt).
//
// Pinned at NUM_QUEUES = 1; the xbar still has N_SOURCES = 2 (fetch +
// completion) so we exercise its arbitration logic end-to-end.
// ============================================================================

module VX_cp_axi_path_top
  import VX_cp_pkg::*;
#(
  parameter int ADDR_W = 64,
  parameter int DATA_W = 512,
  parameter int ID_W   = VX_CP_AXI_TID_WIDTH_C
)(
  input  wire                       clk,
  input  wire                       reset,

  // ---- Per-CPE state inputs (flattened cpe_state_t) ----
  input  wire [$bits(cpe_state_t)-1:0] state_in_packed,
  output wire [63:0]                head_out,

  // ---- Decoded command stream from fetch → would feed engine ----
  output wire                       cmd_out_valid,
  output wire [$bits(cmd_t)-1:0]    cmd_out_packed,
  input  wire                       cmd_out_ready,

  // ---- Retire pulses to completion ----
  input  wire                       retire_evt,
  input  wire [63:0]                retire_seqnum,
  input  wire [63:0]                cmpl_addr,

  // ---- Upstream AXI4 master (driven by xbar; harness implements slave) ----
  output wire                       m_awvalid,
  input  wire                       m_awready,
  output wire [ADDR_W-1:0]          m_awaddr,
  output wire [ID_W-1:0]            m_awid,
  output wire [7:0]                 m_awlen,
  output wire [2:0]                 m_awsize,
  output wire [1:0]                 m_awburst,

  output wire                       m_wvalid,
  input  wire                       m_wready,
  output wire [DATA_W-1:0]          m_wdata,
  output wire [DATA_W/8-1:0]        m_wstrb,
  output wire                       m_wlast,

  input  wire                       m_bvalid,
  output wire                       m_bready,
  input  wire [ID_W-1:0]            m_bid,
  input  wire [1:0]                 m_bresp,

  output wire                       m_arvalid,
  input  wire                       m_arready,
  output wire [ADDR_W-1:0]          m_araddr,
  output wire [ID_W-1:0]            m_arid,
  output wire [7:0]                 m_arlen,
  output wire [2:0]                 m_arsize,
  output wire [1:0]                 m_arburst,

  input  wire                       m_rvalid,
  output wire                       m_rready,
  input  wire [DATA_W-1:0]          m_rdata,
  input  wire [ID_W-1:0]            m_rid,
  input  wire                       m_rlast,
  input  wire [1:0]                 m_rresp
);

  // ---- Interface instances ----
  VX_cp_axi_m_if #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W)) fetch_if ();
  VX_cp_axi_m_if #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W)) cmpl_if  ();
  VX_cp_axi_m_if #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W)) xbar_if  ();

  // Source 0 = fetch, source 1 = completion. The xbar's TID-prefix
  // routing uses high $clog2(2) = 1 bit, so fetch's TID_PREFIX must
  // resolve to source ID 0 and completion's to source ID 1. The xbar
  // sets the high bit on egress and inspects it on R/B for routing.
  // The sources can leave the high bit alone; only the low bits are
  // their per-source sub-tag.

  // ---- Pack source array for the xbar (verilator needs an unpacked-
  //      array port; we wrap our two named interfaces into an array). ----
  // SystemVerilog interface arrays in module ports are awkward with verilator
  // when array elements are named separately; use an interface-array decl
  // and assign with always_comb.
  VX_cp_axi_m_if #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W)) src_arr [2] ();

  // Wire fetch_if <-> src_arr[0]
  assign src_arr[0].awvalid = fetch_if.awvalid;
  assign src_arr[0].awaddr  = fetch_if.awaddr;
  assign src_arr[0].awid    = fetch_if.awid;
  assign src_arr[0].awlen   = fetch_if.awlen;
  assign src_arr[0].awsize  = fetch_if.awsize;
  assign src_arr[0].awburst = fetch_if.awburst;
  assign fetch_if.awready   = src_arr[0].awready;
  assign src_arr[0].wvalid  = fetch_if.wvalid;
  assign src_arr[0].wdata   = fetch_if.wdata;
  assign src_arr[0].wstrb   = fetch_if.wstrb;
  assign src_arr[0].wlast   = fetch_if.wlast;
  assign fetch_if.wready    = src_arr[0].wready;
  assign fetch_if.bvalid    = src_arr[0].bvalid;
  assign fetch_if.bid       = src_arr[0].bid;
  assign fetch_if.bresp     = src_arr[0].bresp;
  assign src_arr[0].bready  = fetch_if.bready;
  assign src_arr[0].arvalid = fetch_if.arvalid;
  assign src_arr[0].araddr  = fetch_if.araddr;
  assign src_arr[0].arid    = fetch_if.arid;
  assign src_arr[0].arlen   = fetch_if.arlen;
  assign src_arr[0].arsize  = fetch_if.arsize;
  assign src_arr[0].arburst = fetch_if.arburst;
  assign fetch_if.arready   = src_arr[0].arready;
  assign fetch_if.rvalid    = src_arr[0].rvalid;
  assign fetch_if.rdata     = src_arr[0].rdata;
  assign fetch_if.rid       = src_arr[0].rid;
  assign fetch_if.rlast     = src_arr[0].rlast;
  assign fetch_if.rresp     = src_arr[0].rresp;
  assign src_arr[0].rready  = fetch_if.rready;

  // Wire cmpl_if <-> src_arr[1] (mirror).
  assign src_arr[1].awvalid = cmpl_if.awvalid;
  assign src_arr[1].awaddr  = cmpl_if.awaddr;
  assign src_arr[1].awid    = cmpl_if.awid;
  assign src_arr[1].awlen   = cmpl_if.awlen;
  assign src_arr[1].awsize  = cmpl_if.awsize;
  assign src_arr[1].awburst = cmpl_if.awburst;
  assign cmpl_if.awready    = src_arr[1].awready;
  assign src_arr[1].wvalid  = cmpl_if.wvalid;
  assign src_arr[1].wdata   = cmpl_if.wdata;
  assign src_arr[1].wstrb   = cmpl_if.wstrb;
  assign src_arr[1].wlast   = cmpl_if.wlast;
  assign cmpl_if.wready     = src_arr[1].wready;
  assign cmpl_if.bvalid     = src_arr[1].bvalid;
  assign cmpl_if.bid        = src_arr[1].bid;
  assign cmpl_if.bresp      = src_arr[1].bresp;
  assign src_arr[1].bready  = cmpl_if.bready;
  assign src_arr[1].arvalid = cmpl_if.arvalid;
  assign src_arr[1].araddr  = cmpl_if.araddr;
  assign src_arr[1].arid    = cmpl_if.arid;
  assign src_arr[1].arlen   = cmpl_if.arlen;
  assign src_arr[1].arsize  = cmpl_if.arsize;
  assign src_arr[1].arburst = cmpl_if.arburst;
  assign cmpl_if.arready    = src_arr[1].arready;
  assign cmpl_if.rvalid     = src_arr[1].rvalid;
  assign cmpl_if.rdata      = src_arr[1].rdata;
  assign cmpl_if.rid        = src_arr[1].rid;
  assign cmpl_if.rlast      = src_arr[1].rlast;
  assign cmpl_if.rresp      = src_arr[1].rresp;
  assign src_arr[1].rready  = cmpl_if.rready;

  // ---- Wire upstream xbar_if to flat ports ----
  assign m_awvalid = xbar_if.awvalid;
  assign xbar_if.awready = m_awready;
  assign m_awaddr  = xbar_if.awaddr;
  assign m_awid    = xbar_if.awid;
  assign m_awlen   = xbar_if.awlen;
  assign m_awsize  = xbar_if.awsize;
  assign m_awburst = xbar_if.awburst;
  assign m_wvalid  = xbar_if.wvalid;
  assign xbar_if.wready = m_wready;
  assign m_wdata   = xbar_if.wdata;
  assign m_wstrb   = xbar_if.wstrb;
  assign m_wlast   = xbar_if.wlast;
  assign xbar_if.bvalid = m_bvalid;
  assign m_bready  = xbar_if.bready;
  assign xbar_if.bid    = m_bid;
  assign xbar_if.bresp  = m_bresp;
  assign m_arvalid = xbar_if.arvalid;
  assign xbar_if.arready = m_arready;
  assign m_araddr  = xbar_if.araddr;
  assign m_arid    = xbar_if.arid;
  assign m_arlen   = xbar_if.arlen;
  assign m_arsize  = xbar_if.arsize;
  assign m_arburst = xbar_if.arburst;
  assign xbar_if.rvalid = m_rvalid;
  assign m_rready  = xbar_if.rready;
  assign xbar_if.rdata  = m_rdata;
  assign xbar_if.rid    = m_rid;
  assign xbar_if.rlast  = m_rlast;
  assign xbar_if.rresp  = m_rresp;

  // ---- DUT instances ----
  cpe_state_t state_typed;
  assign state_typed = cpe_state_t'(state_in_packed);

  cmd_t cmd_typed;
  assign cmd_out_packed = cmd_typed;

  VX_cp_fetch #(.QID(0)) u_fetch (
    .clk           (clk),
    .reset         (reset),
    .state_in      (state_typed),
    .head_out      (head_out),
    .cmd_out_valid (cmd_out_valid),
    .cmd_out       (cmd_typed),
    .cmd_out_ready (cmd_out_ready),
    .axi_m         (fetch_if)
  );

  // Pack retire signals into arrays for completion.
  wire        retire_evt_arr    [1];
  wire [63:0] retire_seqnum_arr [1];
  wire [63:0] cmpl_addr_arr     [1];
  assign retire_evt_arr[0]    = retire_evt;
  assign retire_seqnum_arr[0] = retire_seqnum;
  assign cmpl_addr_arr[0]     = cmpl_addr;

  // retire_ready back-pressure from completion to engine — not driven
  // back to the testbench harness, so just sink it.
  wire retire_ready_arr [1];
  `UNUSED_VAR (retire_ready_arr[0])

  VX_cp_completion #(.NUM_QUEUES(1)) u_cmpl (
    .clk            (clk),
    .reset          (reset),
    .retire_evt     (retire_evt_arr),
    .retire_seqnum  (retire_seqnum_arr),
    .cmpl_addr      (cmpl_addr_arr),
    .retire_ready   (retire_ready_arr),
    .axi_m          (cmpl_if)
  );

  VX_cp_axi_xbar #(.N_SOURCES(2)) u_xbar (
    .clk   (clk),
    .reset (reset),
    .src   (src_arr),
    .axi_m (xbar_if)
  );

endmodule : VX_cp_axi_path_top
