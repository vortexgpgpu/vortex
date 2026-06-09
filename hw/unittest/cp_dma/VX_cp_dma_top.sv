// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_dma_top — verilator-friendly wrapper around VX_cp_dma.
//
// VX_cp_dma is a dual-port engine: it owns one AXI master to host memory
// (axi_host) and one to device memory (axi_dev). This wrapper instantiates
// both interface bundles and flattens each to a scalar port set the C++
// harness drives:
//   - h_* : host-memory AXI master
//   - d_* : device-memory AXI master
// ============================================================================

module VX_cp_dma_top
  import VX_cp_pkg::*;
#(
  parameter int ADDR_W = 64,
  parameter int DATA_W = 512,
  parameter int ID_W   = VX_CP_AXI_TID_WIDTH_C
)(
  input  wire                       clk,
  input  wire                       reset,

  input  wire                       grant,
  input  wire [$bits(cmd_t)-1:0]    cmd_packed,
  output wire                       done,

  // ---- Host-memory AXI master (flat) ----
  output wire                       h_awvalid,
  input  wire                       h_awready,
  output wire [ADDR_W-1:0]          h_awaddr,
  output wire [ID_W-1:0]            h_awid,
  output wire [7:0]                 h_awlen,
  output wire [2:0]                 h_awsize,
  output wire [1:0]                 h_awburst,

  output wire                       h_wvalid,
  input  wire                       h_wready,
  output wire [DATA_W-1:0]          h_wdata,
  output wire [DATA_W/8-1:0]        h_wstrb,
  output wire                       h_wlast,

  input  wire                       h_bvalid,
  output wire                       h_bready,
  input  wire [ID_W-1:0]            h_bid,
  input  wire [1:0]                 h_bresp,

  output wire                       h_arvalid,
  input  wire                       h_arready,
  output wire [ADDR_W-1:0]          h_araddr,
  output wire [ID_W-1:0]            h_arid,
  output wire [7:0]                 h_arlen,
  output wire [2:0]                 h_arsize,
  output wire [1:0]                 h_arburst,

  input  wire                       h_rvalid,
  output wire                       h_rready,
  input  wire [DATA_W-1:0]          h_rdata,
  input  wire [ID_W-1:0]            h_rid,
  input  wire                       h_rlast,
  input  wire [1:0]                 h_rresp,

  // ---- Device-memory AXI master (flat) ----
  output wire                       d_awvalid,
  input  wire                       d_awready,
  output wire [ADDR_W-1:0]          d_awaddr,
  output wire [ID_W-1:0]            d_awid,
  output wire [7:0]                 d_awlen,
  output wire [2:0]                 d_awsize,
  output wire [1:0]                 d_awburst,

  output wire                       d_wvalid,
  input  wire                       d_wready,
  output wire [DATA_W-1:0]          d_wdata,
  output wire [DATA_W/8-1:0]        d_wstrb,
  output wire                       d_wlast,

  input  wire                       d_bvalid,
  output wire                       d_bready,
  input  wire [ID_W-1:0]            d_bid,
  input  wire [1:0]                 d_bresp,

  output wire                       d_arvalid,
  input  wire                       d_arready,
  output wire [ADDR_W-1:0]          d_araddr,
  output wire [ID_W-1:0]            d_arid,
  output wire [7:0]                 d_arlen,
  output wire [2:0]                 d_arsize,
  output wire [1:0]                 d_arburst,

  input  wire                       d_rvalid,
  output wire                       d_rready,
  input  wire [DATA_W-1:0]          d_rdata,
  input  wire [ID_W-1:0]            d_rid,
  input  wire                       d_rlast,
  input  wire [1:0]                 d_rresp
);

  VX_cp_axi_m_if #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W)) axi_host ();
  VX_cp_axi_m_if #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W)) axi_dev  ();

  // ---- Host AXI flat pass-through ----
  assign h_awvalid        = axi_host.awvalid;
  assign axi_host.awready = h_awready;
  assign h_awaddr         = axi_host.awaddr;
  assign h_awid           = axi_host.awid;
  assign h_awlen          = axi_host.awlen;
  assign h_awsize         = axi_host.awsize;
  assign h_awburst        = axi_host.awburst;

  assign h_wvalid         = axi_host.wvalid;
  assign axi_host.wready  = h_wready;
  assign h_wdata          = axi_host.wdata;
  assign h_wstrb          = axi_host.wstrb;
  assign h_wlast          = axi_host.wlast;

  assign axi_host.bvalid  = h_bvalid;
  assign h_bready         = axi_host.bready;
  assign axi_host.bid     = h_bid;
  assign axi_host.bresp   = h_bresp;

  assign h_arvalid        = axi_host.arvalid;
  assign axi_host.arready = h_arready;
  assign h_araddr         = axi_host.araddr;
  assign h_arid           = axi_host.arid;
  assign h_arlen          = axi_host.arlen;
  assign h_arsize         = axi_host.arsize;
  assign h_arburst        = axi_host.arburst;

  assign axi_host.rvalid  = h_rvalid;
  assign h_rready         = axi_host.rready;
  assign axi_host.rdata   = h_rdata;
  assign axi_host.rid     = h_rid;
  assign axi_host.rlast   = h_rlast;
  assign axi_host.rresp   = h_rresp;

  // ---- Device AXI flat pass-through ----
  assign d_awvalid        = axi_dev.awvalid;
  assign axi_dev.awready  = d_awready;
  assign d_awaddr         = axi_dev.awaddr;
  assign d_awid           = axi_dev.awid;
  assign d_awlen          = axi_dev.awlen;
  assign d_awsize         = axi_dev.awsize;
  assign d_awburst        = axi_dev.awburst;

  assign d_wvalid         = axi_dev.wvalid;
  assign axi_dev.wready   = d_wready;
  assign d_wdata          = axi_dev.wdata;
  assign d_wstrb          = axi_dev.wstrb;
  assign d_wlast          = axi_dev.wlast;

  assign axi_dev.bvalid   = d_bvalid;
  assign d_bready         = axi_dev.bready;
  assign axi_dev.bid      = d_bid;
  assign axi_dev.bresp    = d_bresp;

  assign d_arvalid        = axi_dev.arvalid;
  assign axi_dev.arready  = d_arready;
  assign d_araddr         = axi_dev.araddr;
  assign d_arid           = axi_dev.arid;
  assign d_arlen          = axi_dev.arlen;
  assign d_arsize         = axi_dev.arsize;
  assign d_arburst        = axi_dev.arburst;

  assign axi_dev.rvalid   = d_rvalid;
  assign d_rready         = axi_dev.rready;
  assign axi_dev.rdata    = d_rdata;
  assign axi_dev.rid      = d_rid;
  assign axi_dev.rlast    = d_rlast;
  assign axi_dev.rresp    = d_rresp;

  cmd_t cmd_typed;
  assign cmd_typed = cmd_t'(cmd_packed);

  VX_cp_dma u_dut (
    .clk      (clk),
    .reset    (reset),
    .grant    (grant),
    .cmd      (cmd_typed),
    .done     (done),
    .axi_host (axi_host),
    .axi_dev  (axi_dev)
  );

endmodule : VX_cp_dma_top
