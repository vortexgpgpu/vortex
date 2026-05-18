// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_dma_top — verilator-friendly wrapper around VX_cp_dma.
//
// Exposes the AXI4 master channels as flat scalar ports; cmd_t input
// as a packed bus.
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

  // AXI master flat ports
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

  VX_cp_axi_m_if #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W)) axi_if ();

  // Pass-through wiring.
  assign m_awvalid       = axi_if.awvalid;
  assign axi_if.awready  = m_awready;
  assign m_awaddr        = axi_if.awaddr;
  assign m_awid          = axi_if.awid;
  assign m_awlen         = axi_if.awlen;
  assign m_awsize        = axi_if.awsize;
  assign m_awburst       = axi_if.awburst;

  assign m_wvalid        = axi_if.wvalid;
  assign axi_if.wready   = m_wready;
  assign m_wdata         = axi_if.wdata;
  assign m_wstrb         = axi_if.wstrb;
  assign m_wlast         = axi_if.wlast;

  assign axi_if.bvalid   = m_bvalid;
  assign m_bready        = axi_if.bready;
  assign axi_if.bid      = m_bid;
  assign axi_if.bresp    = m_bresp;

  assign m_arvalid       = axi_if.arvalid;
  assign axi_if.arready  = m_arready;
  assign m_araddr        = axi_if.araddr;
  assign m_arid          = axi_if.arid;
  assign m_arlen         = axi_if.arlen;
  assign m_arsize        = axi_if.arsize;
  assign m_arburst       = axi_if.arburst;

  assign axi_if.rvalid   = m_rvalid;
  assign m_rready        = axi_if.rready;
  assign axi_if.rdata    = m_rdata;
  assign axi_if.rid      = m_rid;
  assign axi_if.rlast    = m_rlast;
  assign axi_if.rresp    = m_rresp;

  cmd_t cmd_typed;
  assign cmd_typed = cmd_t'(cmd_packed);

  VX_cp_dma u_dut (
    .clk   (clk),
    .reset (reset),
    .grant (grant),
    .cmd   (cmd_typed),
    .done  (done),
    .axi_m (axi_if)
  );

endmodule : VX_cp_dma_top
