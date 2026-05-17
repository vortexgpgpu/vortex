// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`ifndef VX_CP_AXI_M_IF_SV
`define VX_CP_AXI_M_IF_SV

`include "VX_define.vh"

// ============================================================================
// VX_cp_axi_m_if.sv — AXI4 master interface bundle used inside rtl/cp/.
//
// Every CP module that needs to issue host-AXI transactions (VX_cp_fetch,
// VX_cp_dma, VX_cp_completion, VX_cp_event_unit, VX_cp_profiling) talks
// through one instance of this interface. VX_cp_axi_xbar fans them into
// the single upstream master that VX_cp_core exposes on its `axi_m` port.
//
// The bundle deliberately omits the optional AW/AR sideband signals
// (LOCK / CACHE / PROT / QOS / REGION) that v1 doesn't drive — they are
// tied off at the cp_core boundary to whatever value the upstream XRT
// shell expects (typically all zero, write-allocate cache attributes).
// ============================================================================

interface VX_cp_axi_m_if
#(
  parameter int ADDR_W = 64,
  parameter int DATA_W = 512,
  parameter int ID_W   = VX_CP_AXI_TID_WIDTH_C
);

  import VX_cp_pkg::*;

  // ---- Write request address channel (AW) ----
  logic              awvalid;
  logic              awready;
  logic [ADDR_W-1:0] awaddr;
  logic [ID_W-1:0]   awid;
  logic [7:0]        awlen;     // number of transfers - 1
  logic [2:0]        awsize;    // log2 bytes per transfer
  logic [1:0]        awburst;   // 2'b01 = INCR

  // ---- Write data channel (W) ----
  logic              wvalid;
  logic              wready;
  logic [DATA_W-1:0] wdata;
  logic [DATA_W/8-1:0] wstrb;
  logic              wlast;

  // ---- Write response channel (B) ----
  logic              bvalid;
  logic              bready;
  logic [ID_W-1:0]   bid;
  logic [1:0]        bresp;     // 2'b00 = OKAY

  // ---- Read request address channel (AR) ----
  logic              arvalid;
  logic              arready;
  logic [ADDR_W-1:0] araddr;
  logic [ID_W-1:0]   arid;
  logic [7:0]        arlen;
  logic [2:0]        arsize;
  logic [1:0]        arburst;

  // ---- Read response channel (R) ----
  logic              rvalid;
  logic              rready;
  logic [DATA_W-1:0] rdata;
  logic [ID_W-1:0]   rid;
  logic              rlast;
  logic [1:0]        rresp;

  // ---- Modports ----
  modport master (
    // AW
    output awvalid, awaddr, awid, awlen, awsize, awburst,
    input  awready,
    // W
    output wvalid, wdata, wstrb, wlast,
    input  wready,
    // B
    input  bvalid, bid, bresp,
    output bready,
    // AR
    output arvalid, araddr, arid, arlen, arsize, arburst,
    input  arready,
    // R
    input  rvalid, rdata, rid, rlast, rresp,
    output rready
  );

  modport slave (
    // AW
    input  awvalid, awaddr, awid, awlen, awsize, awburst,
    output awready,
    // W
    input  wvalid, wdata, wstrb, wlast,
    output wready,
    // B
    output bvalid, bid, bresp,
    input  bready,
    // AR
    input  arvalid, araddr, arid, arlen, arsize, arburst,
    output arready,
    // R
    output rvalid, rdata, rid, rlast, rresp,
    input  rready
  );

endinterface : VX_cp_axi_m_if

`endif // VX_CP_AXI_M_IF_SV
