// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_core_top — verilator-friendly wrapper around VX_cp_core.
//
// Exposes every interface as flat scalar ports so the C++ harness can
// drive the host control plane, act as the upstream AXI memories, and
// simulate the Vortex start/busy + DCR handshake. VX_cp_core has two
// AXI4 data-plane masters:
//   - m_* : host-memory master  (command ring + completion + DMA host side)
//   - d_* : device-memory master (DMA device side + event-counter traffic)
// ============================================================================

module VX_cp_core_top
  import VX_cp_pkg::*;
#(
  parameter int NUM_QUEUES = 1,
  parameter int ADDR_W     = 64,
  parameter int DATA_W     = 512,
  parameter int ID_W       = VX_CP_AXI_TID_WIDTH_C,
  parameter int AXIL_AW    = 16
)(
  input  wire                       clk,
  input  wire                       reset,

  // ---- AXI-Lite slave (host control) ----
  input  wire                       s_awvalid,
  output wire                       s_awready,
  input  wire [AXIL_AW-1:0]         s_awaddr,
  input  wire                       s_wvalid,
  output wire                       s_wready,
  input  wire [31:0]                s_wdata,
  input  wire [3:0]                 s_wstrb,
  output wire                       s_bvalid,
  input  wire                       s_bready,
  output wire [1:0]                 s_bresp,
  input  wire                       s_arvalid,
  output wire                       s_arready,
  input  wire [AXIL_AW-1:0]         s_araddr,
  output wire                       s_rvalid,
  input  wire                       s_rready,
  output wire [31:0]                s_rdata,
  output wire [1:0]                 s_rresp,

  // ---- AXI4 host-memory master (m_*) ----
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
  input  wire [1:0]                 m_rresp,

  // ---- AXI4 device-memory master (d_*) ----
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
  input  wire [1:0]                 d_rresp,

  // ---- GPU interface (Vortex DCR + start/busy) ----
  output wire                       gpu_dcr_req_valid,
  output wire                       gpu_dcr_req_rw,
  output wire [`VX_DCR_ADDR_BITS-1:0] gpu_dcr_req_addr,
  output wire [`VX_DCR_DATA_BITS-1:0] gpu_dcr_req_data,
  input  wire                       gpu_dcr_req_ready,
  input  wire                       gpu_dcr_rsp_valid,
  input  wire [`VX_DCR_DATA_BITS-1:0] gpu_dcr_rsp_data,
  output wire                       gpu_start,
  input  wire                       gpu_busy,

  // ---- Interrupt ----
  /* verilator lint_off SYMRSVDWORD */
  output wire                       interrupt,
  /* verilator lint_on SYMRSVDWORD */

  // ---- Debug taps into the inner regfile state for the TB ----
  output wire                       dbg_q0_enabled,
  output wire [63:0]                dbg_q0_tail
);

  VX_cp_axil_s_if #(.ADDR_W(AXIL_AW)) axil_s_if ();
  VX_cp_axi_m_if  #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W)) axi_host_if ();
  VX_cp_axi_m_if  #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W)) axi_dev_if  ();
  VX_cp_gpu_if    gpu_if_inst ();

  // AXI-Lite slave passthrough.
  assign axil_s_if.awvalid = s_awvalid;
  assign s_awready         = axil_s_if.awready;
  assign axil_s_if.awaddr  = s_awaddr;
  assign axil_s_if.wvalid  = s_wvalid;
  assign s_wready          = axil_s_if.wready;
  assign axil_s_if.wdata   = s_wdata;
  assign axil_s_if.wstrb   = s_wstrb;
  assign s_bvalid          = axil_s_if.bvalid;
  assign axil_s_if.bready  = s_bready;
  assign s_bresp           = axil_s_if.bresp;
  assign axil_s_if.arvalid = s_arvalid;
  assign s_arready         = axil_s_if.arready;
  assign axil_s_if.araddr  = s_araddr;
  assign s_rvalid          = axil_s_if.rvalid;
  assign axil_s_if.rready  = s_rready;
  assign s_rdata           = axil_s_if.rdata;
  assign s_rresp           = axil_s_if.rresp;

  // AXI host-memory master passthrough.
  assign m_awvalid          = axi_host_if.awvalid;
  assign axi_host_if.awready = m_awready;
  assign m_awaddr           = axi_host_if.awaddr;
  assign m_awid             = axi_host_if.awid;
  assign m_awlen            = axi_host_if.awlen;
  assign m_awsize           = axi_host_if.awsize;
  assign m_awburst          = axi_host_if.awburst;
  assign m_wvalid           = axi_host_if.wvalid;
  assign axi_host_if.wready  = m_wready;
  assign m_wdata            = axi_host_if.wdata;
  assign m_wstrb            = axi_host_if.wstrb;
  assign m_wlast            = axi_host_if.wlast;
  assign axi_host_if.bvalid  = m_bvalid;
  assign m_bready           = axi_host_if.bready;
  assign axi_host_if.bid     = m_bid;
  assign axi_host_if.bresp   = m_bresp;
  assign m_arvalid          = axi_host_if.arvalid;
  assign axi_host_if.arready = m_arready;
  assign m_araddr           = axi_host_if.araddr;
  assign m_arid             = axi_host_if.arid;
  assign m_arlen            = axi_host_if.arlen;
  assign m_arsize           = axi_host_if.arsize;
  assign m_arburst          = axi_host_if.arburst;
  assign axi_host_if.rvalid  = m_rvalid;
  assign m_rready           = axi_host_if.rready;
  assign axi_host_if.rdata   = m_rdata;
  assign axi_host_if.rid     = m_rid;
  assign axi_host_if.rlast   = m_rlast;
  assign axi_host_if.rresp   = m_rresp;

  // AXI device-memory master passthrough.
  assign d_awvalid          = axi_dev_if.awvalid;
  assign axi_dev_if.awready  = d_awready;
  assign d_awaddr           = axi_dev_if.awaddr;
  assign d_awid             = axi_dev_if.awid;
  assign d_awlen            = axi_dev_if.awlen;
  assign d_awsize           = axi_dev_if.awsize;
  assign d_awburst          = axi_dev_if.awburst;
  assign d_wvalid           = axi_dev_if.wvalid;
  assign axi_dev_if.wready   = d_wready;
  assign d_wdata            = axi_dev_if.wdata;
  assign d_wstrb            = axi_dev_if.wstrb;
  assign d_wlast            = axi_dev_if.wlast;
  assign axi_dev_if.bvalid   = d_bvalid;
  assign d_bready           = axi_dev_if.bready;
  assign axi_dev_if.bid      = d_bid;
  assign axi_dev_if.bresp    = d_bresp;
  assign d_arvalid          = axi_dev_if.arvalid;
  assign axi_dev_if.arready  = d_arready;
  assign d_araddr           = axi_dev_if.araddr;
  assign d_arid             = axi_dev_if.arid;
  assign d_arlen            = axi_dev_if.arlen;
  assign d_arsize           = axi_dev_if.arsize;
  assign d_arburst          = axi_dev_if.arburst;
  assign axi_dev_if.rvalid   = d_rvalid;
  assign d_rready           = axi_dev_if.rready;
  assign axi_dev_if.rdata    = d_rdata;
  assign axi_dev_if.rid      = d_rid;
  assign axi_dev_if.rlast    = d_rlast;
  assign axi_dev_if.rresp    = d_rresp;

  // gpu_if passthrough.
  assign gpu_dcr_req_valid = gpu_if_inst.dcr_req_valid;
  assign gpu_dcr_req_rw    = gpu_if_inst.dcr_req_rw;
  assign gpu_dcr_req_addr  = gpu_if_inst.dcr_req_addr;
  assign gpu_dcr_req_data  = gpu_if_inst.dcr_req_data;
  assign gpu_if_inst.dcr_req_ready = gpu_dcr_req_ready;
  assign gpu_if_inst.dcr_rsp_valid = gpu_dcr_rsp_valid;
  assign gpu_if_inst.dcr_rsp_data  = gpu_dcr_rsp_data;
  assign gpu_start         = gpu_if_inst.start;
  assign gpu_if_inst.busy  = gpu_busy;

  VX_cp_core #(
    .NUM_QUEUES (NUM_QUEUES),
    .ADDR_W     (ADDR_W),
    .DATA_W     (DATA_W),
    .ID_W       (ID_W),
    .AXIL_AW    (AXIL_AW)
  ) u_dut (
    .clk       (clk),
    .reset     (reset),
    .axil_s    (axil_s_if),
    .axi_host  (axi_host_if),
    .axi_dev   (axi_dev_if),
    .gpu_if    (gpu_if_inst),
    .interrupt (interrupt)
  );

  // Debug taps — read q_state from the inner regfile hierarchically.
  // Cross-module references resolve at elaboration time.
  assign dbg_q0_enabled = u_dut.q_state[0].enabled;
  assign dbg_q0_tail    = u_dut.q_state[0].tail;

endmodule : VX_cp_core_top
