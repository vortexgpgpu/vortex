// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_core_top — verilator-friendly wrapper around VX_cp_core.
//
// Exposes all three interfaces (AXI-Lite slave, AXI4 master, gpu_if) as
// flat scalar ports so the C++ harness can drive the host control
// plane, act as the upstream AXI memory, and simulate the Vortex
// start/busy + DCR handshake.
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

  // ---- AXI4 master (data plane upstream) ----
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
  VX_cp_axi_m_if  #(.ADDR_W(ADDR_W), .DATA_W(DATA_W), .ID_W(ID_W)) axi_m_if ();
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

  // AXI master passthrough.
  assign m_awvalid       = axi_m_if.awvalid;
  assign axi_m_if.awready = m_awready;
  assign m_awaddr        = axi_m_if.awaddr;
  assign m_awid          = axi_m_if.awid;
  assign m_awlen         = axi_m_if.awlen;
  assign m_awsize        = axi_m_if.awsize;
  assign m_awburst       = axi_m_if.awburst;
  assign m_wvalid        = axi_m_if.wvalid;
  assign axi_m_if.wready = m_wready;
  assign m_wdata         = axi_m_if.wdata;
  assign m_wstrb         = axi_m_if.wstrb;
  assign m_wlast         = axi_m_if.wlast;
  assign axi_m_if.bvalid = m_bvalid;
  assign m_bready        = axi_m_if.bready;
  assign axi_m_if.bid    = m_bid;
  assign axi_m_if.bresp  = m_bresp;
  assign m_arvalid       = axi_m_if.arvalid;
  assign axi_m_if.arready = m_arready;
  assign m_araddr        = axi_m_if.araddr;
  assign m_arid          = axi_m_if.arid;
  assign m_arlen         = axi_m_if.arlen;
  assign m_arsize        = axi_m_if.arsize;
  assign m_arburst       = axi_m_if.arburst;
  assign axi_m_if.rvalid = m_rvalid;
  assign m_rready        = axi_m_if.rready;
  assign axi_m_if.rdata  = m_rdata;
  assign axi_m_if.rid    = m_rid;
  assign axi_m_if.rlast  = m_rlast;
  assign axi_m_if.rresp  = m_rresp;

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
    .axi_m     (axi_m_if),
    .gpu_if    (gpu_if_inst),
    .interrupt (interrupt)
  );

  // Debug taps — read q_state from the inner regfile hierarchically.
  // Cross-module references resolve at elaboration time.
  assign dbg_q0_enabled = u_dut.q_state[0].enabled;
  assign dbg_q0_tail    = u_dut.q_state[0].tail;

endmodule : VX_cp_core_top
