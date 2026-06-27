// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_axi_slice — AXI4 register slice for VX_cp_axi_m_if.
//
// Inserts a full valid/ready register stage (VX_skid_buffer, OUT_REG) on every
// AXI channel between an upstream master `s` and the downstream port `m`.
// Placed at the VX_cp_core axi_host/axi_dev boundary to break the long route
// from the Command Processor to the HBM / memory-subsystem AXI: without it the
// AW/AR/W signals launch combinationally from the CP DMA registers across a
// die-spanning, routing-dominated path. The CP DMA issues one outstanding
// burst at a time, so the extra channel latency is throughput-neutral.
// ============================================================================

module VX_cp_axi_slice
  import VX_cp_pkg::*;
#(
  parameter int ADDR_W = 64,
  parameter int DATA_W = 512,
  parameter int ID_W   = VX_CP_AXI_TID_WIDTH_C
)(
  input  wire           clk,
  input  wire           reset,
  VX_cp_axi_m_if.slave  s,   // from upstream master (xbar)
  VX_cp_axi_m_if.master m    // to downstream slave  (cp_core boundary)
);
  localparam int AWR_W = ADDR_W + ID_W + 8 + 3 + 2;  // addr,id,len,size,burst
  localparam int W_W   = DATA_W + (DATA_W/8) + 1;     // data,strb,last
  localparam int B_W   = ID_W + 2;                    // id,resp
  localparam int R_W   = DATA_W + ID_W + 1 + 2;       // data,id,last,resp

  // ---- AW : s -> m ----
  VX_skid_buffer #(
    .DATAW   (AWR_W),
    .OUT_REG (1)
  ) aw_slice (
    .clk       (clk),
    .reset     (reset),
    .valid_in  (s.awvalid),
    .ready_in  (s.awready),
    .data_in   ({s.awaddr, s.awid, s.awlen, s.awsize, s.awburst}),
    .valid_out (m.awvalid),
    .ready_out (m.awready),
    .data_out  ({m.awaddr, m.awid, m.awlen, m.awsize, m.awburst})
  );

  // ---- W : s -> m ----
  VX_skid_buffer #(
    .DATAW   (W_W),
    .OUT_REG (1)
  ) w_slice (
    .clk       (clk),
    .reset     (reset),
    .valid_in  (s.wvalid),
    .ready_in  (s.wready),
    .data_in   ({s.wdata, s.wstrb, s.wlast}),
    .valid_out (m.wvalid),
    .ready_out (m.wready),
    .data_out  ({m.wdata, m.wstrb, m.wlast})
  );

  // ---- B : m -> s ----
  VX_skid_buffer #(
    .DATAW   (B_W),
    .OUT_REG (1)
  ) b_slice (
    .clk       (clk),
    .reset     (reset),
    .valid_in  (m.bvalid),
    .ready_in  (m.bready),
    .data_in   ({m.bid, m.bresp}),
    .valid_out (s.bvalid),
    .ready_out (s.bready),
    .data_out  ({s.bid, s.bresp})
  );

  // ---- AR : s -> m ----
  VX_skid_buffer #(
    .DATAW   (AWR_W),
    .OUT_REG (1)
  ) ar_slice (
    .clk       (clk),
    .reset     (reset),
    .valid_in  (s.arvalid),
    .ready_in  (s.arready),
    .data_in   ({s.araddr, s.arid, s.arlen, s.arsize, s.arburst}),
    .valid_out (m.arvalid),
    .ready_out (m.arready),
    .data_out  ({m.araddr, m.arid, m.arlen, m.arsize, m.arburst})
  );

  // ---- R : m -> s ----
  VX_skid_buffer #(
    .DATAW   (R_W),
    .OUT_REG (1)
  ) r_slice (
    .clk       (clk),
    .reset     (reset),
    .valid_in  (m.rvalid),
    .ready_in  (m.rready),
    .data_in   ({m.rdata, m.rid, m.rlast, m.rresp}),
    .valid_out (s.rvalid),
    .ready_out (s.rready),
    .data_out  ({s.rdata, s.rid, s.rlast, s.rresp})
  );

endmodule : VX_cp_axi_slice
