// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`ifndef VX_CP_AXIL_S_IF_SV
`define VX_CP_AXIL_S_IF_SV

`include "VX_define.vh"

// ============================================================================
// VX_cp_axil_s_if.sv — AXI4-Lite slave interface bundle used inside
// rtl/cp/. The host's control plane drives this; VX_cp_axil_regfile is
// the only slave inside the CP.
//
// AXI4-Lite has no burst, ID, or last signals — just AW/W/B and AR/R
// with 32-bit data and a byte enable. Single-beat per transaction.
// ============================================================================

interface VX_cp_axil_s_if
#(
  parameter int ADDR_W = 16,    // 64 KiB control space
  parameter int DATA_W = 32
);

  // ---- AW ----
  logic              awvalid;
  logic              awready;
  logic [ADDR_W-1:0] awaddr;

  // ---- W ----
  logic              wvalid;
  logic              wready;
  logic [DATA_W-1:0] wdata;
  logic [DATA_W/8-1:0] wstrb;

  // ---- B ----
  logic              bvalid;
  logic              bready;
  logic [1:0]        bresp;     // 2'b00 OKAY, 2'b11 DECERR

  // ---- AR ----
  logic              arvalid;
  logic              arready;
  logic [ADDR_W-1:0] araddr;

  // ---- R ----
  logic              rvalid;
  logic              rready;
  logic [DATA_W-1:0] rdata;
  logic [1:0]        rresp;

  // Slave-side: receives requests, produces responses.
  modport slave (
    input  awvalid, awaddr,
    output awready,
    input  wvalid, wdata, wstrb,
    output wready,
    output bvalid, bresp,
    input  bready,
    input  arvalid, araddr,
    output arready,
    output rvalid, rdata, rresp,
    input  rready
  );

  // Master-side: drives requests, receives responses. Useful for
  // test harnesses that emulate the host.
  modport master (
    output awvalid, awaddr,
    input  awready,
    output wvalid, wdata, wstrb,
    input  wready,
    input  bvalid, bresp,
    output bready,
    output arvalid, araddr,
    input  arready,
    input  rvalid, rdata, rresp,
    output rready
  );

endinterface : VX_cp_axil_s_if

`endif // VX_CP_AXIL_S_IF_SV
