// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_axil_regfile_top — verilator-friendly wrapper.
//
// Exposes the AXI4-Lite slave channels as flat scalar ports so the C++
// harness can drive transactions directly. Per-queue telemetry inputs
// (q_head / q_seqnum / q_error) are flattened to packed buses; q_state
// output is similarly flattened.
//
// Tied to NUM_QUEUES=1 to keep the harness simple — the regfile RTL is
// generic but the multi-queue case can be exercised in a future TB.
// ============================================================================

module VX_cp_axil_regfile_top
  import VX_cp_pkg::*;
#(
  parameter int NUM_QUEUES = 1,
  parameter int ADDR_W     = 16
)(
  input  wire                            clk,
  input  wire                            reset,

  // AXI-Lite W/AW/B
  input  wire                            awvalid,
  output wire                            awready,
  input  wire [ADDR_W-1:0]               awaddr,
  input  wire                            wvalid,
  output wire                            wready,
  input  wire [31:0]                     wdata,
  input  wire [3:0]                      wstrb,
  output wire                            bvalid,
  input  wire                            bready,
  output wire [1:0]                      bresp,

  // AXI-Lite AR/R
  input  wire                            arvalid,
  output wire                            arready,
  input  wire [ADDR_W-1:0]               araddr,
  output wire                            rvalid,
  input  wire                            rready,
  output wire [31:0]                     rdata,
  output wire [1:0]                      rresp,

  // Status inputs (driven by harness)
  input  wire                            cp_busy,
  input  wire                            cp_error,
  input  wire [NUM_QUEUES*64-1:0]        q_head_packed,
  input  wire [NUM_QUEUES*64-1:0]        q_seqnum_packed,
  input  wire [NUM_QUEUES*32-1:0]        q_error_packed,

  // q_state outputs (flattened) + reset pulses
  output wire [NUM_QUEUES*$bits(cpe_state_t)-1:0] q_state_packed,
  output wire [NUM_QUEUES-1:0]                     q_reset_pulse
);

  VX_cp_axil_s_if #(.ADDR_W(ADDR_W)) s_if ();

  // Drive the interface from flat ports.
  assign s_if.awvalid = awvalid;
  assign awready      = s_if.awready;
  assign s_if.awaddr  = awaddr;

  assign s_if.wvalid  = wvalid;
  assign wready       = s_if.wready;
  assign s_if.wdata   = wdata;
  assign s_if.wstrb   = wstrb;

  assign bvalid       = s_if.bvalid;
  assign s_if.bready  = bready;
  assign bresp        = s_if.bresp;

  assign s_if.arvalid = arvalid;
  assign arready      = s_if.arready;
  assign s_if.araddr  = araddr;

  assign rvalid       = s_if.rvalid;
  assign s_if.rready  = rready;
  assign rdata        = s_if.rdata;
  assign rresp        = s_if.rresp;

  // Unpack telemetry buses into per-queue arrays for the regfile.
  wire [63:0] q_head_arr   [NUM_QUEUES];
  wire [63:0] q_seqnum_arr [NUM_QUEUES];
  wire [31:0] q_error_arr  [NUM_QUEUES];
  cpe_state_t q_state_arr  [NUM_QUEUES];
  logic       q_reset_arr  [NUM_QUEUES];

  generate
    for (genvar i = 0; i < NUM_QUEUES; ++i) begin : g_pack
      assign q_head_arr  [i] = q_head_packed  [i*64 +: 64];
      assign q_seqnum_arr[i] = q_seqnum_packed[i*64 +: 64];
      assign q_error_arr [i] = q_error_packed [i*32 +: 32];
      assign q_state_packed[i*$bits(cpe_state_t) +: $bits(cpe_state_t)] = q_state_arr[i];
      assign q_reset_pulse[i] = q_reset_arr[i];
    end
  endgenerate

  VX_cp_axil_regfile #(.NUM_QUEUES(NUM_QUEUES), .ADDR_W(ADDR_W)) u_dut (
    .clk            (clk),
    .reset          (reset),
    .axil_s         (s_if),
    .cp_busy        (cp_busy),
    .cp_error       (cp_error),
    .q_head         (q_head_arr),
    .q_seqnum       (q_seqnum_arr),
    .q_error        (q_error_arr),
    .last_dcr_rsp   (32'd0),
    .q_state        (q_state_arr),
    .q_reset_pulse  (q_reset_arr)
  );

endmodule : VX_cp_axil_regfile_top
