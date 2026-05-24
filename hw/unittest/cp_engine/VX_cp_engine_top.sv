// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_engine_top — verilator-friendly wrapper around VX_cp_engine.
//
// VX_cp_engine talks to the four resource arbiters through SystemVerilog
// interfaces, which can't be driven directly from C++ harnesses. This
// wrapper instantiates the four bid interfaces locally, exposes them as
// flat packed ports the harness reads/writes, and connects them through
// modports to the engine.
//
// The CPE state mirror is reduced to a single `state_prio` input — the
// only queue-state field the engine FSM consumes (it tags the arbiter
// bids). The engine's `seqnum_out` telemetry output is left unobserved.
// ============================================================================

module VX_cp_engine_top
  import VX_cp_pkg::*;
(
  input  wire        clk,
  input  wire        reset,

  // CPE state mirror — only `prio` matters to the engine's bid lines.
  input  wire [1:0]  state_prio,

  // Command stream input (packed cmd_t).
  input  wire                          cmd_in_valid,
  input  wire [$bits(cmd_t)-1:0]       cmd_in_packed,
  output wire                          cmd_in_ready,

  // Per-resource bid lines (flat).
  output wire                          bid_kmu_valid,
  output wire [1:0]                    bid_kmu_prio,
  output wire [$bits(cmd_t)-1:0]       bid_kmu_cmd,
  input  wire                          bid_kmu_grant,

  output wire                          bid_dma_valid,
  output wire [1:0]                    bid_dma_prio,
  output wire [$bits(cmd_t)-1:0]       bid_dma_cmd,
  input  wire                          bid_dma_grant,

  output wire                          bid_dcr_valid,
  output wire [1:0]                    bid_dcr_prio,
  output wire [$bits(cmd_t)-1:0]       bid_dcr_cmd,
  input  wire                          bid_dcr_grant,

  output wire                          bid_event_valid,
  output wire [1:0]                    bid_event_prio,
  output wire [$bits(cmd_t)-1:0]       bid_event_cmd,
  input  wire                          bid_event_grant,

  // Resource done pulses (harness drives these to simulate the resource
  // modules finishing). For backwards-compatible tests that still treat
  // grant as done, the harness can simply tie these to the corresponding
  // bid_*_grant inputs delayed by one cycle.
  input  wire                          kmu_done_i,
  input  wire                          dma_done_i,
  input  wire                          dcr_done_i,
  input  wire                          event_done_i,

  // Retirement.
  output wire                          retire_evt,
  output wire [63:0]                   retire_seqnum,

  // Profiling pulses.
  output wire                          submit_evt,
  output wire                          start_evt,
  output wire                          end_evt,
  output wire [63:0]                   profile_slot
);

  // ---- Wrap cmd_in_packed back into cmd_t for the engine ----------------
  cmd_t cmd_in_typed;
  assign cmd_in_typed = cmd_t'(cmd_in_packed);

  // ---- Engine retired-seqnum telemetry (unobserved by the harness) ------
  /* verilator lint_off UNUSED */
  wire [63:0] seqnum_out_w;
  /* verilator lint_on UNUSED */

  // ---- Bid interfaces ---------------------------------------------------
  VX_cp_engine_bid_if bid_kmu_if   ();
  VX_cp_engine_bid_if bid_dma_if   ();
  VX_cp_engine_bid_if bid_dcr_if   ();
  VX_cp_engine_bid_if bid_event_if ();

  // Drive engine grants from the harness, surface engine outputs to harness.
  assign bid_kmu_if.grant   = bid_kmu_grant;
  assign bid_dma_if.grant   = bid_dma_grant;
  assign bid_dcr_if.grant   = bid_dcr_grant;
  assign bid_event_if.grant = bid_event_grant;

  assign bid_kmu_valid = bid_kmu_if.valid;
  assign bid_kmu_prio  = bid_kmu_if.priority_;
  assign bid_kmu_cmd   = bid_kmu_if.cmd;

  assign bid_dma_valid = bid_dma_if.valid;
  assign bid_dma_prio  = bid_dma_if.priority_;
  assign bid_dma_cmd   = bid_dma_if.cmd;

  assign bid_dcr_valid = bid_dcr_if.valid;
  assign bid_dcr_prio  = bid_dcr_if.priority_;
  assign bid_dcr_cmd   = bid_dcr_if.cmd;

  assign bid_event_valid = bid_event_if.valid;
  assign bid_event_prio  = bid_event_if.priority_;
  assign bid_event_cmd   = bid_event_if.cmd;

  // ---- DUT --------------------------------------------------------------
  logic cmd_in_ready_w;
  assign cmd_in_ready = cmd_in_ready_w;

  VX_cp_engine #(.QID(0)) u_engine (
    .clk           (clk),
    .reset         (reset),
    .prio_in       (state_prio),
    .seqnum_out    (seqnum_out_w),
    .cmd_in_valid  (cmd_in_valid),
    .cmd_in        (cmd_in_typed),
    .cmd_in_ready  (cmd_in_ready_w),
    .bid_kmu       (bid_kmu_if),
    .bid_dma       (bid_dma_if),
    .bid_dcr       (bid_dcr_if),
    .bid_event     (bid_event_if),
    .kmu_done_i    (kmu_done_i),
    .dma_done_i    (dma_done_i),
    .dcr_done_i    (dcr_done_i),
    .event_done_i  (event_done_i),
    .retire_evt    (retire_evt),
    .retire_seqnum (retire_seqnum),
    .retire_ready_i(1'b1),                // unit-test: completion is always ready
    .submit_evt    (submit_evt),
    .start_evt     (start_evt),
    .end_evt       (end_evt),
    .profile_slot  (profile_slot)
  );

endmodule : VX_cp_engine_top
