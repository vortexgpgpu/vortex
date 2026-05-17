// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_engine_top — verilator-friendly wrapper around VX_cp_engine.
//
// VX_cp_engine talks to the three resource arbiters through SystemVerilog
// interfaces, which can't be driven directly from C++ harnesses. This
// wrapper instantiates the three bid interfaces locally, exposes them as
// flat packed ports the harness reads/writes, and connects them through
// modports to the engine.
//
// The state_in mirror is reduced to a single `state_prio` input — the
// other cpe_state_t fields aren't read by the engine FSM (they live there
// for the future fetch/unpack path that the engine forwards untouched).
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

  // ---- Synthesize a minimal cpe_state_t with the harness-provided prio --
  cpe_state_t state_in_typed;
  /* verilator lint_off UNUSED */
  cpe_state_t state_out_typed;
  /* verilator lint_on UNUSED */
  always_comb begin
    state_in_typed = '0;
    state_in_typed.prio = state_prio;
  end

  // ---- Bid interfaces ---------------------------------------------------
  VX_cp_engine_bid_if bid_kmu_if ();
  VX_cp_engine_bid_if bid_dma_if ();
  VX_cp_engine_bid_if bid_dcr_if ();

  // Drive engine grants from the harness, surface engine outputs to harness.
  assign bid_kmu_if.grant = bid_kmu_grant;
  assign bid_dma_if.grant = bid_dma_grant;
  assign bid_dcr_if.grant = bid_dcr_grant;

  assign bid_kmu_valid = bid_kmu_if.valid;
  assign bid_kmu_prio  = bid_kmu_if.priority_;
  assign bid_kmu_cmd   = bid_kmu_if.cmd;

  assign bid_dma_valid = bid_dma_if.valid;
  assign bid_dma_prio  = bid_dma_if.priority_;
  assign bid_dma_cmd   = bid_dma_if.cmd;

  assign bid_dcr_valid = bid_dcr_if.valid;
  assign bid_dcr_prio  = bid_dcr_if.priority_;
  assign bid_dcr_cmd   = bid_dcr_if.cmd;

  // ---- DUT --------------------------------------------------------------
  logic cmd_in_ready_w;
  assign cmd_in_ready = cmd_in_ready_w;

  VX_cp_engine #(.QID(0)) u_engine (
    .clk           (clk),
    .reset         (reset),
    .state_in      (state_in_typed),
    .state_out     (state_out_typed),
    .cmd_in_valid  (cmd_in_valid),
    .cmd_in        (cmd_in_typed),
    .cmd_in_ready  (cmd_in_ready_w),
    .bid_kmu       (bid_kmu_if),
    .bid_dma       (bid_dma_if),
    .bid_dcr       (bid_dcr_if),
    .retire_evt    (retire_evt),
    .retire_seqnum (retire_seqnum),
    .submit_evt    (submit_evt),
    .start_evt     (start_evt),
    .end_evt       (end_evt),
    .profile_slot  (profile_slot)
  );

endmodule : VX_cp_engine_top
