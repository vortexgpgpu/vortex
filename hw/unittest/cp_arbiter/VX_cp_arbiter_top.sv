// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_arbiter_top — verilator-friendly wrapper around VX_cp_arbiter.
//
// The arbiter module ports use unpacked arrays (`wire bid_valid [N]`) which
// are awkward to drive from Verilator C++ harnesses. This wrapper exposes a
// fixed N=4 instance with packed-bus ports the harness can read/write as
// plain scalars.
// ============================================================================

module VX_cp_arbiter_top
  import VX_cp_pkg::*;
#(
  parameter int N = 4
)(
  input  wire             clk,
  input  wire             reset,

  input  wire [N-1:0]     bid_valid,        // packed: bit i = bidder i valid
  input  wire [2*N-1:0]   bid_priority,     // packed: 2 bits per bidder
  output wire [N-1:0]     bid_grant         // packed: bit i = bidder i granted
);

  // Unpacked arrays for the DUT.
  wire        in_valid [N];
  wire [1:0]  in_prio  [N];
  logic       out_grant[N];

  generate
    for (genvar i = 0; i < N; ++i) begin : g_unpack
      assign in_valid[i] = bid_valid[i];
      assign in_prio[i]  = bid_priority[2*i +: 2];
      assign bid_grant[i] = out_grant[i];
    end
  endgenerate

  VX_cp_arbiter #(.N(N)) u_arb (
    .clk          (clk),
    .reset        (reset),
    .bid_valid    (in_valid),
    .bid_priority (in_prio),
    .bid_grant    (out_grant)
  );

endmodule : VX_cp_arbiter_top
