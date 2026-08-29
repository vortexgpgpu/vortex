// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_unpack_top — verilator-friendly wrapper around VX_cp_unpack.
//
// VX_cp_unpack decodes exactly ONE cmd_t located at byte `offset` within a
// 64 B cache line (walking the line is the fetch FSM's job, not the
// unpacker's). Expose the single decoded `cmd` as a flat packed bus so the
// C++ harness can read its fields with a simple index expression; the harness
// drives `offset` and advances it by `cmd_size` to walk the line itself.
// ============================================================================

module VX_cp_unpack_top
  import VX_cp_pkg::*;
#(
  // Byte-offset/size domain: 0 .. CL_BYTES — mirrors VX_cp_unpack's OFF_W.
  parameter int OFF_W = $clog2(CL_BYTES + 1)
)(
  input  wire                       clk,    // tied unused; kept so the
  input  wire                       reset,  // wrapper matches the
                                            // vl_simulator template
  input  wire [CL_BITS-1:0]         cl_data,
  input  wire [OFF_W-1:0]           offset,

  output wire                       has_cmd,
  output wire [$bits(cmd_t)-1:0]    cmd_packed,
  output wire [OFF_W-1:0]           cmd_size
);

  `UNUSED_VAR (clk)
  `UNUSED_VAR (reset)

  cmd_t dut_cmd;

  VX_cp_unpack u_dut (
    .cl_data  (cl_data),
    .offset   (offset),
    .has_cmd  (has_cmd),
    .cmd      (dut_cmd),
    .cmd_size (cmd_size)
  );

  // Flatten the single decoded record into a packed bus for the C++ harness.
  assign cmd_packed = dut_cmd;

endmodule : VX_cp_unpack_top
