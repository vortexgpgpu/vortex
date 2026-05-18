// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_launch_top — verilator-friendly wrapper around VX_cp_launch.
//
// VX_cp_launch already has only plain scalar ports, so the wrapper just
// passes them through. It exists for consistency with the other unittest
// targets (each DUT has a *_top.sv harness).
// ============================================================================

module VX_cp_launch_top (
  input  wire  clk,
  input  wire  reset,
  input  wire  grant,
  output wire  start,
  input  wire  gpu_busy,
  output wire  done
);

  VX_cp_launch u_dut (
    .clk      (clk),
    .reset    (reset),
    .grant    (grant),
    .start    (start),
    .gpu_busy (gpu_busy),
    .done     (done)
  );

endmodule : VX_cp_launch_top
