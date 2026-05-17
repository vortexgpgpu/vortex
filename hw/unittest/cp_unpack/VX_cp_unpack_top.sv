// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_unpack_top — verilator-friendly wrapper around VX_cp_unpack.
//
// VX_cp_unpack outputs `cmds [MAX_CMDS]` as an unpacked array of `cmd_t`;
// flatten into a single packed bus so the C++ harness can read all the
// decoded fields with a simple index expression.
// ============================================================================

module VX_cp_unpack_top
  import VX_cp_pkg::*;
#(
  parameter int MAX_CMDS = VX_CP_MAX_CMDS_PER_CL_C
)(
  input  wire                              clk,    // tied unused; kept so
  input  wire                              reset,  // wrapper matches the
                                                   // vl_simulator template
  input  wire [CL_BITS-1:0]                cl_data,

  output wire [$clog2(MAX_CMDS+1)-1:0]     cmd_count,
  output wire [MAX_CMDS*$bits(cmd_t)-1:0]  cmds_packed
);

  `UNUSED_VAR (clk)
  `UNUSED_VAR (reset)

  // Unpacked sink for the DUT.
  cmd_t dut_cmds [MAX_CMDS];

  VX_cp_unpack #(.MAX_CMDS(MAX_CMDS)) u_dut (
    .cl_data   (cl_data),
    .cmd_count (cmd_count),
    .cmds      (dut_cmds)
  );

  // Pack the unpacked array into a flat bus, slot 0 in the LSBs.
  generate
    for (genvar i = 0; i < MAX_CMDS; ++i) begin : g_pack
      assign cmds_packed[i*$bits(cmd_t) +: $bits(cmd_t)] = dut_cmds[i];
    end
  endgenerate

endmodule : VX_cp_unpack_top
