// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_event_unit — implements CMD_EVENT_WAIT. Reads the 8 B value at
// event_addr via the CP's AXI master, compares to expected under the wait
// op (EQ/GE/GT/NE), and signals the requesting CPE when the comparison
// succeeds. A small LRU cache reduces AXI traffic when multiple CPEs spin
// on the same slot.
//
// Stub — `rsp_match` is tied low; the engine currently retires
// CMD_EVENT_WAIT as a NOP.
// ============================================================================

module VX_cp_event_unit
  import VX_cp_pkg::*;
(
  input  wire clk,
  input  wire reset,

  input  wire           req_valid,
  input  wire [63:0]    req_addr,
  input  wire [63:0]    req_value,
  input  wait_op_e      req_op,
  output logic          rsp_match
);

  assign rsp_match = 1'b0;

  `UNUSED_VAR (clk)
  `UNUSED_VAR (reset)
  `UNUSED_VAR (req_valid)
  `UNUSED_VAR (req_addr)
  `UNUSED_VAR (req_value)
  `UNUSED_VAR (req_op)

endmodule : VX_cp_event_unit
