// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_profiling — free-running 64-bit cycle counter + per-command 32 B
// timestamp writeback. The cycle counter is exposed to the host via the
// AXI-Lite slave register block at CP_CYCLE_LO/HI.
//
// The writeback path (per-CPE timestamp FIFO → AXI master) is not yet
// implemented; the engine fires the submit/start/end pulses today but
// they are consumed only by this counter.
// ============================================================================

module VX_cp_profiling
  import VX_cp_pkg::*;
#(
  parameter int NUM_QUEUES = VX_CP_NUM_QUEUES_C
)(
  input  wire        clk,
  input  wire        reset,

  // RO output exposed via AXI-Lite (CP_CYCLE_LO/HI at 0x040/0x044).
  output logic [63:0] cp_cycle,

  // Per-CPE sample pulses + the slot address to write back to.
  input  wire         submit_evt [NUM_QUEUES],
  input  wire         start_evt  [NUM_QUEUES],
  input  wire         end_evt    [NUM_QUEUES],
  input  wire [63:0]  slot_addr  [NUM_QUEUES]
);

  // Free-running cycle counter.
  always_ff @(posedge clk) begin
    if (reset)
      cp_cycle <= '0;
    else
      cp_cycle <= cp_cycle + 64'd1;
  end

  // Future work: per-CPE timestamp FIFO; on end_evt, pop and write
  // {queued_ns=0, submit_ts, start_ts, end_ts} (32 B) to slot_addr.
  `UNUSED_VAR (submit_evt)
  `UNUSED_VAR (start_evt)
  `UNUSED_VAR (end_evt)
  `UNUSED_VAR (slot_addr)

endmodule : VX_cp_profiling
