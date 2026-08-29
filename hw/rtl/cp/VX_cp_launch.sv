// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_launch — KMU start/busy wrapper. Owned by the KMU resource arbiter.
//
// KMU arbitration holds for the entire duration of a launch:
//   IDLE         : no grant yet
//   PULSE_START  : grant just observed; assert `start` for one cycle
//   WAIT_BUSY    : Vortex pulls `busy` high (kernel started)
//   WAIT_DRAIN   : Vortex drops `busy` low (kernel done) → fire `done`,
//                  go back to IDLE
//
// The CPE that won the KMU arbiter holds its bid across all of these
// states; `done` releasing the bid lets the next CPE take its turn.
//
// `grant` is the OR of per-CPE grants from the KMU arbiter (the CP core
// glues all N bids onto this single input).
// ============================================================================

module VX_cp_launch (
  input  wire  clk,
  input  wire  reset,

  input  wire  grant,         // OR of per-CPE grants from KMU arbiter
  output logic start,         // pulsed to gpu_if.start (Vortex)
  input  wire  gpu_busy,      // from gpu_if.busy (Vortex)
  output logic done           // back to engine: launch fully drained
);

  typedef enum logic [1:0] {
    S_IDLE,
    S_PULSE_START,
    S_WAIT_BUSY,
    S_WAIT_DRAIN
  } state_e;

  state_e state;

  always_ff @(posedge clk) begin
    if (reset) begin
      state <= S_IDLE;
    end else begin
      case (state)
        S_IDLE: begin
          if (grant) state <= S_PULSE_START;
        end
        S_PULSE_START: begin
          state <= S_WAIT_BUSY;
        end
        S_WAIT_BUSY: begin
          // Vortex's busy might rise the next cycle after `start` fires;
          // we wait for that rising edge.
          if (gpu_busy) state <= S_WAIT_DRAIN;
        end
        S_WAIT_DRAIN: begin
          if (!gpu_busy) state <= S_IDLE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end

  always_comb begin
    start = (state == S_PULSE_START);
    done  = (state == S_WAIT_DRAIN) && !gpu_busy;
  end

endmodule : VX_cp_launch
