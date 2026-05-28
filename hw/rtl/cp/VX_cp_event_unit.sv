// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_event_unit — handles CMD_EVENT_SIGNAL and CMD_EVENT_WAIT.
//
// Owned by the EVENT resource arbiter. One instance per CP_core.
//
// Command encoding (cmd_t fields):
//   arg0 = 64-bit byte address of the event counter slot in host memory
//   arg1 = 64-bit value (for SIGNAL: write this; for WAIT: target value)
//   arg2 = bits [1:0] = wait_op_e  (WAIT_OP_EQ / GE / GT / NE)
//          remaining bits reserved
//
// FSM:
//   S_IDLE     : grant ↑ → latch cmd + opcode, → S_REQ_AW (SIGNAL)
//                                              or S_REQ_AR (WAIT)
//
//   ---- SIGNAL path ----
//   S_REQ_AW   : drive AW at arg0, awsize=3 (8 B beat); awready → S_REQ_W
//   S_REQ_W    : drive W with arg1 in low 8 B (wstrb selects bytes 0..7);
//                wready → S_WAIT_B
//   S_WAIT_B   : bvalid → S_DONE
//
//   ---- WAIT path ----
//   S_REQ_AR   : drive AR at arg0, arsize=3; arready → S_WAIT_R
//   S_WAIT_R   : rvalid → capture rdata low 8 B; compare to arg1 under
//                wait_op:  EQ  match if read == arg1
//                          GE  match if read >= arg1
//                          GT  match if read >  arg1
//                          NE  match if read != arg1
//                match → S_DONE
//                miss  → S_REQ_AR (spin until satisfied; no backoff in v1
//                        — the round-trip itself rate-limits)
//
//   S_DONE     : pulse `done` for one cycle → S_IDLE
//
// This unit holds the EVENT bid grant for the entire wait duration. The
// arbiter is round-robin across CPEs, so other queues' WAITs interleave
// fairly when several spin concurrently.
// ============================================================================

module VX_cp_event_unit
  import VX_cp_pkg::*;
#(
  parameter int ID_W = VX_CP_AXI_TID_WIDTH_C,
  parameter logic [ID_W-1:0] TID_PREFIX = '0
)(
  input  wire                       clk,
  input  wire                       reset,

  input  wire                       grant,
  // cmd carries arg0/arg1/arg2 plus header (which we read for opcode);
  // remaining fields are forwarded but unused by this unit.
  input  cmd_t                      cmd,
  output logic                      done,

  VX_cp_axi_m_if.master             axi_m
);

  // cmd fields not consumed by this unit (opcode/arg0/arg1/arg2[1:0] are read above).
  `UNUSED_VAR (cmd.hdr.reserved)
  `UNUSED_VAR (cmd.hdr.flags)
  `UNUSED_VAR (cmd.arg2[63:2])
  `UNUSED_VAR (cmd.profile_slot)

  // ---- FSM ----
  typedef enum logic [3:0] {
    S_IDLE, S_REQ_AW, S_REQ_W, S_WAIT_B,
            S_REQ_AR, S_WAIT_R, S_DONE
  } state_e;

  state_e          state;
  logic [63:0]     addr_r;
  logic [63:0]     value_r;      // SIGNAL: value to write; WAIT: target
  wait_op_e        wait_op_r;
  logic            is_signal_r;

  // ---- Combinational compare for WAIT ----
  logic [63:0] rdata_lo;
  assign rdata_lo = axi_m.rdata[63:0];

  logic match;
  always_comb begin
    match = 1'b0;
    case (wait_op_r)
      WAIT_OP_EQ: match = (rdata_lo == value_r);
      WAIT_OP_GE: match = (rdata_lo >= value_r);
      WAIT_OP_GT: match = (rdata_lo >  value_r);
      WAIT_OP_NE: match = (rdata_lo != value_r);
      default:    match = 1'b0;
    endcase
  end

  // ---- State transitions ----
  always_ff @(posedge clk) begin
    if (reset) begin
      state       <= S_IDLE;
      addr_r      <= '0;
      value_r     <= '0;
      wait_op_r   <= WAIT_OP_EQ;
      is_signal_r <= 1'b0;
    end else begin
      case (state)
        S_IDLE: begin
          if (grant) begin
            addr_r      <= cmd.arg0;
            value_r     <= cmd.arg1;
            wait_op_r   <= wait_op_e'(cmd.arg2[1:0]);
            is_signal_r <= (cmd.hdr.opcode == CMD_EVENT_SIGNAL);
            state       <= (cmd.hdr.opcode == CMD_EVENT_SIGNAL)
                             ? S_REQ_AW : S_REQ_AR;
          end
        end

        // SIGNAL path
        S_REQ_AW: if (axi_m.awvalid && axi_m.awready) state <= S_REQ_W;
        S_REQ_W:  if (axi_m.wvalid  && axi_m.wready)  state <= S_WAIT_B;
        S_WAIT_B: if (axi_m.bvalid  && axi_m.bready)  state <= S_DONE;

        // WAIT path
        S_REQ_AR: if (axi_m.arvalid && axi_m.arready) state <= S_WAIT_R;
        S_WAIT_R: begin
          if (axi_m.rvalid && axi_m.rready) begin
            state <= match ? S_DONE : S_REQ_AR;
          end
        end

        S_DONE: state <= S_IDLE;
        default: state <= S_IDLE;
      endcase
    end
  end

  // ---- AXI master output drivers ----
  always_comb begin
    // ---- AW (SIGNAL) ----
    axi_m.awvalid = (state == S_REQ_AW);
    axi_m.awaddr  = addr_r;
    axi_m.awid    = TID_PREFIX;
    axi_m.awlen   = 8'd0;        // 1 beat
    axi_m.awsize  = 3'd3;        // 2^3 = 8 bytes
    axi_m.awburst = 2'b01;       // INCR

    // ---- W (SIGNAL) ----
    axi_m.wvalid = (state == S_REQ_W);
    axi_m.wdata  = '0;
    axi_m.wdata[63:0] = value_r;
    axi_m.wstrb  = '0;
    axi_m.wstrb[7:0] = 8'hFF;    // bytes 0..7 valid (low 8 B of bus)
    axi_m.wlast  = 1'b1;

    // ---- B (SIGNAL) ----
    axi_m.bready = (state == S_WAIT_B);

    // ---- AR (WAIT) ----
    axi_m.arvalid = (state == S_REQ_AR);
    axi_m.araddr  = addr_r;
    axi_m.arid    = TID_PREFIX;
    axi_m.arlen   = 8'd0;
    axi_m.arsize  = 3'd3;
    axi_m.arburst = 2'b01;

    // ---- R (WAIT) ----
    axi_m.rready = (state == S_WAIT_R);

    // Done pulse
    done = (state == S_DONE);
  end

  // Sanity / unused.
  `UNUSED_VAR (axi_m.bid)
  `UNUSED_VAR (axi_m.bresp)
  `UNUSED_VAR (axi_m.rid)
  `UNUSED_VAR (axi_m.rlast)
  `UNUSED_VAR (axi_m.rresp)
  `UNUSED_VAR (is_signal_r)

endmodule : VX_cp_event_unit
