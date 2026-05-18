// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_dcr_proxy — DCR request/response gateway between the CP and Vortex.
// Owned by the DCR resource arbiter.
//
// For CMD_DCR_WRITE (cmd.arg0 = dcr_addr, cmd.arg1 = dcr_value):
//   IDLE → REQ (drive dcr_req with rw=1) → DONE → IDLE.
//
// For CMD_DCR_READ (cmd.arg0 = dcr_addr):
//   IDLE → REQ (drive dcr_req with rw=0) → WAIT_RSP (latch dcr_rsp_data
//        when valid) → DONE → IDLE.
//
// The most-recent read response is published on `last_rsp_data` and is
// also exposed on the AXI-Lite regfile so the host can poll it after
// observing the seqnum advance.
// ============================================================================

module VX_cp_dcr_proxy
  import VX_cp_pkg::*;
(
  input  wire clk,
  input  wire reset,

  input  wire  grant,
  // verilator lint_off UNUSED
  // Only cmd.hdr.opcode, cmd.arg0, and cmd.arg1 are read here. arg2 and
  // profile_slot pass through untouched on the way to the engine; the
  // top-level instantiation hands us the full struct.
  input  cmd_t cmd,
  // verilator lint_on UNUSED
  output logic done,

  // Most recent CMD_DCR_READ response value (valid while `done` is high
  // after a read; tied to 0 after writes). Engine snapshots this when it
  // observes done for a read command.
  output logic [`VX_DCR_DATA_BITS-1:0] last_rsp_data,

  // Vortex DCR port (driven through VX_cp_gpu_if by VX_cp_core).
  output logic                         dcr_req_valid,
  output logic                         dcr_req_rw,
  output logic [`VX_DCR_ADDR_BITS-1:0] dcr_req_addr,
  output logic [`VX_DCR_DATA_BITS-1:0] dcr_req_data,
  input  wire                          dcr_rsp_valid,
  input  wire  [`VX_DCR_DATA_BITS-1:0] dcr_rsp_data
);

  typedef enum logic [1:0] {
    S_IDLE,
    S_REQ,           // hold dcr_req_valid until consumed (single cycle here)
    S_WAIT_RSP,      // read commands only
    S_DONE
  } state_e;

  state_e state;
  logic   pending_is_read;
  // The full DCR payload is latched on grant: granted_dcr_cmd is a
  // combinational mux gated on the arbiter's grant pulse, which drops
  // the cycle after, so any downstream state that consumes cmd fields
  // must capture them on the same edge as the IDLE → REQ transition.
  logic [`VX_DCR_ADDR_BITS-1:0]  pending_addr;
  logic [`VX_DCR_DATA_BITS-1:0]  pending_data;
  logic [`VX_DCR_DATA_BITS-1:0]  rsp_data_r;

  wire                          is_read    = (cmd.hdr.opcode == 8'(CMD_DCR_READ));
  wire [`VX_DCR_ADDR_BITS-1:0]  cmd_addr   = cmd.arg0[`VX_DCR_ADDR_BITS-1:0];
  wire [`VX_DCR_DATA_BITS-1:0]  cmd_data   = cmd.arg1[`VX_DCR_DATA_BITS-1:0];

  always_ff @(posedge clk) begin
    if (reset) begin
      state           <= S_IDLE;
      pending_is_read <= 1'b0;
      pending_addr    <= '0;
      pending_data    <= '0;
      rsp_data_r      <= '0;
    end else begin
      case (state)
        S_IDLE: begin
          if (grant) begin
            state           <= S_REQ;
            pending_is_read <= is_read;
            pending_addr    <= cmd_addr;
            pending_data    <= cmd_data;
          end
        end
        S_REQ: begin
          // The Vortex DCR bus consumes the request in a single cycle
          // (req_valid handshakes combinationally; no req_ready backpressure).
          if (pending_is_read)
            state <= S_WAIT_RSP;
          else
            state <= S_DONE;
        end
        S_WAIT_RSP: begin
          if (dcr_rsp_valid) begin
            rsp_data_r <= dcr_rsp_data;
            state      <= S_DONE;
          end
        end
        S_DONE: begin
          state <= S_IDLE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end

  always_comb begin
    dcr_req_valid = (state == S_REQ);
    dcr_req_rw    = !pending_is_read;
    dcr_req_addr  = pending_addr;
    dcr_req_data  = pending_data;
    done          = (state == S_DONE);
    last_rsp_data = rsp_data_r;
  end

endmodule : VX_cp_dcr_proxy
