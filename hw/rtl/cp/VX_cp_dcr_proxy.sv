// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_dcr_proxy — DCR request/response gateway between the CP and Vortex.
// Owned by the DCR resource arbiter (parent §6.4 / RTL impl §11).
//
// For CMD_DCR_WRITE (cmd.arg0 = dcr_addr, cmd.arg1 = dcr_value):
//   IDLE → REQ_WRITE (drive dcr_req with rw=1 until ready) → DONE → IDLE.
//
// For CMD_DCR_READ (cmd.arg0 = dcr_addr, cmd.arg1 = host_writeback_addr):
//   IDLE → REQ_READ (drive dcr_req with rw=0 until ready) → WAIT_RSP
//        (latch dcr_rsp_data when valid) → WRITEBACK_HOST → DONE → IDLE.
//
// The WRITEBACK_HOST step requires the AXI master and is deferred to
// the next commit; for now CMD_DCR_READ completes after WAIT_RSP and
// publishes the read value on `last_rsp_data` for the engine to capture.
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
  logic [`VX_DCR_DATA_BITS-1:0] rsp_data_r;

  // Extract address / data / rw from cmd. CMD_DCR_WRITE: arg1 = value;
  // CMD_DCR_READ: arg1 = host_writeback_addr (not driven on the DCR bus).
  wire                          is_read    = (cmd.hdr.opcode == 8'(CMD_DCR_READ));
  wire [`VX_DCR_ADDR_BITS-1:0]  cmd_addr   = cmd.arg0[`VX_DCR_ADDR_BITS-1:0];
  wire [`VX_DCR_DATA_BITS-1:0]  cmd_data   = cmd.arg1[`VX_DCR_DATA_BITS-1:0];

  always_ff @(posedge clk) begin
    if (reset) begin
      state           <= S_IDLE;
      pending_is_read <= 1'b0;
      rsp_data_r      <= '0;
    end else begin
      case (state)
        S_IDLE: begin
          if (grant) begin
            state           <= S_REQ;
            pending_is_read <= is_read;
          end
        end
        S_REQ: begin
          // In this DCR bus model the request is consumed in one cycle
          // (req_valid handshakes with the Vortex DCR arbiter combinationally;
          // there is no req_ready backpressure in v1).
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
    dcr_req_rw    = !is_read;
    dcr_req_addr  = cmd_addr;
    dcr_req_data  = cmd_data;
    done          = (state == S_DONE);
    last_rsp_data = rsp_data_r;
  end

endmodule : VX_cp_dcr_proxy
