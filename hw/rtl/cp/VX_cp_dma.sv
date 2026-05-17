// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_dma — generic DMA engine for CMD_MEM_READ / CMD_MEM_WRITE /
// CMD_MEM_COPY. Owned by the DMA resource arbiter (parent §6.4 / RTL
// impl §10).
//
// Command encoding (parent §6.5):
//   arg0 = dst address (device or host AXI address)
//   arg1 = src address (device or host AXI address)
//   arg2 = size in bytes (must be 64 in v1)
//
// All three opcodes resolve to the same hardware behavior — issue an
// AXI read at src, capture the data into an internal CL buffer, then
// issue an AXI write at dst. CMD_MEM_READ / CMD_MEM_WRITE differ from
// CMD_MEM_COPY only in *which* address is host- vs device-resident;
// the CP itself doesn't care.
//
// v1 limitations (documented):
//   - Single-cache-line transfers only (size must equal CL_BYTES = 64).
//     Multi-CL chunking comes in a follow-up; the runtime side already
//     splits enqueue_copy larger than this into multiple commands.
//   - Read-modify-write hazard: arg0 and arg1 must not overlap. (The
//     runtime layer enforces this.)
//
// FSM:
//   S_IDLE     : grant ↑ → latch cmd, → S_REQ_AR
//   S_REQ_AR   : drive AR at src; on arready → S_WAIT_R
//   S_WAIT_R   : capture rdata into buf_r; on rvalid+rlast → S_REQ_AW
//   S_REQ_AW   : drive AW at dst; on awready → S_REQ_W
//   S_REQ_W    : drive W from buf_r with wlast; on wready → S_WAIT_B
//   S_WAIT_B   : on bvalid → S_DONE
//   S_DONE     : pulse `done` for one cycle → S_IDLE
// ============================================================================

module VX_cp_dma
  import VX_cp_pkg::*;
#(
  parameter int ID_W = VX_CP_AXI_TID_WIDTH_C,
  parameter logic [ID_W-1:0] TID_PREFIX = '0
)(
  input  wire                       clk,
  input  wire                       reset,

  input  wire                       grant,
  // cmd is wider than what DMA actually reads; suppress the upstream
  // (engine forwards the whole cmd_t to every resource consumer).
  /* verilator lint_off UNUSED */
  input  cmd_t                      cmd,
  /* verilator lint_on UNUSED */
  output logic                      done,

  VX_cp_axi_m_if.master             axi_m
);

  // ---- FSM + state ----
  typedef enum logic [2:0] {
    S_IDLE, S_REQ_AR, S_WAIT_R, S_REQ_AW, S_REQ_W, S_WAIT_B, S_DONE
  } state_e;

  state_e            state;
  logic [63:0]       dst_r, src_r;
  logic [CL_BITS-1:0] buf_r;

  always_ff @(posedge clk) begin
    if (reset) begin
      state <= S_IDLE;
      dst_r <= '0;
      src_r <= '0;
      buf_r <= '0;
    end else begin
      case (state)
        S_IDLE: begin
          if (grant) begin
            dst_r <= cmd.arg0;
            src_r <= cmd.arg1;
            state <= S_REQ_AR;
          end
        end
        S_REQ_AR: begin
          if (axi_m.arvalid && axi_m.arready) state <= S_WAIT_R;
        end
        S_WAIT_R: begin
          if (axi_m.rvalid && axi_m.rready) begin
            buf_r <= axi_m.rdata;
            state <= S_REQ_AW;
          end
        end
        S_REQ_AW: begin
          if (axi_m.awvalid && axi_m.awready) state <= S_REQ_W;
        end
        S_REQ_W: begin
          if (axi_m.wvalid && axi_m.wready) state <= S_WAIT_B;
        end
        S_WAIT_B: begin
          if (axi_m.bvalid && axi_m.bready) state <= S_DONE;
        end
        S_DONE: begin
          state <= S_IDLE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end

  // ---- Output drivers ----
  always_comb begin
    // AR
    axi_m.arvalid = (state == S_REQ_AR);
    axi_m.araddr  = src_r;
    axi_m.arid    = TID_PREFIX;
    axi_m.arlen   = 8'd0;          // single beat (one cache line)
    axi_m.arsize  = 3'd6;          // 64 bytes per transfer
    axi_m.arburst = 2'b01;
    axi_m.rready  = (state == S_WAIT_R);

    // AW
    axi_m.awvalid = (state == S_REQ_AW);
    axi_m.awaddr  = dst_r;
    axi_m.awid    = TID_PREFIX;
    axi_m.awlen   = 8'd0;
    axi_m.awsize  = 3'd6;
    axi_m.awburst = 2'b01;

    // W
    axi_m.wvalid = (state == S_REQ_W);
    axi_m.wdata  = buf_r;
    axi_m.wstrb  = '1;             // full-line write
    axi_m.wlast  = 1'b1;

    // B
    axi_m.bready = (state == S_WAIT_B);

    // Done pulse
    done = (state == S_DONE);
  end

  // Sanity / unused.
  `UNUSED_VAR (axi_m.bid)
  `UNUSED_VAR (axi_m.bresp)
  `UNUSED_VAR (axi_m.rid)
  `UNUSED_VAR (axi_m.rlast)
  `UNUSED_VAR (axi_m.rresp)

endmodule : VX_cp_dma
