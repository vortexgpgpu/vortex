// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_fetch — per-CPE ring-buffer fetcher.
//
// One instance per VX_cp_engine. Reads 64 B cache lines from the host-
// pinned ring buffer over an AXI4 master sub-port (the per-CPE input to
// VX_cp_axi_xbar), decodes them with an embedded VX_cp_unpack, and streams
// the decoded cmd_t records one at a time to its CPE's cmd_in port.
//
// FSM:
//   S_IDLE       : head < tail → S_ISSUE_AR
//                  head == tail → wait (host hasn't published more)
//   S_ISSUE_AR   : drive AR with addr = ring_base + (head & mask),
//                  arlen=0 (single 64 B beat), arsize=6, arburst=INCR
//                  → S_WAIT_R on arready
//   S_WAIT_R     : wait for rvalid; latch rdata into cl_data_r
//                  → S_EMIT on rvalid && rlast
//   S_EMIT       : present cmds[slot]; on cmd_out_ready advance slot.
//                  When slot == cmd_count - 1: head += 64, → S_IDLE
//                  Pure-padding lines (cmd_count == 0) skip directly to
//                  head advance + IDLE.
//
// Issues a single-beat 512 b AR (one cache line) per ring transaction.
// The ring is `1 << ring_size_log2` bytes; head/tail are byte offsets
// that wrap via ring_size_mask. Tail is monotonic from the host's
// perspective; this fetcher does not watch for wraparound.
// ============================================================================

module VX_cp_fetch
  import VX_cp_pkg::*;
#(
  parameter int  QID    = 0,
  parameter int  ID_W   = VX_CP_AXI_TID_WIDTH_C,
  // The xbar packs source ID into the high bits of arid. Caller assigns
  // a unique TID_PREFIX per fetch instance so responses route back.
  parameter logic [ID_W-1:0] TID_PREFIX = '0
)(
  input  wire                       clk,
  input  wire                       reset,

  // Per-CPE state mirror from the regfile.
  input  cpe_state_t                state_in,
  // Updated head pointer — the regfile / CPE-state mirror tracks this
  // for the host to read back.
  output logic [63:0]               head_out,

  // Decoded command stream out to the CPE.
  output logic                      cmd_out_valid,
  output cmd_t                      cmd_out,
  input  wire                       cmd_out_ready,

  // AXI4 master sub-port (one of the sources on VX_cp_axi_xbar).
  VX_cp_axi_m_if.master             axi_m
);

  // ---- Internal head register (byte offset, monotonic) ----
  logic [63:0] head_r;
  assign head_out = head_r;

  // ---- Latched cache line + sequential single-command decode ----
  logic [CL_BITS-1:0]  cl_data_r;
  localparam int       OFF_W = $clog2(CL_BYTES + 1);
  logic [OFF_W-1:0]    offset_r;     // byte offset of the command being emitted
  cmd_t                cmd_w;        // decoded command at offset_r
  logic                has_cmd_w;    // 1 = valid command at offset_r (0 = end-of-line)
  logic [OFF_W-1:0]    cmd_size_w;   // bytes consumed by cmd_w

  // Decode exactly the command at the current offset. The FSM walks the line
  // by advancing offset_r += cmd_size_w after each emit, instead of decoding
  // the whole line combinationally (which was a 35-level critical path).
  VX_cp_unpack u_unpack (
    .cl_data   (cl_data_r),
    .offset    (offset_r),
    .has_cmd   (has_cmd_w),
    .cmd       (cmd_w),
    .cmd_size  (cmd_size_w)
  );

  typedef enum logic [1:0] { S_IDLE, S_ISSUE_AR, S_WAIT_R, S_EMIT } state_e;
  state_e state;

  // Wrap-aware ring offset.
  wire [63:0] ring_offset = head_r & {48'd0, state_in.ring_size_mask};

  always_ff @(posedge clk) begin
    if (reset) begin
      state     <= S_IDLE;
      head_r    <= '0;
      cl_data_r <= '0;
      offset_r  <= '0;
    end else begin
      case (state)
        S_IDLE: begin
          if (state_in.enabled && (head_r < state_in.tail)) begin
            state <= S_ISSUE_AR;
          end
        end
        S_ISSUE_AR: begin
          if (axi_m.arvalid && axi_m.arready) begin
            state <= S_WAIT_R;
          end
        end
        S_WAIT_R: begin
          if (axi_m.rvalid && axi_m.rready) begin
            cl_data_r <= axi_m.rdata;
            offset_r  <= '0;
            state     <= S_EMIT;
          end
        end
        S_EMIT: begin
          // Decode-and-emit one command per cycle. has_cmd_w==0 means the
          // line is exhausted (zero-header padding, no room for a header, or
          // we have walked past the last command) → advance head, next line.
          if (!has_cmd_w) begin
            head_r <= head_r + 64'd64;
            state  <= S_IDLE;
          end else if (cmd_out_ready) begin
            offset_r <= offset_r + cmd_size_w;
          end
        end
        default: state <= S_IDLE;
      endcase
    end
  end

  // ---- Output drivers ----
  always_comb begin
    // AXI master defaults. fetch only uses AR/R; AW/W/B are tied off.
    axi_m.awvalid = 1'b0;
    axi_m.awaddr  = '0;
    axi_m.awid    = '0;
    axi_m.awlen   = '0;
    axi_m.awsize  = '0;
    axi_m.awburst = 2'b01;
    axi_m.wvalid  = 1'b0;
    axi_m.wdata   = '0;
    axi_m.wstrb   = '0;
    axi_m.wlast   = 1'b0;
    axi_m.bready  = 1'b1;
    axi_m.rready  = (state == S_WAIT_R);

    // AR drive
    axi_m.arvalid = (state == S_ISSUE_AR);
    axi_m.araddr  = state_in.ring_base + ring_offset;
    axi_m.arid    = TID_PREFIX;
    axi_m.arlen   = 8'd0;                  // single beat
    axi_m.arsize  = 3'd6;                  // 64 bytes per transfer
    axi_m.arburst = 2'b01;                 // INCR

    // Command output
    cmd_out_valid = (state == S_EMIT) && has_cmd_w;
    cmd_out       = cmd_w;
  end

  // Sanity / unused.
  `UNUSED_VAR (axi_m.bvalid)
  `UNUSED_VAR (axi_m.bid)
  `UNUSED_VAR (axi_m.bresp)
  `UNUSED_VAR (axi_m.awready)
  `UNUSED_VAR (axi_m.wready)
  `UNUSED_VAR (axi_m.rid)
  `UNUSED_VAR (axi_m.rlast)
  `UNUSED_VAR (axi_m.rresp)
  `UNUSED_VAR (state_in.head_addr)
  `UNUSED_VAR (state_in.cmpl_addr)
  `UNUSED_VAR (state_in.head)
  `UNUSED_VAR (state_in.seqnum)
  `UNUSED_VAR (state_in.prio)
  `UNUSED_VAR (state_in.profile_en)
  `UNUSED_PARAM (QID)

endmodule : VX_cp_fetch
