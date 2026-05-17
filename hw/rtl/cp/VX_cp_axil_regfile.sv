// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_axil_regfile — the CP's AXI4-Lite host-control register block.
//
// Specified in `docs/proposals/cp_runtime_impl_proposal.md §6.10` and
// `cp_rtl_impl_proposal.md §17.4`. This is the *only* slave on the CP's
// AXI-Lite port; VX_cp_core hands its `axil_s` interface here.
//
// Register map (16-bit byte address):
//
//   Global (0x000..0x0FF)
//     0x000 CP_CTRL     RW   bit0=enable_global, bit1=reset_all
//     0x004 CP_STATUS   RO   bit0=busy, bit1=error
//     0x008 CP_DEV_CAPS RO   [7:0]NUM_QUEUES | [15:8]RING_SIZE_LOG2_MAX
//                            [23:16]AXI_TID_WIDTH
//     0x010 CP_CYCLE_LO RO   free-running cycle counter low 32 bits
//     0x014 CP_CYCLE_HI RO   high 32 bits
//
//   Per-queue, base = 0x100 + qid * 0x40
//     +0x00 Q_RING_BASE_LO  RW
//     +0x04 Q_RING_BASE_HI  RW
//     +0x08 Q_HEAD_ADDR_LO  RW
//     +0x0C Q_HEAD_ADDR_HI  RW
//     +0x10 Q_CMPL_ADDR_LO  RW
//     +0x14 Q_CMPL_ADDR_HI  RW
//     +0x18 Q_RING_SIZE_LOG2 RW (mask is derived: (1<<value) - 1)
//     +0x1C Q_CONTROL       RW   bit0=enable, bit1=reset_pulse,
//                                bit[3:2]=prio, bit4=profile_en
//     +0x20 Q_TAIL_LO       WO staging
//     +0x24 Q_TAIL_HI       WO staging + atomic commit pulse
//     +0x28 Q_SEQNUM        RO  latest retired seqnum (mirrors cmpl slot)
//     +0x2C Q_ERROR         RO  per-queue error word
//
// Atomic-tail rule (parent §6.10): the host writes Q_TAIL_LO into a
// staging register *without* advancing q_state.tail, then writes
// Q_TAIL_HI which both stages the high half AND commits the full
// 64-bit value into q_state.tail in the same cycle. A host that writes
// only Q_TAIL_LO does not advance the queue.
// ============================================================================

module VX_cp_axil_regfile
  import VX_cp_pkg::*;
#(
  parameter int NUM_QUEUES = VX_CP_NUM_QUEUES_C,
  parameter int ADDR_W     = 16,
  // Static device-caps fields (set at synthesis time from VX_cp_pkg).
  parameter int RING_SIZE_LOG2_MAX = VX_CP_RING_SIZE_LOG2_C,
  parameter int AXI_TID_W          = VX_CP_AXI_TID_WIDTH_C
)(
  input  wire                       clk,
  input  wire                       reset,

  // AXI-Lite slave port (single instance per cp_core).
  VX_cp_axil_s_if.slave             axil_s,

  // Aggregated CP status (OR of per-queue states, driven by cp_core).
  input  wire                       cp_busy,
  input  wire                       cp_error,

  // Per-queue runtime telemetry from each CPE.
  input  wire [63:0]                q_head    [NUM_QUEUES],
  input  wire [63:0]                q_seqnum  [NUM_QUEUES],
  input  wire [31:0]                q_error   [NUM_QUEUES],

  // Programmed state out to every CPE.
  output cpe_state_t                q_state   [NUM_QUEUES],

  // One-cycle reset pulse per queue when the host writes Q_CONTROL.reset.
  output logic                      q_reset_pulse [NUM_QUEUES]
);

  localparam int QID_W = (NUM_QUEUES > 1) ? $clog2(NUM_QUEUES) : 1;

  // ---- Per-queue programmable state ----
  logic [63:0] r_ring_base       [NUM_QUEUES];
  logic [63:0] r_head_addr       [NUM_QUEUES];
  logic [63:0] r_cmpl_addr       [NUM_QUEUES];
  logic [7:0]  r_ring_size_log2  [NUM_QUEUES];
  logic [31:0] r_control         [NUM_QUEUES];
  logic [63:0] r_tail            [NUM_QUEUES];

  // Tail-half staging registers. The host can write Q_TAIL_LO multiple
  // times before committing; we always present the most recent value
  // on the Q_TAIL_HI atomic commit.
  logic [31:0] r_tail_lo_staging [NUM_QUEUES];

  // The slave ignores wstrb — every host write is treated as full-32-bit.
  // Partial writes are a documented restriction (parent §6.10); none of
  // the runtime code emits sub-word writes to CP registers.
  `UNUSED_VAR (axil_s.wstrb)

  // ---- Global registers ----
  logic [31:0] r_cp_ctrl;
  logic [63:0] r_cycle_count;

  always_ff @(posedge clk) begin
    if (reset) r_cycle_count <= '0;
    else       r_cycle_count <= r_cycle_count + 64'd1;
  end

  // ---- Address-decode helpers ----
  // Returns 1 if `addr` is the global register at `g_off`. Globals occupy
  // 0x000..0x0FF.
  function automatic logic is_global(input logic [ADDR_W-1:0] addr,
                                     input logic [7:0]        g_off);
    return (addr[ADDR_W-1:8] == '0) && (addr[7:0] == g_off);
  endfunction

  // Returns 1 + decodes (qid, offset) if `addr` falls in a per-queue
  // block (0x100..0x100 + NUM_QUEUES * 0x40 - 1).
  function automatic logic decode_queue(input logic [ADDR_W-1:0] addr,
                                        output logic [QID_W-1:0] qid_o,
                                        output logic [5:0]       off_o);
    // Queue stride is 0x40 = 64 B, so the low 6 bits of (addr - 0x100)
    // are the per-queue offset and the next $clog2(NUM_QUEUES) bits
    // are the queue id. High bits above (qid|off) are deliberately
    // truncated — we range-check `addr` first.
    /* verilator lint_off UNUSED */
    logic [ADDR_W-1:0] rel;
    /* verilator lint_on UNUSED */
    logic [ADDR_W-1:0] end_addr;
    int                slot_idx;
    qid_o = '0;
    off_o = '0;
    end_addr = ADDR_W'(16'h0100) + ADDR_W'(NUM_QUEUES) * ADDR_W'(16'h0040);
    if (addr < ADDR_W'(16'h0100)) return 1'b0;
    if (addr >= end_addr)         return 1'b0;
    rel = addr - ADDR_W'(16'h0100);
    off_o = rel[5:0];
    qid_o = rel[QID_W+6-1:6];
    slot_idx = int'(qid_o);
    if (slot_idx >= NUM_QUEUES) return 1'b0;
    return 1'b1;
  endfunction

  // ---- Read data combinational decode ----
  function automatic logic [31:0] read_reg(input logic [ADDR_W-1:0] addr);
    logic [QID_W-1:0] qid;
    logic [5:0]       off;
    if (is_global(addr, 8'h00)) return r_cp_ctrl;
    if (is_global(addr, 8'h04)) return {30'd0, cp_error, cp_busy};
    if (is_global(addr, 8'h08)) return {8'd0,
                                        8'(AXI_TID_W),
                                        8'(RING_SIZE_LOG2_MAX),
                                        8'(NUM_QUEUES)};
    if (is_global(addr, 8'h10)) return r_cycle_count[31:0];
    if (is_global(addr, 8'h14)) return r_cycle_count[63:32];
    if (decode_queue(addr, qid, off)) begin
      case (off)
        6'h00: return r_ring_base[qid][31:0];
        6'h04: return r_ring_base[qid][63:32];
        6'h08: return r_head_addr[qid][31:0];
        6'h0C: return r_head_addr[qid][63:32];
        6'h10: return r_cmpl_addr[qid][31:0];
        6'h14: return r_cmpl_addr[qid][63:32];
        6'h18: return {24'd0, r_ring_size_log2[qid]};
        6'h1C: return r_control[qid];
        6'h20: return r_tail_lo_staging[qid];     // WO; readback for debug
        6'h24: return r_tail[qid][63:32];         // returns currently committed HI
        6'h28: return q_seqnum[qid][31:0];        // RO mirror
        6'h2C: return q_error[qid];               // RO
        default: return 32'h0;
      endcase
    end
    return 32'hDEAD_BEEF;   // returned with DECERR; sentinel aids debug
  endfunction

  function automatic logic is_decoded(input logic [ADDR_W-1:0] addr);
    /* verilator lint_off UNUSED */
    logic [QID_W-1:0] qid;   // qid is only used by callers that act on the write
    /* verilator lint_on UNUSED */
    logic [5:0]       off;
    if (is_global(addr, 8'h00)) return 1'b1;
    if (is_global(addr, 8'h04)) return 1'b1;
    if (is_global(addr, 8'h08)) return 1'b1;
    if (is_global(addr, 8'h10)) return 1'b1;
    if (is_global(addr, 8'h14)) return 1'b1;
    if (decode_queue(addr, qid, off)) begin
      case (off)
        6'h00, 6'h04, 6'h08, 6'h0C, 6'h10, 6'h14,
        6'h18, 6'h1C, 6'h20, 6'h24, 6'h28, 6'h2C: return 1'b1;
        default: return 1'b0;
      endcase
    end
    return 1'b0;
  endfunction

  // ============================================================================
  // Write channel — AW + W must both arrive before the write commits.
  // We accept them in any order and commit when both have landed.
  // ============================================================================

  logic              wr_addr_buf_valid;
  logic [ADDR_W-1:0] wr_addr_buf;
  logic              wr_data_buf_valid;
  logic [31:0]       wr_data_buf;

  // Ready when nothing is pending in the corresponding buffer.
  assign axil_s.awready = !wr_addr_buf_valid;
  assign axil_s.wready  = !wr_data_buf_valid;

  logic wr_commit;
  assign wr_commit = wr_addr_buf_valid && wr_data_buf_valid && !axil_s.bvalid;

  always_ff @(posedge clk) begin
    if (reset) begin
      wr_addr_buf_valid <= 1'b0;
      wr_data_buf_valid <= 1'b0;
      wr_addr_buf       <= '0;
      wr_data_buf       <= '0;
    end else begin
      if (axil_s.awvalid && axil_s.awready) begin
        wr_addr_buf       <= axil_s.awaddr;
        wr_addr_buf_valid <= 1'b1;
      end
      if (axil_s.wvalid && axil_s.wready) begin
        wr_data_buf       <= axil_s.wdata;
        wr_data_buf_valid <= 1'b1;
      end
      if (wr_commit) begin
        wr_addr_buf_valid <= 1'b0;
        wr_data_buf_valid <= 1'b0;
      end
    end
  end

  // Write response (B). Held until the host acknowledges with bready.
  always_ff @(posedge clk) begin
    if (reset) begin
      axil_s.bvalid <= 1'b0;
      axil_s.bresp  <= 2'b00;
    end else begin
      if (wr_commit) begin
        axil_s.bvalid <= 1'b1;
        axil_s.bresp  <= is_decoded(wr_addr_buf) ? 2'b00 : 2'b11; // OKAY / DECERR
      end else if (axil_s.bvalid && axil_s.bready) begin
        axil_s.bvalid <= 1'b0;
      end
    end
  end

  // ---- Apply the write to the underlying registers ----
  // q_reset_pulse is a 1-cycle pulse driven by Q_CONTROL.bit1 OR
  // CP_CTRL.bit1; it goes back to 0 next cycle.
  always_ff @(posedge clk) begin
    automatic logic [QID_W-1:0] qid;
    automatic logic [5:0]       off;
    if (reset) begin
      r_cp_ctrl <= '0;
      for (int i = 0; i < NUM_QUEUES; ++i) begin
        r_ring_base[i]       <= '0;
        r_head_addr[i]       <= '0;
        r_cmpl_addr[i]       <= '0;
        r_ring_size_log2[i]  <= 8'(RING_SIZE_LOG2_MAX);
        r_control[i]         <= '0;
        r_tail[i]            <= '0;
        r_tail_lo_staging[i] <= '0;
        q_reset_pulse[i]     <= 1'b0;
      end
    end else begin
      // Default the pulse low every cycle; the commit path below
      // overrides it for the one cycle when reset is requested.
      for (int i = 0; i < NUM_QUEUES; ++i) q_reset_pulse[i] <= 1'b0;

      if (wr_commit && is_decoded(wr_addr_buf)) begin
        if (is_global(wr_addr_buf, 8'h00)) begin
          r_cp_ctrl <= wr_data_buf;
          if (wr_data_buf[1]) begin
            for (int i = 0; i < NUM_QUEUES; ++i) q_reset_pulse[i] <= 1'b1;
          end
        end else if (decode_queue(wr_addr_buf, qid, off)) begin
          case (off)
            6'h00: r_ring_base[qid][31:0]  <= wr_data_buf;
            6'h04: r_ring_base[qid][63:32] <= wr_data_buf;
            6'h08: r_head_addr[qid][31:0]  <= wr_data_buf;
            6'h0C: r_head_addr[qid][63:32] <= wr_data_buf;
            6'h10: r_cmpl_addr[qid][31:0]  <= wr_data_buf;
            6'h14: r_cmpl_addr[qid][63:32] <= wr_data_buf;
            6'h18: r_ring_size_log2[qid]   <= wr_data_buf[7:0];
            6'h1C: begin
              r_control[qid] <= wr_data_buf;
              // bit1 = self-clearing reset pulse
              if (wr_data_buf[1]) q_reset_pulse[qid] <= 1'b1;
            end
            6'h20: r_tail_lo_staging[qid] <= wr_data_buf;
            6'h24: begin
              // Atomic tail commit: latch staging:hi -> tail
              r_tail[qid] <= {wr_data_buf, r_tail_lo_staging[qid]};
            end
            default: ;
          endcase
        end
      end
    end
  end

  // ============================================================================
  // Read channel — single-beat. AR latches into a buffer, R returns the
  // decoded value the next cycle (so the decode chain is registered).
  // ============================================================================

  logic              rd_addr_buf_valid;
  logic [ADDR_W-1:0] rd_addr_buf;

  assign axil_s.arready = !rd_addr_buf_valid;

  always_ff @(posedge clk) begin
    if (reset) begin
      rd_addr_buf_valid <= 1'b0;
      rd_addr_buf       <= '0;
      axil_s.rvalid     <= 1'b0;
      axil_s.rdata      <= '0;
      axil_s.rresp      <= 2'b00;
    end else begin
      if (axil_s.arvalid && axil_s.arready) begin
        rd_addr_buf       <= axil_s.araddr;
        rd_addr_buf_valid <= 1'b1;
      end
      if (rd_addr_buf_valid && !axil_s.rvalid) begin
        axil_s.rdata      <= read_reg(rd_addr_buf);
        axil_s.rresp      <= is_decoded(rd_addr_buf) ? 2'b00 : 2'b11;
        axil_s.rvalid     <= 1'b1;
        rd_addr_buf_valid <= 1'b0;
      end else if (axil_s.rvalid && axil_s.rready) begin
        axil_s.rvalid <= 1'b0;
      end
    end
  end

  // ============================================================================
  // Drive q_state outputs from the programmable registers + telemetry.
  // ============================================================================
  always_comb begin
    for (int i = 0; i < NUM_QUEUES; ++i) begin
      q_state[i]                = '0;
      q_state[i].ring_base      = r_ring_base[i];
      q_state[i].ring_size_mask = (VX_CP_RING_SIZE_LOG2_C)'(
                                    ((64'd1) << r_ring_size_log2[i]) - 64'd1);
      q_state[i].head_addr      = r_head_addr[i];
      q_state[i].cmpl_addr      = r_cmpl_addr[i];
      q_state[i].tail           = r_tail[i];
      q_state[i].head           = q_head[i];
      q_state[i].seqnum         = q_seqnum[i];
      q_state[i].prio           = r_control[i][3:2];
      q_state[i].enabled        = r_control[i][0] & r_cp_ctrl[0];
      q_state[i].profile_en     = r_control[i][4];
    end
  end

  // ============================================================================
  // Read-only telemetry needs to be unused-suppressed when NUM_QUEUES==1
  // and not all bits are consumed by q_state.
  // ============================================================================
  generate
    for (genvar gi = 0; gi < NUM_QUEUES; ++gi) begin : g_unused_telemetry
      `UNUSED_VAR (q_head[gi])
      `UNUSED_VAR (q_seqnum[gi])
      `UNUSED_VAR (q_error[gi])
    end
  endgenerate

endmodule : VX_cp_axil_regfile
