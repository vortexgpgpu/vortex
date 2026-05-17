// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_completion — writes per-queue retired seqnums to host memory
// via the CP's AXI master. Triggered by per-CPE `retire_evt` pulses.
// Parent §6.8 / RTL impl §13.
//
// Per parent §6.8: the host reads `cmpl_slot[qid]` to learn the most
// recent retired sequence number. This module is what writes that slot.
//
// Architecture for NUM_QUEUES > 1: a small FIFO captures `retire_evt`
// pulses so concurrent retires don't drop on the floor. The AXI master
// drains the FIFO one entry at a time (AW → W → B). Round-robin would
// be needed for true fairness but in practice retires from different
// CPEs are rare per-cycle events, so a simple priority encoder is fine.
//
// FSM:
//   S_IDLE     : FIFO empty → wait. Non-empty → pop, → S_REQ_AW
//   S_REQ_AW   : drive awvalid + awaddr; on awready → S_REQ_W
//   S_REQ_W    : drive wvalid + wdata = seqnum (LE in low 64 b of bus);
//                on wready → S_WAIT_B
//   S_WAIT_B   : wait for bvalid → S_IDLE
//
// For v1 (NUM_QUEUES=1) the FIFO is depth-2 — enough to absorb one
// in-flight write + one pending retire. Multi-CPE configurations
// should bump the depth proportional to NUM_QUEUES.
// ============================================================================

module VX_cp_completion
  import VX_cp_pkg::*;
#(
  parameter int NUM_QUEUES = VX_CP_NUM_QUEUES_C,
  parameter int FIFO_DEPTH = 2 * NUM_QUEUES,
  parameter int ID_W       = VX_CP_AXI_TID_WIDTH_C,
  parameter logic [ID_W-1:0] TID_PREFIX = '0
)(
  input  wire                       clk,
  input  wire                       reset,

  // Retire pulses + payload from each CPE.
  input  wire                       retire_evt    [NUM_QUEUES],
  input  wire [63:0]                retire_seqnum [NUM_QUEUES],
  input  wire [63:0]                cmpl_addr     [NUM_QUEUES],

  // AXI4 master sub-port.
  VX_cp_axi_m_if.master             axi_m
);

  // Capture (addr, seqnum) into a small FIFO each time a retire fires.
  typedef struct packed {
    logic [63:0] addr;
    logic [63:0] seqnum;
  } cmpl_ent_t;

  localparam int FIFO_PTR_W = (FIFO_DEPTH > 1) ? $clog2(FIFO_DEPTH) : 1;

  cmpl_ent_t       fifo [FIFO_DEPTH];
  logic [FIFO_PTR_W:0] wptr, rptr;   // one extra bit for full/empty disambiguation

  wire fifo_empty = (wptr == rptr);
  wire fifo_full  = ((wptr[FIFO_PTR_W-1:0] == rptr[FIFO_PTR_W-1:0])
                  && (wptr[FIFO_PTR_W] != rptr[FIFO_PTR_W]));

  // Priority-encode the retires this cycle to enqueue one per cycle.
  // Two CPEs retiring in the same cycle is unusual (KMU is single-
  // context); if it ever happens, the lower-QID retire wins this
  // cycle and the higher-QID retire's payload must be re-driven by
  // the engine next cycle (the engine's S_RETIRE only spans one cycle,
  // so this race ISN'T possible today — but the priority encoder is
  // future-proof for multi-resource retires).
  logic         enq;
  cmpl_ent_t    enq_ent;
  always_comb begin
    enq     = 1'b0;
    enq_ent = '0;
    for (int i = 0; i < NUM_QUEUES; ++i) begin
      if (!enq && retire_evt[i]) begin
        enq         = 1'b1;
        enq_ent.addr   = cmpl_addr[i];
        enq_ent.seqnum = retire_seqnum[i];
      end
    end
  end

  // FSM driving the AXI write.
  typedef enum logic [1:0] { S_IDLE, S_REQ_AW, S_REQ_W, S_WAIT_B } state_e;
  state_e state;

  cmpl_ent_t cur_ent;

  always_ff @(posedge clk) begin
    if (reset) begin
      wptr <= '0;
      rptr <= '0;
      state <= S_IDLE;
      cur_ent <= '0;
    end else begin
      // ----- Enqueue side -----
      if (enq && !fifo_full) begin
        fifo[wptr[FIFO_PTR_W-1:0]] <= enq_ent;
        wptr <= wptr + 1'b1;
      end
      // We silently drop on FIFO full — this only happens if FIFO_DEPTH
      // was sized too small for the workload. Document this as a
      // parameter tuning concern; the host can detect it via
      // CP_STATUS.error in a future revision.

      // ----- Dequeue / state machine -----
      case (state)
        S_IDLE: begin
          if (!fifo_empty) begin
            cur_ent <= fifo[rptr[FIFO_PTR_W-1:0]];
            rptr    <= rptr + 1'b1;
            state   <= S_REQ_AW;
          end
        end
        S_REQ_AW: begin
          if (axi_m.awvalid && axi_m.awready) state <= S_REQ_W;
        end
        S_REQ_W: begin
          if (axi_m.wvalid && axi_m.wready) state <= S_WAIT_B;
        end
        S_WAIT_B: begin
          if (axi_m.bvalid && axi_m.bready) state <= S_IDLE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end

  // ---- Output drivers ----
  always_comb begin
    // AR/R unused.
    axi_m.arvalid = 1'b0;
    axi_m.araddr  = '0;
    axi_m.arid    = '0;
    axi_m.arlen   = '0;
    axi_m.arsize  = '0;
    axi_m.arburst = 2'b01;
    axi_m.rready  = 1'b1;

    // AW
    axi_m.awvalid = (state == S_REQ_AW);
    axi_m.awaddr  = cur_ent.addr;
    axi_m.awid    = TID_PREFIX;
    axi_m.awlen   = 8'd0;        // single 8 B beat per write
    axi_m.awsize  = 3'd3;        // 2^3 = 8 bytes
    axi_m.awburst = 2'b01;

    // W: 64-bit seqnum at the low 8 bytes of the data bus; wstrb selects
    // those bytes. (The xbar's downstream master treats wstrb as a byte
    // enable; the host shell maps that to a partial write.)
    axi_m.wvalid = (state == S_REQ_W);
    axi_m.wdata  = '0;
    axi_m.wdata[63:0] = cur_ent.seqnum;
    axi_m.wstrb  = '0;
    axi_m.wstrb[7:0]  = 8'hFF;
    axi_m.wlast  = 1'b1;

    // B
    axi_m.bready = (state == S_WAIT_B);
  end

  // Sanity / unused.
  `UNUSED_VAR (axi_m.bid)
  `UNUSED_VAR (axi_m.bresp)
  `UNUSED_VAR (axi_m.arready)
  `UNUSED_VAR (axi_m.rvalid)
  `UNUSED_VAR (axi_m.rdata)
  `UNUSED_VAR (axi_m.rid)
  `UNUSED_VAR (axi_m.rlast)
  `UNUSED_VAR (axi_m.rresp)

endmodule : VX_cp_completion
