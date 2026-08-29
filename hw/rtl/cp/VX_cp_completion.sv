// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_completion — writes per-queue retired seqnums to host memory via
// the CP's AXI master. Triggered by per-CPE `retire_evt` pulses; the host
// reads `cmpl_addr[qid]` to learn the most recently retired seqnum.
//
// Per-source 1-deep latch + shared drain FIFO, with `retire_ready[i]`
// back-pressure so no retire is ever dropped:
//
//   * When `retire_evt[i]` fires, the source's payload is latched in a
//     per-source pending register. The engine holds `retire_evt[i]` high
//     until `retire_ready[i]` asserts, then deasserts and advances.
//   * A round-robin selector picks one pending source per cycle and pushes
//     its payload into the shared drain FIFO (if not full). On enqueue,
//     the chosen source's pending bit clears AND its `retire_ready` is
//     asserted that cycle, releasing the engine.
//   * If two CPEs retire on the same cycle, BOTH latch immediately; the
//     selector drains them on consecutive cycles, so neither retire is
//     lost.
//   * The drain FIFO absorbs bursty AXI back-pressure; if it ever fills,
//     `retire_ready[i]` stays low and the engine stalls in S_RETIRE
//     until space frees — propagating back-pressure all the way to fetch
//     instead of silently dropping a seqnum on the floor.
//
// AXI drain FSM: S_IDLE → S_REQ_AW → S_REQ_W → S_WAIT_B → S_FLUSH_AR →
// S_FLUSH_R (read-back of the written line; see the note at the FSM).
// FIFO_DEPTH defaults to 2 * NUM_QUEUES, sized to keep at least one
// completion in flight per queue under typical AXI latency.
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

  // Retire pulses + payload from each CPE. retire_evt[i] is held by the
  // engine until retire_ready[i] is observed (valid/ready handshake).
  input  wire                       retire_evt    [NUM_QUEUES],
  input  wire [63:0]                retire_seqnum [NUM_QUEUES],
  input  wire [63:0]                cmpl_addr     [NUM_QUEUES],
  output logic                      retire_ready  [NUM_QUEUES],

  // AXI4 master sub-port.
  VX_mem_axi_if.master             axi_m
);

  // Per-source latch: addr+seqnum captured on retire_evt fire. `pending[i]`
  // is set the cycle a retire is captured; cleared the cycle the selector
  // picks it. The engine cannot fire a second retire while pending is set
  // because retire_ready stays low — back-pressure to the engine.
  typedef struct packed {
    logic [63:0] addr;
    logic [63:0] seqnum;
  } cmpl_ent_t;

  cmpl_ent_t pending_ent [NUM_QUEUES];
  logic      pending     [NUM_QUEUES];

  localparam int FIFO_PTR_W = (FIFO_DEPTH > 1) ? $clog2(FIFO_DEPTH) : 1;

  cmpl_ent_t       fifo [FIFO_DEPTH];
  logic [FIFO_PTR_W:0] wptr, rptr;   // one extra bit for full/empty disambiguation

  wire fifo_empty = (wptr == rptr);
  wire fifo_full  = ((wptr[FIFO_PTR_W-1:0] == rptr[FIFO_PTR_W-1:0])
                  && (wptr[FIFO_PTR_W] != rptr[FIFO_PTR_W]));

  // Round-robin selector — picks one pending source per cycle. Rotating
  // start prevents starvation of higher-QID sources when several retire
  // back-to-back. SEL_W is at least 1 so the signal is always declared
  // even for the NUM_QUEUES==1 build (where the selector is degenerate).
  localparam int SEL_W = (NUM_QUEUES > 1) ? $clog2(NUM_QUEUES) : 1;

  logic [SEL_W-1:0]       rr_ptr;
  logic                   sel_valid;
  logic [SEL_W-1:0]       sel_idx;
  always_comb begin
    sel_valid = 1'b0;
    sel_idx   = '0;
    for (int k = 0; k < NUM_QUEUES; ++k) begin
      // SEL_W-bit modular index — wraps automatically when NUM_QUEUES is a
      // power of two; non-pow2 NUM_QUEUES uses an explicit modulo.
      logic [SEL_W-1:0] idx;
      if (NUM_QUEUES > 1) begin
        idx = SEL_W'((int'(rr_ptr) + k) % NUM_QUEUES);
      end else begin
        idx = '0;
      end
      if (!sel_valid && pending[idx]) begin
        sel_valid = 1'b1;
        sel_idx   = idx;
      end
    end
  end

  // Engine handshake: the selected source's retire_ready goes high THIS
  // cycle iff its payload is being pushed to the FIFO. Capture-side then
  // clears `pending` on the next clock.
  wire enq = sel_valid && !fifo_full;
  always_comb begin
    for (int i = 0; i < NUM_QUEUES; ++i) begin
      retire_ready[i] = enq && (sel_idx == SEL_W'(i));
    end
  end

  // FSM driving the AXI write, then a SHOVE write to the adjacent line.
  //
  // On the V80's HBM host path, the newest write into the NoC ingress stays
  // invisible to the host's QDMA reads until later WRITE traffic pushes it
  // through -- measured on silicon as the completion line trailing Q_SEQNUM
  // by exactly one retire, for over a second, immune to polling patterns,
  // cache-eviction sweeps and a same-port read-back (the drain counters
  // prove every write left the AFU with its BRESP collected, so the parking
  // is beyond the AFU and reads do not dislodge it). What does dislodge it,
  // every time, is the next command's writes.
  //
  // So supply that push ourselves: after the completion write's BRESP, write
  // the seqnum again to the line 64 B above (the completion region is a
  // dedicated >=4 KB allocation, so the address exists and the host ignores
  // it). Ingress order is data flits -> completion flit -> shove flit; when
  // the host observes the completion, everything written before it -- the
  // download payload included -- is observable too. The shove itself parks
  // where nothing reads.
  typedef enum logic [2:0] {
    S_IDLE, S_REQ_AW, S_REQ_W, S_WAIT_B, S_SHOVE_AW, S_SHOVE_W, S_SHOVE_B
  } state_e;
  state_e state;

  cmpl_ent_t cur_ent;

  always_ff @(posedge clk) begin
    if (reset) begin
      wptr <= '0;
      rptr <= '0;
      state <= S_IDLE;
      cur_ent <= '0;
      rr_ptr <= '0;
      for (int i = 0; i < NUM_QUEUES; ++i) begin
        pending[i]     <= 1'b0;
        pending_ent[i] <= '0;
      end
    end else begin
      // ----- Capture side: per-source latch -----
      // retire_evt[i] is held by the engine until retire_ready[i] fires.
      // While !pending[i], latch the payload; while pending[i], the engine
      // sees retire_ready=0 and re-presents the same retire next cycle.
      for (int i = 0; i < NUM_QUEUES; ++i) begin
        if (retire_evt[i] && !pending[i]) begin
          pending[i]     <= 1'b1;
          pending_ent[i] <= '{addr:   cmpl_addr[i],
                              seqnum: retire_seqnum[i]};
        end
      end
      // ----- Selector / drain into shared FIFO -----
      if (enq) begin
        fifo[wptr[FIFO_PTR_W-1:0]] <= pending_ent[sel_idx];
        wptr <= wptr + 1'b1;
        pending[sel_idx] <= 1'b0;     // clears the captured source
        // Rotate the round-robin pointer past the served source so the
        // next cycle starts the scan AFTER it.
        if (NUM_QUEUES > 1)
          rr_ptr <= SEL_W'((int'(sel_idx) + 1) % NUM_QUEUES);
      end

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
          if (axi_m.bvalid && axi_m.bready) state <= S_SHOVE_AW;
        end
        S_SHOVE_AW: begin
          if (axi_m.awvalid && axi_m.awready) state <= S_SHOVE_W;
        end
        S_SHOVE_W: begin
          if (axi_m.wvalid && axi_m.wready) state <= S_SHOVE_B;
        end
        S_SHOVE_B: begin
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
    axi_m.arlen   = 8'd0;
    axi_m.arsize  = 3'd6;
    axi_m.arburst = 2'b01;
    axi_m.rready  = 1'b1;

    // AW. A full cacheline, not an 8-byte narrow write. The line is a
    // dedicated CP-owned allocation, so the padding clobbers nothing -- and
    // the full-width beat is load-bearing on hardware: this used to be
    // awsize=3 with wstrb[7:0], the ONLY partial-line write any CP master
    // issues, and on the V80 that write parked between the AFU port and HBM
    // (a partial write costs the memory system a read-modify-write) and
    // stayed invisible to host reads of the line long after its BRESP --
    // measured: still absent ONE SECOND later, landing only around the next
    // retire's write. Every full-line write from the very same port lands
    // promptly. A completion signal that trails its BRESP by a second is not
    // a completion signal.
    axi_m.awvalid = (state == S_REQ_AW) || (state == S_SHOVE_AW);
    axi_m.awaddr  = (state == S_SHOVE_AW) ? (cur_ent.addr + 64'd64)
                                          : cur_ent.addr;
    axi_m.awid    = TID_PREFIX;
    axi_m.awlen   = 8'd0;        // single full-width beat
    axi_m.awsize  = 3'd6;        // 2^6 = 64 bytes
    axi_m.awburst = 2'b01;

    // W: 64-bit seqnum in the low 8 bytes, the rest of the line written as
    // zero. All strobes set -- see the AW note.
    axi_m.wvalid = (state == S_REQ_W) || (state == S_SHOVE_W);
    axi_m.wdata  = '0;
    axi_m.wdata[63:0] = cur_ent.seqnum;
    axi_m.wstrb  = '1;
    axi_m.wlast  = 1'b1;

    // B
    axi_m.bready = (state == S_WAIT_B) || (state == S_SHOVE_B);
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
