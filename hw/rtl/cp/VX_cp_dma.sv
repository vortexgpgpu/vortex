// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_cp_dma — dual-port burst DMA engine for CMD_MEM_WRITE / CMD_MEM_READ /
// CMD_MEM_COPY. Owned by the DMA resource arbiter.
//
// Command encoding:
//   arg0 = dst address
//   arg1 = src address
//   arg2 = transfer size in bytes (rounded up to a 64-byte multiple)
//
// Dual port: XRT pins each kernel AXI master to exactly one memory resource
// (an HBM/DDR bank or HOST[0]), so the CP carries two masters — axi_host for
// host memory (the command ring lives there and it is one end of every
// upload/download) and axi_dev for device memory. The opcode selects the
// read-source and write-destination port:
//   CMD_MEM_WRITE : src = host,   dst = device   (upload)
//   CMD_MEM_READ  : src = device, dst = host     (download)
//   CMD_MEM_COPY  : src = device, dst = device   (device-local copy)
//
// Transfers are streamed in <=4 KB chunks; each chunk is one AXI INCR burst
// (up to MAX_BURST 512-bit beats) so no burst crosses a 4 KB address
// boundary. A chunk is read fully into buf_r, then written out — sequential,
// not pipelined (pipelining R against W is a later optimization).
//
// FSM:
//   S_IDLE   : grant -> latch op/dst/src/size                 -> S_SETUP
//   S_SETUP  : size 0 -> S_DONE; else size the chunk          -> S_REQ_AR
//   S_REQ_AR : drive AR on the read port; arready             -> S_READ
//   S_READ   : capture rdata beats into buf_r; last beat      -> S_REQ_AW
//   S_REQ_AW : drive AW on the write port; awready            -> S_WRITE
//   S_WRITE  : drive W beats from buf_r; last beat            -> S_WAIT_B
//   S_WAIT_B : bvalid -> advance chunk                        -> S_SETUP
//   S_DONE   : pulse `done` for one cycle                     -> S_IDLE
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
  /* verilator lint_off UNUSED */
  input  cmd_t                      cmd,
  /* verilator lint_on UNUSED */
  output logic                      done,

  // Host-memory AXI master (command-ring side / upload source / download dst).
  VX_cp_axi_m_if.master             axi_host,
  // Device-memory AXI master.
  VX_cp_axi_m_if.master             axi_dev
);

  localparam int MAX_BURST = 64;          // 64 x 64 B = 4 KB max per burst
  localparam int BIDX_W    = 6;           // beat index 0..63
  localparam int BCNT_W    = 7;           // chunk length 1..64

  typedef enum logic [2:0] {
    S_IDLE, S_SETUP, S_REQ_AR, S_READ, S_REQ_AW, S_WRITE, S_WAIT_B, S_DONE
  } state_e;

  state_e               state;
  logic [7:0]           op_r;             // latched opcode (host/dev routing)
  logic [63:0]          dst_r, src_r;
  logic [63:0]          rem_beats;        // 64 B beats still to move
  logic [BCNT_W-1:0]    chunk_beats;      // beats in the current chunk
  logic [BIDX_W-1:0]    beat_idx;
  logic [CL_BITS-1:0]   buf_r [MAX_BURST];

  // Beats from a 64 B-aligned address to the next 4 KB boundary. `cl_idx`
  // is the cache-line index inside the 4 KB page (addr[11:6], 0..63).
  function automatic logic [BCNT_W-1:0] beats_to_4k(input logic [5:0] cl_idx);
    return BCNT_W'(MAX_BURST) - BCNT_W'({1'b0, cl_idx});
  endfunction

  // Next chunk length = min(rem_beats, src 4K span, dst 4K span).
  logic [BCNT_W-1:0] next_chunk;
  always_comb begin
    logic [BCNT_W-1:0] s4k, d4k, lim;
    s4k = beats_to_4k(src_r[11:6]);
    d4k = beats_to_4k(dst_r[11:6]);
    lim = (s4k < d4k) ? s4k : d4k;
    if (rem_beats < 64'({1'b0, lim}))
      next_chunk = BCNT_W'(rem_beats);
    else
      next_chunk = lim;
  end

  // Read-source / write-destination port selection from the latched opcode.
  wire rd_from_host = (cp_opcode_e'(op_r) == CMD_MEM_WRITE);  // upload: read host
  wire wr_to_host   = (cp_opcode_e'(op_r) == CMD_MEM_READ);   // download: write host

  // Last beat of the current chunk.
  wire last_beat = (BCNT_W'({1'b0, beat_idx}) == (chunk_beats - BCNT_W'(1)));

  // ---- FSM ----
  always_ff @(posedge clk) begin
    if (reset) begin
      state       <= S_IDLE;
      op_r        <= '0;
      dst_r       <= '0;
      src_r       <= '0;
      rem_beats   <= '0;
      chunk_beats <= '0;
      beat_idx    <= '0;
    end else begin
      case (state)
        S_IDLE: begin
          if (grant) begin
            op_r      <= cmd.hdr.opcode;
            dst_r     <= cmd.arg0;
            src_r     <= cmd.arg1;
            // Round the byte count up to a whole cache line.
            rem_beats <= (cmd.arg2 + 64'd63) >> 6;
            state     <= S_SETUP;
          end
        end
        S_SETUP: begin
          if (rem_beats == 64'd0) begin
            state <= S_DONE;
          end else begin
            chunk_beats <= next_chunk;
            beat_idx    <= '0;
            state       <= S_REQ_AR;
          end
        end
        S_REQ_AR: begin
          if (rd_arvalid && rd_arready) begin
            beat_idx <= '0;
            state    <= S_READ;
          end
        end
        S_READ: begin
          if (rd_rvalid && rd_rready) begin
            buf_r[beat_idx] <= rd_rdata;
            if (last_beat) begin
              beat_idx <= '0;
              state    <= S_REQ_AW;
            end else begin
              beat_idx <= beat_idx + BIDX_W'(1);
            end
          end
        end
        S_REQ_AW: begin
          if (wr_awvalid && wr_awready) begin
            beat_idx <= '0;
            state    <= S_WRITE;
          end
        end
        S_WRITE: begin
          if (wr_wvalid && wr_wready) begin
            if (last_beat) begin
              state <= S_WAIT_B;
            end else begin
              beat_idx <= beat_idx + BIDX_W'(1);
            end
          end
        end
        S_WAIT_B: begin
          if (wr_bvalid && wr_bready) begin
            src_r     <= src_r + (64'({1'b0, chunk_beats}) << 6);
            dst_r     <= dst_r + (64'({1'b0, chunk_beats}) << 6);
            rem_beats <= rem_beats - 64'({1'b0, chunk_beats});
            state     <= S_SETUP;
          end
        end
        S_DONE: begin
          state <= S_IDLE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end

  // ---- Logical read channel ----
  wire               rd_arvalid = (state == S_REQ_AR);
  wire               rd_rready  = (state == S_READ);
  wire               rd_arready = rd_from_host ? axi_host.arready : axi_dev.arready;
  wire               rd_rvalid  = rd_from_host ? axi_host.rvalid  : axi_dev.rvalid;
  wire [CL_BITS-1:0] rd_rdata   = rd_from_host ? axi_host.rdata   : axi_dev.rdata;

  // ---- Logical write channel ----
  wire               wr_awvalid = (state == S_REQ_AW);
  wire               wr_wvalid  = (state == S_WRITE);
  wire               wr_bready  = (state == S_WAIT_B);
  wire               wr_awready = wr_to_host ? axi_host.awready : axi_dev.awready;
  wire               wr_wready  = wr_to_host ? axi_host.wready  : axi_dev.wready;
  wire               wr_bvalid  = wr_to_host ? axi_host.bvalid  : axi_dev.bvalid;

  wire [7:0]         burst_len  = 8'({1'b0, chunk_beats - BCNT_W'(1)});

  // ---- Drive both AXI masters; only the routed port asserts valid ----
  always_comb begin
    // ----- axi_host -----
    axi_host.arvalid = rd_arvalid &  rd_from_host;
    axi_host.araddr  = src_r;
    axi_host.arid    = TID_PREFIX;
    axi_host.arlen   = burst_len;
    axi_host.arsize  = 3'd6;                 // 64 bytes per beat
    axi_host.arburst = 2'b01;                // INCR
    axi_host.rready  = rd_rready  &  rd_from_host;

    axi_host.awvalid = wr_awvalid &  wr_to_host;
    axi_host.awaddr  = dst_r;
    axi_host.awid    = TID_PREFIX;
    axi_host.awlen   = burst_len;
    axi_host.awsize  = 3'd6;
    axi_host.awburst = 2'b01;
    axi_host.wvalid  = wr_wvalid  &  wr_to_host;
    axi_host.wdata   = buf_r[beat_idx];
    axi_host.wstrb   = '1;
    axi_host.wlast   = last_beat;
    axi_host.bready  = wr_bready  &  wr_to_host;

    // ----- axi_dev -----
    axi_dev.arvalid  = rd_arvalid & ~rd_from_host;
    axi_dev.araddr   = src_r;
    axi_dev.arid     = TID_PREFIX;
    axi_dev.arlen    = burst_len;
    axi_dev.arsize   = 3'd6;
    axi_dev.arburst  = 2'b01;
    axi_dev.rready   = rd_rready  & ~rd_from_host;

    axi_dev.awvalid  = wr_awvalid & ~wr_to_host;
    axi_dev.awaddr   = dst_r;
    axi_dev.awid     = TID_PREFIX;
    axi_dev.awlen    = burst_len;
    axi_dev.awsize   = 3'd6;
    axi_dev.awburst  = 2'b01;
    axi_dev.wvalid   = wr_wvalid  & ~wr_to_host;
    axi_dev.wdata    = buf_r[beat_idx];
    axi_dev.wstrb    = '1;
    axi_dev.wlast    = last_beat;
    axi_dev.bready   = wr_bready  & ~wr_to_host;

    done = (state == S_DONE);
  end

  // Sanity / unused.
  `UNUSED_VAR (cmd.hdr.flags)
  `UNUSED_VAR (cmd.hdr.reserved)
  `UNUSED_VAR (cmd.profile_slot)
  `UNUSED_VAR (axi_host.bid)
  `UNUSED_VAR (axi_host.bresp)
  `UNUSED_VAR (axi_host.rid)
  `UNUSED_VAR (axi_host.rlast)
  `UNUSED_VAR (axi_host.rresp)
  `UNUSED_VAR (axi_dev.bid)
  `UNUSED_VAR (axi_dev.bresp)
  `UNUSED_VAR (axi_dev.rid)
  `UNUSED_VAR (axi_dev.rlast)
  `UNUSED_VAR (axi_dev.rresp)

endmodule : VX_cp_dma
