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
// not pipelined.
//
// FSM:
//   S_IDLE   : grant -> latch op/dst/src/size                 -> S_XLATE_AR
//              (no device operand to translate)               -> S_SETUP
//   S_XLATE_AR/S_XLATE_R : page-table walk of each device operand, one PTE
//              read per level on the device port; the leaf rewrites the
//              operand with its physical address              -> S_SETUP
//   S_SETUP  : size 0 -> S_DONE; else size the chunk          -> S_REQ_AR
//   S_REQ_AR : drive AR on the read port; arready             -> S_READ
//   S_READ   : capture rdata beats into buf_r; last beat      -> S_REQ_AW
//   S_REQ_AW : drive AW on the write port; awready            -> S_WRITE
//   S_WRITE  : drive W beats from buf_r; last beat            -> S_WAIT_B
//   S_WAIT_B : bvalid -> host-direction: validate the chunk   -> S_FLUSH_AR
//              device-direction: advance chunk                -> S_SETUP
//   S_FLUSH_AR/S_FLUSH_R : host-direction only -- read back the last line of
//              the chunk just written and COMPARE it against the copy still
//              held in buf_r; mismatch re-issues the read     -> S_SETUP
//   S_DONE   : pulse `done` for one cycle                     -> S_IDLE
//
// WHY THE VALIDATING FLUSH EXISTS. On the V80's HBM host path, a write's
// BRESP does not mean the data is visible to another master: the response
// can be issued upstream of the memory controller while the beats are still
// in flight, and the host's QDMA reads (a different master, and the CP's
// completion writer a different AXI ID) have no ordering against them. The
// only proof of commitment is a read of the written address RETURNING THE
// WRITTEN DATA: HBM serializes per address, so a matching read-back means
// the write reached the controller, and every earlier write on this ID to
// the same destination is ordered before it. Each chunk lives inside one
// 4 KB page (see beats_to_4k), so its beats share one destination and the
// chunk's last line vouches for the whole chunk. A stale read-back simply
// retries -- the write is in flight and lands within microseconds. Only
// after every chunk has validated does `done` (and therefore the retire the
// host polls, and the completion line that follows it) fire.
// Device-direction writes don't need this: nothing reads them through a
// foreign port on the strength of Q_SEQNUM alone.
//
// VIRTUAL MEMORY. With a page-table root in CP_SATP, a device operand is a
// virtual address, as on a GPU copy engine. Every allocation maps to one
// contiguous physical range, so translating the operand's base covers the
// whole transfer. Commands flagged F_MEM_PHYSICAL (the page tables
// themselves) skip the walk.
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
  input  cmd_t                      cmd,
  output logic                      done,

  // CP_SATP: page-table root for the device operands (0 or bare: physical).
  input  wire [63:0]                satp,

  // Host-memory AXI master (command-ring side / upload source / download dst).
  VX_mem_axi_if.master             axi_host,
  // Device-memory AXI master.
  VX_mem_axi_if.master             axi_dev
);

  localparam int MAX_BURST = 64;          // 64 x 64 B = 4 KB max per burst
  localparam int BIDX_W    = 6;           // beat index 0..63
  localparam int BCNT_W    = 7;           // chunk length 1..64

  typedef enum logic [3:0] {
    S_IDLE, S_XLATE_AR, S_XLATE_R, S_SETUP, S_REQ_AR, S_READ, S_REQ_AW,
    S_WRITE, S_WAIT_B, S_FLUSH_AR, S_FLUSH_R, S_DONE
  } state_e;

  // Page-table geometry: Sv32 at XLEN=32, Sv39 at XLEN=64.
  localparam int PAGE_LOG2  = `VX_VM_PAGE_LOG2_SIZE;
  localparam int PT_LEVELS  = `VX_VM_PT_LEVEL;
  localparam int LEVEL_BITS = `CLOG2(`VX_VM_PT_SIZE / `VX_VM_PTE_SIZE);
  localparam int PTE_BITS   = `VX_VM_PTE_SIZE * 8;
  localparam int PTE_SHIFT  = `CLOG2(`VX_VM_PTE_SIZE);
  localparam int PTE_SEL_W  = 6 - PTE_SHIFT;
  localparam int PPN_W      = (PTE_BITS == 32) ? 22 : 44;
  localparam int LVL_W      = `UP(`CLOG2(PT_LEVELS));

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
  //
  // The span comparison is done on the cache-line indices rather than on the
  // spans: since beats_to_4k is (MAX_BURST - cl_idx) and cl_idx is 6 bits so
  // never underflows, min(MAX_BURST - s, MAX_BURST - d) == MAX_BURST -
  // max(s, d). That is one 6-bit compare followed by one subtract, instead of
  // two subtracts followed by a compare, and it shortens the src_r ->
  // chunk_beats path that limits this design at 300 MHz.
  logic [BCNT_W-1:0] next_chunk;
  always_comb begin
    logic [5:0]        cl_max;
    logic [BCNT_W-1:0] lim;
    cl_max = (src_r[11:6] > dst_r[11:6]) ? src_r[11:6] : dst_r[11:6];
    lim    = beats_to_4k(cl_max);
    if (rem_beats < 64'({1'b0, lim}))
      next_chunk = BCNT_W'(rem_beats);
    else
      next_chunk = lim;
  end

  // Read-source / write-destination port selection from the latched opcode.
  wire rd_from_host = (cp_opcode_e'(op_r) == CMD_MEM_WRITE);  // upload: read host
  wire wr_to_host   = (cp_opcode_e'(op_r) == CMD_MEM_READ);   // download: write host

  // ---- Page-table walk of the device operands ----
`ifdef VX_VM_ADDR_MODE_SV32
  wire             satp_on  = satp[31];
  wire [PPN_W-1:0] satp_ppn = satp[PPN_W-1:0];
  `UNUSED_VAR (satp[63:32])
  `UNUSED_VAR (satp[30:PPN_W])
`else
  wire             satp_on  = (satp[63:60] != '0);
  wire [PPN_W-1:0] satp_ppn = satp[PPN_W-1:0];
  `UNUSED_VAR (satp[59:PPN_W])
`endif

  wire cmd_is_mem   = (cp_opcode_e'(cmd.hdr.opcode) == CMD_MEM_WRITE)
                   || (cp_opcode_e'(cmd.hdr.opcode) == CMD_MEM_READ)
                   || (cp_opcode_e'(cmd.hdr.opcode) == CMD_MEM_COPY);
  wire cmd_xlate    = (`VX_CFG_VM_ENABLED != 0) && satp_on && cmd_is_mem
                   && ~cmd.hdr.flags[F_MEM_PHYSICAL];
  wire cmd_xl_dst   = cmd_xlate && (cp_opcode_e'(cmd.hdr.opcode) != CMD_MEM_READ);
  wire cmd_xl_src   = cmd_xlate && (cp_opcode_e'(cmd.hdr.opcode) != CMD_MEM_WRITE);

  logic             xl_dst_r, xl_src_r;   // operands still to translate (dst first)
  logic [LVL_W-1:0] xl_level_r;
  logic [PPN_W-1:0] xl_ppn_r;             // table base at the current level

  wire [63:0]           xl_va     = xl_dst_r ? dst_r : src_r;
  wire [LEVEL_BITS-1:0] xl_vpn    = xl_va[PAGE_LOG2 + xl_level_r * LEVEL_BITS +: LEVEL_BITS];
  wire [63:0]           pte_addr  = (64'(xl_ppn_r) << PAGE_LOG2) | (64'(xl_vpn) << PTE_SHIFT);
  wire [PTE_SEL_W-1:0]  pte_sel   = pte_addr[5:PTE_SHIFT];
  wire [PTE_BITS-1:0]   pte       = axi_dev.rdata[pte_sel * PTE_BITS +: PTE_BITS];
  wire [PPN_W-1:0]      pte_ppn   = pte[10 +: PPN_W];
  wire                  pte_v     = pte[0];
  wire                  pte_leaf  = pte[1] | pte[2] | pte[3];   // R | W | X
  // A leaf above level 0 is a superpage: the VA supplies its low bits.
  wire [63:0]           off_mask  = (64'(1) << (PAGE_LOG2 + xl_level_r * LEVEL_BITS)) - 64'(1);
  wire [63:0]           leaf_pa   = ((64'(pte_ppn) << PAGE_LOG2) & ~off_mask) | (xl_va & off_mask);
  // An unmapped or leafless walk leaves the address as it is, as the
  // functional model does.
  wire                  xl_last   = ~pte_v || pte_leaf || (xl_level_r == '0);
  wire [63:0]           xl_result = (pte_v && pte_leaf) ? leaf_pa : xl_va;
  if (PTE_BITS > 10 + PPN_W) begin : g_pte_unused
    `UNUSED_VAR (pte[PTE_BITS-1:10+PPN_W])
  end
  `UNUSED_VAR (pte[9:4])

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
      xl_dst_r    <= 1'b0;
      xl_src_r    <= 1'b0;
      xl_level_r  <= '0;
      xl_ppn_r    <= '0;
    end else begin
      case (state)
        S_IDLE: begin
          if (grant) begin
            op_r       <= cmd.hdr.opcode;
            dst_r      <= cmd.arg0;
            src_r      <= cmd.arg1;
            // Round the byte count up to a whole cache line.
            rem_beats  <= (cmd.arg2 + 64'd63) >> 6;
            xl_dst_r   <= cmd_xl_dst;
            xl_src_r   <= cmd_xl_src;
            xl_level_r <= LVL_W'(PT_LEVELS - 1);
            xl_ppn_r   <= satp_ppn;
            state      <= (cmd_xl_dst || cmd_xl_src) ? S_XLATE_AR : S_SETUP;
          end
        end
        S_XLATE_AR: begin
          if (axi_dev.arvalid && axi_dev.arready) begin
            state <= S_XLATE_R;
          end
        end
        S_XLATE_R: begin
          if (axi_dev.rvalid && axi_dev.rready) begin
            if (xl_last) begin
              if (xl_dst_r) begin
                dst_r    <= xl_result;
                xl_dst_r <= 1'b0;
              end else begin
                src_r    <= xl_result;
                xl_src_r <= 1'b0;
              end
              xl_level_r <= LVL_W'(PT_LEVELS - 1);
              xl_ppn_r   <= satp_ppn;
              state      <= (xl_dst_r && xl_src_r) ? S_XLATE_AR : S_SETUP;
            end else begin
              xl_level_r <= xl_level_r - LVL_W'(1);
              xl_ppn_r   <= pte_ppn;
              state      <= S_XLATE_AR;
            end
          end
        end
        S_SETUP: begin
          if (rem_beats == 64'd0) begin
            // Every host-direction chunk validated inline (S_WAIT_B ->
            // S_FLUSH_*), so nothing is left to prove here.
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
            // Host-direction: prove this chunk committed before moving on.
            // (dst_r advances this cycle, so the flush address computed in
            // S_FLUSH_AR, dst_r - 64, is this chunk's last line.)
            state     <= wr_to_host ? S_FLUSH_AR : S_SETUP;
          end
        end
        S_FLUSH_AR: begin
          if (axi_host.arvalid && axi_host.arready) begin
            state <= S_FLUSH_R;
          end
        end
        S_FLUSH_R: begin
          if (axi_host.rvalid && axi_host.rready) begin
            // Validate: the read-back must equal the line still held in
            // buf_r. Stale data means the write has not reached the memory
            // controller yet -- re-issue the read until it has.
            state <= (axi_host.rdata == buf_r[BIDX_W'(chunk_beats - BCNT_W'(1))])
                       ? S_SETUP : S_FLUSH_AR;
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

  // The read-ready decode drives the clock enable of every bit of the stream
  // buffer feeding this engine -- 264 enables that the placer spreads well
  // away from this FSM. At 300 MHz that path is the design's critical one:
  // three logic levels carrying 0.267ns of logic behind 2.008ns of route.
  // Replicating the decode lets each copy enable a nearby group. Logically
  // identical to inlining these expressions; affects implementation only.
  (* MAX_FANOUT = 32 *) wire host_rready = (rd_rready & rd_from_host) || (state == S_FLUSH_R);
  (* MAX_FANOUT = 32 *) wire dev_rready  = (rd_rready & ~rd_from_host) || (state == S_XLATE_R);

  // ---- Drive both AXI masters; only the routed port asserts valid ----
  always_comb begin
    // ----- axi_host -----
    axi_host.arvalid = (rd_arvalid & rd_from_host) || (state == S_FLUSH_AR);
    axi_host.araddr  = (state == S_FLUSH_AR) ? (dst_r - 64'd64) : src_r;
    axi_host.arid    = TID_PREFIX;
    axi_host.arlen   = (state == S_FLUSH_AR) ? 8'd0 : burst_len;
    axi_host.arsize  = 3'd6;                 // 64 bytes per beat
    axi_host.arburst = 2'b01;                // INCR
    axi_host.rready  = host_rready;

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
    axi_dev.arvalid  = (rd_arvalid & ~rd_from_host) || (state == S_XLATE_AR);
    axi_dev.araddr   = (state == S_XLATE_AR) ? {pte_addr[63:6], 6'd0} : src_r;
    axi_dev.arid     = TID_PREFIX;
    axi_dev.arlen    = (state == S_XLATE_AR) ? 8'd0 : burst_len;
    axi_dev.arsize   = 3'd6;
    axi_dev.arburst  = 2'b01;
    axi_dev.rready   = dev_rready;

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
  `UNUSED_VAR (pte_addr[5:0])
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
