// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// DXA CL-to-SMEM width converter (v3 — per-CL independent drain).
//
// Each CL carries an explicit SMEM byte address from the tracker.
// The CL is drained completely before accepting the next CL (no cross-CL
// residual). This enables out-of-order GMEM responses.
//
// General-purpose format converter: works for any CL/SMEM ratio.
// Compresses valid bytes via barrel shift (requires contiguous byte masks).
// Uses 2-state FSM (ACCEPT/DRAIN) with overlap: accept fires on the
// last drain cycle to eliminate pipeline bubbles.
//
// Key protocol property: smem_out_valid and smem_out_data are purely
// register-driven (no dependency on smem_out_ready), preventing
// combinational deadlock in the valid/ready handshake.

`include "VX_define.vh"

module VX_dxa_cl2smem import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter CL_SIZE        = `L1_LINE_SIZE,
    parameter SMEM_WORD_SIZE = 16,
    parameter SMEM_ADDR_WIDTH = DXA_SMEM_ADDR_WIDTH
) (
    input  wire clk,
    input  wire reset,
    input  wire start,

    // CL input (from rd_ctrl, valid/ready).
    input  wire                        cl_in_valid,
    output wire                        cl_in_ready,
    input  wire [CL_SIZE*8-1:0]        cl_in_data,
    input  wire [CL_SIZE-1:0]          cl_in_byte_mask,
    input  wire                        cl_in_last,

    // Explicit SMEM byte address for this CL (from tracker via rd_ctrl).
    input  wire [`MEM_ADDR_WIDTH-1:0]  cl_in_smem_byte_addr,

    // SMEM word output (to wr_ctrl, valid/ready).
    output wire                        smem_out_valid,
    input  wire                        smem_out_ready,
    output wire [SMEM_WORD_SIZE*8-1:0] smem_out_data,
    output wire [SMEM_WORD_SIZE-1:0]   smem_out_byteen,
    output wire                        smem_out_last,
    output wire [SMEM_ADDR_WIDTH-1:0]  smem_out_word_addr,

    // Pipeline idle: fill buffer is in ACCEPT state with no pending data.
    output wire                        idle
);
    localparam CL_BITS = `CLOG2(CL_SIZE);
    localparam COUNT_W = CL_BITS + 1;
    localparam SMEM_OFF_W = `CLOG2(SMEM_WORD_SIZE);
    localparam FILL_CAP = CL_SIZE + SMEM_WORD_SIZE;
    localparam FILL_W   = `CLOG2(FILL_CAP + 1);

    `STATIC_ASSERT(`IS_POW2(CL_SIZE), ("CL_SIZE must be power of 2"))
    `STATIC_ASSERT(`IS_POW2(SMEM_WORD_SIZE), ("SMEM_WORD_SIZE must be power of 2"))

    // ════════════════════════════════════════════════════════════
    // Sub-A: Barrel Shift (contiguous byte mask compression)
    // ════════════════════════════════════════════════════════════

    // Barrel Shift: relies on Phase 1 invariant that cl_in_byte_mask
    // contains one contiguous run of 1s. Find the first set bit and
    // right-shift data to compress valid bytes to position 0.

    wire [CL_BITS-1:0] shift_amount;
    wire               mask_nonzero;

    VX_priority_encoder #(
        .N(CL_SIZE)
    ) barrel_pe (
        .data_in   (cl_in_byte_mask),
        .index_out (shift_amount),
        .valid_out (mask_nonzero),
        `UNUSED_PIN(onehot_out)
    );

    wire [COUNT_W-1:0] valid_count;
    VX_popcount #(
        .N(CL_SIZE)
    ) barrel_pc (
        .data_in  (cl_in_byte_mask),
        .data_out (valid_count)
    );

    wire [CL_SIZE*8-1:0] compressed_data = mask_nonzero
        ? (cl_in_data >> {shift_amount, 3'b000})
        : '0;

    // ════════════════════════════════════════════════════════════
    // Sub-B: Fill Buffer — 3-state FSM with per-CL drain
    // ════════════════════════════════════════════════════════════
    // ACCEPT: capture compressed_data and metadata into _r staging regs,
    //         transition to LOAD. Does NOT write fb_data_r (the wide
    //         barrel-shifted load happens in LOAD, one cycle later).
    // LOAD:   shift the registered compressed_data into fb_data_r at
    //         byte_offset position, transfer level/word_addr/byte_offset/
    //         start_word/is_final from staging regs to public regs.
    //         Unconditional single-cycle pass-through to DRAIN.
    // DRAIN:  emit SMEM words. On the LAST drain cycle (drain_will_empty),
    //         simultaneously accept next CL if available into staging regs
    //         and transition to LOAD (one cycle of pipeline latency
    //         between CLs instead of the old zero-bubble overlap).
    //
    // Why the LOAD split exists (Fix #7, 2026-04-09):
    //   The previous 2-state FSM combined (compressed_data >> shift_amount)
    //   AND (shifted_data << byte_offset) AND the fb_data_r load AND the
    //   upstream rc2cs_buf output register into a single cycle. The
    //   resulting 10-level comb cone on ~600 bits was the FPGA critical
    //   path at 300 MHz (~3.29 ns, 73% route). Splitting the load into
    //   its own FB_LOAD state cuts that cone in half.
    //
    // Per-CL address decomposition:
    //   cl_byte_offset = cl_in_smem_byte_addr[SMEM_OFF_W-1:0]
    //   cl_start_word  = cl_in_smem_byte_addr >> SMEM_OFF_W
    //
    // Output (smem_out_valid, smem_out_data) is purely register-driven:
    //   smem_out_valid = (state == DRAIN) && (has_full_word || has_last_partial)
    //   smem_out_data  = fb_data_r[SMEM_WORD_SIZE-1:0]
    // This prevents valid/ready combinational deadlock.

    localparam FB_ACCEPT = 2'd0;
    localparam FB_LOAD   = 2'd1;
    localparam FB_DRAIN  = 2'd2;

    reg [1:0]              fb_state_r;
    reg [FILL_CAP*8-1:0]  fb_data_r /*verilator split_var*/;
    reg [FILL_W-1:0]      fb_level_r;
    reg                    fb_is_final_cl_r;  // true if the last CL in the transfer

    // Per-CL address tracking registers.
    reg [SMEM_ADDR_WIDTH-1:0] fb_word_addr_r;
    reg [SMEM_OFF_W-1:0]      fb_byte_offset_r;
    reg [SMEM_ADDR_WIDTH-1:0] fb_start_word_r;

    // Staging registers captured in FB_ACCEPT (and overlap branch of
    // FB_DRAIN) and consumed by FB_LOAD to drive fb_data_r. The
    // fb_compressed_data_r register stores the post-barrel-shift-right
    // data one cycle before fb_data_r receives the
    // `<< byte_offset` shifted version.
    reg [CL_SIZE*8-1:0]       fb_compressed_data_r /*verilator split_var*/;
    reg [FILL_W-1:0]          fb_load_level_r;
    reg                       fb_load_is_final_cl_r;
    reg [SMEM_OFF_W-1:0]      fb_load_byte_offset_r;
    reg [SMEM_ADDR_WIDTH-1:0] fb_load_start_word_r;

    // ── Output (register-driven, no combinational dependency) ──
    wire has_full_word = (fb_level_r >= FILL_W'(SMEM_WORD_SIZE));
    // Per-CL drain: every CL must drain completely, including partial last word.
    // has_last_partial fires for ANY CL's trailing partial word, not just the final CL.
    wire has_last_partial = !has_full_word && (fb_level_r > 0);
    assign smem_out_valid = (fb_state_r == FB_DRAIN) && (has_full_word || has_last_partial);
    assign smem_out_data  = fb_data_r[SMEM_WORD_SIZE*8-1:0];
    assign smem_out_word_addr = fb_word_addr_r;

    // Byteen generation:
    // - First word of CL: mask out bytes [0..byte_offset-1] (these are before our data)
    // - Last partial word: mask out bytes beyond remaining level
    // - Middle words: all 1s
    wire is_first_word = (fb_word_addr_r == fb_start_word_r);
    /* verilator lint_off CMPCONST */
    for (genvar i = 0; i < SMEM_WORD_SIZE; ++i) begin : g_byteen
        wire byte_has_data   = (FILL_W'(i) < fb_level_r);
        wire byte_is_offset  = is_first_word && (SMEM_OFF_W'(i) < fb_byte_offset_r);
        assign smem_out_byteen[i] = byte_has_data && !byte_is_offset;
    end
    /* verilator lint_on CMPCONST */

    wire drain_will_empty = (fb_level_r <= FILL_W'(SMEM_WORD_SIZE));
    assign smem_out_last  = fb_is_final_cl_r && drain_will_empty;

    wire drain_fire = smem_out_valid && smem_out_ready;

    // ── Input acceptance ──
    // ACCEPT state: always accept (room guaranteed by FILL_CAP design).
    // DRAIN state: accept on the last drain cycle (overlap) when drain fires.
    wire accept_in_accept = (fb_state_r == FB_ACCEPT) && !fb_is_final_cl_r;
    wire accept_in_drain  = (fb_state_r == FB_DRAIN) && drain_will_empty
                          && drain_fire && !fb_is_final_cl_r;

    assign cl_in_ready = accept_in_accept || accept_in_drain;
    wire   cl_in_fire  = cl_in_valid && cl_in_ready;

    // ── Per-CL address decomposition (combinatorial from input) ──
    wire [SMEM_OFF_W-1:0]     cl_byte_offset = cl_in_smem_byte_addr[SMEM_OFF_W-1:0];
    wire [SMEM_ADDR_WIDTH-1:0] cl_start_word  = SMEM_ADDR_WIDTH'(cl_in_smem_byte_addr >> SMEM_OFF_W);

    // ── New level calculation ──
    wire [FILL_W-1:0] new_level_accept = FILL_W'(cl_byte_offset) + FILL_W'(valid_count);

    // Bit offset for appending compressed data to fill buffer at byte_offset
    // position. With the LOAD split (Fix #7), the wide `<< byte_offset` shift
    // runs from REGISTERED fb_load_byte_offset_r in the FB_LOAD state, not
    // from the live combinational cl_byte_offset. Using the registered
    // version is the whole point of the register cut.
    wire [FILL_W+2:0] fb_load_bit_offset = {FILL_W'(fb_load_byte_offset_r), 3'b000};

    // Always-clock fb_data_r through an explicit next-state mux so Vivado
    // does not materialize a high-fanout CE from the FB_LOAD / FB_DRAIN
    // decode onto the entire ~700-bit bank.
    wire fb_data_load_fire  = (fb_state_r == FB_LOAD);
    wire fb_data_clear_fire = (fb_state_r == FB_DRAIN) && drain_fire
                            && drain_will_empty && !cl_in_fire;
    wire fb_data_shift_fire = (fb_state_r == FB_DRAIN) && drain_fire
                            && !drain_will_empty;
    wire [FILL_CAP*8-1:0] fb_data_load_next =
        (FILL_CAP*8)'(fb_compressed_data_r) << fb_load_bit_offset;
    wire [FILL_CAP*8-1:0] fb_data_shift_next = fb_data_r >> (SMEM_WORD_SIZE * 8);
    wire [FILL_CAP*8-1:0] fb_data_next = fb_data_load_fire  ? fb_data_load_next
                                     : fb_data_clear_fire ? '0
                                     : fb_data_shift_fire ? fb_data_shift_next
                                     : fb_data_r;

    // ── Registered state update ──
    always @(posedge clk) begin
        if (reset || start) begin
            fb_state_r        <= FB_ACCEPT;
            fb_data_r         <= '0;
            fb_level_r        <= '0;
            fb_is_final_cl_r  <= 1'b0;
            fb_word_addr_r    <= '0;
            fb_byte_offset_r  <= '0;
            fb_start_word_r   <= '0;
            fb_compressed_data_r  <= '0;
            fb_load_level_r       <= '0;
            fb_load_is_final_cl_r <= 1'b0;
            fb_load_byte_offset_r <= '0;
            fb_load_start_word_r  <= '0;
        end else begin
            fb_data_r <= fb_data_next;
            case (fb_state_r)
            FB_ACCEPT: begin
                if (cl_in_fire) begin
                    // Stage compressed_data and metadata for the next-cycle LOAD.
                    // Does NOT touch fb_data_r here — that happens in FB_LOAD.
                    fb_compressed_data_r  <= compressed_data;
                    fb_load_level_r       <= new_level_accept;
                    fb_load_is_final_cl_r <= cl_in_last;
                    fb_load_byte_offset_r <= cl_byte_offset;
                    fb_load_start_word_r  <= cl_start_word;
                    fb_state_r            <= FB_LOAD;
                end
            end
            FB_LOAD: begin
                // One-cycle register boundary between barrel-compress and
                // the wide `<< byte_offset` load. Transfers staging regs
                // into the public fb_* regs and transitions to DRAIN.
                fb_level_r       <= fb_load_level_r;
                fb_is_final_cl_r <= fb_load_is_final_cl_r;
                fb_word_addr_r   <= fb_load_start_word_r;
                fb_byte_offset_r <= fb_load_byte_offset_r;
                fb_start_word_r  <= fb_load_start_word_r;
                fb_state_r       <= FB_DRAIN;
            end
            FB_DRAIN: begin
                if (drain_fire) begin
                    if (drain_will_empty && cl_in_fire) begin
                        // OVERLAP: drain last word AND accept new CL.
                        // With the LOAD split, overlap now parks the new
                        // CL in staging regs and transitions to FB_LOAD
                        // so the wide shift runs in a clean next cycle.
                        // Old behavior (single-cycle drain+load) is the
                        // exact comb cone that capped Fmax, so keeping
                        // the split here is important.
                        fb_compressed_data_r  <= compressed_data;
                        fb_load_level_r       <= FILL_W'(cl_byte_offset) + FILL_W'(valid_count);
                        fb_load_is_final_cl_r <= cl_in_last;
                        fb_load_byte_offset_r <= cl_byte_offset;
                        fb_load_start_word_r  <= cl_start_word;
                        fb_state_r <= FB_LOAD;
                    end else if (drain_will_empty) begin
                        // Last drain word, no new CL available.
                        fb_state_r <= FB_ACCEPT;
                        fb_level_r <= '0;
                        fb_is_final_cl_r <= 1'b0;
                    end else begin
                        // More words to drain from current CL.
                        fb_level_r <= fb_level_r - FILL_W'(SMEM_WORD_SIZE);
                        fb_word_addr_r <= fb_word_addr_r + SMEM_ADDR_WIDTH'(1);
                    end
                end
            end
            default: fb_state_r <= FB_ACCEPT;
            endcase
        end
    end

    // Idle when in ACCEPT state (fill buffer fully drained, ready for next CL).
    // Note: fb_level_r may be non-zero briefly during the ACCEPT→DRAIN transition
    // cycle, but in ACCEPT state after a drain, level is always 0.
    assign idle = (fb_state_r == FB_ACCEPT);

`ifdef DBG_TRACE_DXA
    wire drain_fire_dbg = smem_out_valid && smem_out_ready;
    always @(posedge clk) begin
        if (~reset && drain_fire_dbg) begin
            $write("DXA_PIPE,%0d,CS_OUT,addr=0x%0h,byteen=0x%0h,level=%0d,last=%0d\n",
                $time, smem_out_word_addr, smem_out_byteen, fb_level_r, smem_out_last);
        end
    end
`endif

endmodule
