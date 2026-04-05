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
    // Sub-B: Fill Buffer — 2-state FSM with per-CL drain
    // ════════════════════════════════════════════════════════════
    // ACCEPT: take CL input, always transition to DRAIN (every CL drains completely).
    // DRAIN:  emit SMEM words. On the LAST drain cycle (drain_will_empty),
    //         simultaneously accept next CL if available (zero bubble).
    //
    // Per-CL address decomposition:
    //   cl_byte_offset = cl_in_smem_byte_addr[SMEM_OFF_W-1:0]
    //   cl_start_word  = cl_in_smem_byte_addr >> SMEM_OFF_W
    //
    // Output (smem_out_valid, smem_out_data) is purely register-driven:
    //   smem_out_valid = (state == DRAIN) && (has_full_word || has_last_partial)
    //   smem_out_data  = fb_data_r[SMEM_WORD_SIZE-1:0]
    // This prevents valid/ready combinational deadlock.

    localparam FB_ACCEPT = 1'b0;
    localparam FB_DRAIN  = 1'b1;

    reg                    fb_state_r;
    reg [FILL_CAP*8-1:0]  fb_data_r /*verilator split_var*/;
    reg [FILL_W-1:0]      fb_level_r;
    reg                    fb_is_final_cl_r;  // true if the last CL in the transfer

    // Per-CL address tracking registers.
    reg [SMEM_ADDR_WIDTH-1:0] fb_word_addr_r;
    reg [SMEM_OFF_W-1:0]      fb_byte_offset_r;
    reg [SMEM_ADDR_WIDTH-1:0] fb_start_word_r;

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

    // Bit offset for appending compressed data to fill buffer at byte_offset position.
    wire [FILL_W+2:0] fb_accept_bit_offset = {FILL_W'(cl_byte_offset), 3'b000};

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
        end else begin
            case (fb_state_r)
            FB_ACCEPT: begin
                if (cl_in_fire) begin
                    // Load compressed data at byte_offset position within the fill buffer.
                    fb_data_r <= (FILL_CAP*8)'(compressed_data) << fb_accept_bit_offset;
                    fb_level_r <= new_level_accept;
                    fb_is_final_cl_r <= cl_in_last;
                    fb_word_addr_r   <= cl_start_word;
                    fb_byte_offset_r <= cl_byte_offset;
                    fb_start_word_r  <= cl_start_word;
                    // Always transition to DRAIN (every CL drains completely).
                    fb_state_r <= FB_DRAIN;
                end
            end
            FB_DRAIN: begin
                if (drain_fire) begin
                    if (drain_will_empty && cl_in_fire) begin
                        // OVERLAP: drain last word of current CL AND accept new CL.
                        fb_data_r  <= (FILL_CAP*8)'(compressed_data) << ({FILL_W'(cl_byte_offset), 3'b000});
                        fb_level_r <= FILL_W'(cl_byte_offset) + FILL_W'(valid_count);
                        fb_is_final_cl_r <= cl_in_last;
                        fb_word_addr_r   <= cl_start_word;
                        fb_byte_offset_r <= cl_byte_offset;
                        fb_start_word_r  <= cl_start_word;
                        // Always drain the new CL too.
                        fb_state_r <= FB_DRAIN;
                    end else if (drain_will_empty) begin
                        // Last drain word, no new CL available.
                        fb_state_r <= FB_ACCEPT;
                        fb_data_r  <= '0;
                        fb_level_r <= '0;
                        fb_is_final_cl_r <= 1'b0;
                    end else begin
                        // More words to drain from current CL.
                        fb_data_r  <= fb_data_r >> (SMEM_WORD_SIZE * 8);
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
