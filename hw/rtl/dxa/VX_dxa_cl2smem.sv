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

// DXA CL-to-SMEM width converter (v2).
//
// General-purpose format converter: works for any CL/SMEM ratio.
// Handles sparse byte masks via prefix-sum scatter.
// Supports initial SMEM byte offset for misaligned launches.
// Uses 2-state FSM (ACCEPT/DRAIN) with overlap: accept fires on the
// last drain cycle to eliminate pipeline bubbles.
//
// Key protocol property: smem_out_valid and smem_out_data are purely
// register-driven (no dependency on smem_out_ready), preventing
// combinational deadlock in the valid/ready handshake.

`include "VX_define.vh"

module VX_dxa_cl2smem import VX_gpu_pkg::*; #(
    parameter CL_SIZE        = `L1_LINE_SIZE,
    parameter SMEM_WORD_SIZE = 16
) (
    input  wire clk,
    input  wire reset,
    input  wire start,

    // Initial byte offset within first SMEM word (from setup).
    input  wire [`CLOG2(SMEM_WORD_SIZE):0] initial_byte_offset,

    // CL input (from rd_ctrl, valid/ready).
    input  wire                        cl_in_valid,
    output wire                        cl_in_ready,
    input  wire [CL_SIZE*8-1:0]        cl_in_data,
    input  wire [CL_SIZE-1:0]          cl_in_byte_mask,
    input  wire                        cl_in_last,

    // SMEM word output (to wr_ctrl, valid/ready).
    output wire                        smem_out_valid,
    input  wire                        smem_out_ready,
    output wire [SMEM_WORD_SIZE*8-1:0] smem_out_data,
    output wire [SMEM_WORD_SIZE-1:0]   smem_out_byteen,
    output wire                        smem_out_last
);
    localparam CL_BITS = `CLOG2(CL_SIZE);
    localparam COUNT_W = CL_BITS + 1;
    localparam FILL_CAP = CL_SIZE + SMEM_WORD_SIZE;
    localparam FILL_W   = `CLOG2(FILL_CAP + 1);

    `STATIC_ASSERT(`IS_POW2(CL_SIZE), ("CL_SIZE must be power of 2"))
    `STATIC_ASSERT(`IS_POW2(SMEM_WORD_SIZE), ("SMEM_WORD_SIZE must be power of 2"))

    // ════════════════════════════════════════════════════════════
    // Sub-A: Prefix-Sum Scatter (combinatorial)
    // ════════════════════════════════════════════════════════════

    /* verilator lint_off UNOPTFLAT */
    wire [COUNT_W-1:0] prefix [CL_SIZE+1];
    assign prefix[0] = '0;
    for (genvar i = 0; i < CL_SIZE; ++i) begin : g_prefix
        assign prefix[i+1] = prefix[i] + COUNT_W'(cl_in_byte_mask[i]);
    end
    wire [COUNT_W-1:0] valid_count = prefix[CL_SIZE];

    wire [CL_SIZE*8-1:0] compressed_data;
    for (genvar out_pos = 0; out_pos < CL_SIZE; ++out_pos) begin : g_scatter
        wire [7:0] or_accum [CL_SIZE+1];
        assign or_accum[0] = 8'd0;
        for (genvar in_pos = 0; in_pos < CL_SIZE; ++in_pos) begin : g_or
            wire hit = cl_in_byte_mask[in_pos]
                     && (prefix[in_pos] == COUNT_W'(out_pos));
            assign or_accum[in_pos+1] = or_accum[in_pos]
                | (hit ? cl_in_data[in_pos*8 +: 8] : 8'd0);
        end
        assign compressed_data[out_pos*8 +: 8] = or_accum[CL_SIZE];
    end
    /* verilator lint_on UNOPTFLAT */

    // ════════════════════════════════════════════════════════════
    // Sub-B: Fill Buffer — 2-state FSM with overlap
    // ════════════════════════════════════════════════════════════
    // ACCEPT: take CL input, transition to DRAIN when fb >= SL.
    // DRAIN:  emit SL words. On the LAST drain cycle (drain_will_empty),
    //         simultaneously accept next CL if available (zero bubble).
    //
    // Output (smem_out_valid, smem_out_data) is purely register-driven:
    //   smem_out_valid = (state == DRAIN)
    //   smem_out_data  = fb_data_r[SL-1:0]
    // This prevents valid/ready combinational deadlock.

    localparam FB_ACCEPT = 1'b0;
    localparam FB_DRAIN  = 1'b1;

    reg                    fb_state_r;
    reg [FILL_CAP*8-1:0]  fb_data_r;
    reg [FILL_W-1:0]      fb_level_r;
    reg                    fb_last_pending_r;
    reg                    fb_first_word_r;  // true until first SL word is emitted

    // ── Output (register-driven, no combinational dependency) ──
    // Only emit when buffer has a full SL word, OR when last pending
    // with any remaining data. Never emit partial words mid-stream.
    wire has_full_word = (fb_level_r >= FILL_W'(SMEM_WORD_SIZE));
    wire has_last_partial = fb_last_pending_r && (fb_level_r > 0);
    assign smem_out_valid = (fb_state_r == FB_DRAIN) && (has_full_word || has_last_partial);
    assign smem_out_data  = fb_data_r[SMEM_WORD_SIZE*8-1:0];

    // Byteen: mark valid bytes, but mask out initial offset bytes on first word.
    // The offset bytes are zeros pre-loaded by initial_byte_offset — they must
    // not be written to SMEM (would overwrite existing data with zeros).
    for (genvar i = 0; i < SMEM_WORD_SIZE; ++i) begin : g_byteen
        wire byte_has_data  = (FILL_W'(i) < fb_level_r);
        wire byte_is_offset = fb_first_word_r
            && (FILL_W'(i) < FILL_W'(initial_byte_offset));
        assign smem_out_byteen[i] = byte_has_data && !byte_is_offset;
    end

    wire drain_will_empty = (fb_level_r <= FILL_W'(SMEM_WORD_SIZE));
    assign smem_out_last  = fb_last_pending_r && drain_will_empty;

    wire drain_fire = smem_out_valid && smem_out_ready;

    // ── Input acceptance ──
    // ACCEPT state: always accept (room guaranteed by FILL_CAP design).
    // DRAIN state: accept on the last drain cycle (overlap) when drain fires.
    wire accept_in_accept = (fb_state_r == FB_ACCEPT) && !fb_last_pending_r;
    wire accept_in_drain  = (fb_state_r == FB_DRAIN) && drain_will_empty
                          && drain_fire && !fb_last_pending_r;

    assign cl_in_ready = accept_in_accept || accept_in_drain;
    wire   cl_in_fire  = cl_in_valid && cl_in_ready;

    // ── New level calculations ──
    wire [FILL_W-1:0] new_level_accept = fb_level_r + FILL_W'(valid_count);
    // Bit offset for appending compressed data to fill buffer.
    wire [FILL_W+2:0] fb_bit_offset = {fb_level_r, 3'b000};

    // ── Registered state update ──
    always @(posedge clk) begin
        if (reset || start) begin
            fb_state_r        <= FB_ACCEPT;
            fb_data_r         <= '0;
            fb_level_r        <= start ? FILL_W'(initial_byte_offset) : '0;
            fb_last_pending_r <= 1'b0;
            fb_first_word_r   <= start ? 1'b1 : 1'b0;
        end else begin
            case (fb_state_r)
            FB_ACCEPT: begin
                if (cl_in_fire) begin
                    fb_data_r <= fb_data_r | ((FILL_CAP*8)'(compressed_data) << fb_bit_offset);
                    fb_level_r <= new_level_accept;
                    fb_last_pending_r <= cl_in_last;
                    if ((new_level_accept >= FILL_W'(SMEM_WORD_SIZE)) || cl_in_last) begin
                        fb_state_r <= FB_DRAIN;
                    end
                end
            end
            FB_DRAIN: begin
                if (drain_fire) begin
                    // Clear first-word flag after emitting any SL word.
                    fb_first_word_r <= 1'b0;

                    if (drain_will_empty && cl_in_fire) begin
                        // OVERLAP: drain last word AND accept new CL.
                        // After drain, residual = 0. Load compressed data at offset 0.
                        fb_data_r  <= (FILL_CAP*8)'(compressed_data);
                        fb_level_r <= FILL_W'(valid_count);
                        fb_last_pending_r <= cl_in_last;
                        if ((FILL_W'(valid_count) >= FILL_W'(SMEM_WORD_SIZE)) || cl_in_last) begin
                            fb_state_r <= FB_DRAIN;
                        end else begin
                            fb_state_r <= FB_ACCEPT;
                        end
                    end else if (drain_will_empty) begin
                        // Last drain word, no new CL available.
                        fb_state_r <= FB_ACCEPT;
                        fb_data_r  <= '0;
                        fb_level_r <= '0;
                        if (fb_last_pending_r) begin
                            fb_last_pending_r <= 1'b0;
                        end
                    end else begin
                        // More words to drain.
                        fb_data_r  <= fb_data_r >> (SMEM_WORD_SIZE * 8);
                        fb_level_r <= fb_level_r - FILL_W'(SMEM_WORD_SIZE);
                    end
                end else if (!smem_out_valid) begin
                    // Residual < SL and more data coming (not last).
                    // Keep residual in buffer, go back to ACCEPT.
                    fb_state_r <= FB_ACCEPT;
                end
            end
            default: fb_state_r <= FB_ACCEPT;
            endcase
        end
    end

`ifdef DBG_TRACE_DXA
    wire drain_fire_dbg = smem_out_valid && smem_out_ready;
    wire cl_in_fire_dbg = cl_in_valid && cl_in_ready;
    always @(posedge clk) begin
        if (~reset) begin
            if (cl_in_fire_dbg) begin
                $write("DXA_PIPE,%0d,CS_IN,mask=0x%0h,vcnt=%0d,level=%0d,last=%0d,data0=0x%0h,cdata0=0x%0h\n",
                    $time, cl_in_byte_mask, valid_count, fb_level_r, cl_in_last,
                    cl_in_data[63:0], compressed_data[63:0]);
            end
            if (drain_fire_dbg) begin
                $write("DXA_PIPE,%0d,CS_OUT,byteen=0x%0h,level=%0d,last=%0d,first_word=%0d,data0=0x%0h,data8=0x%0h\n",
                    $time, smem_out_byteen, fb_level_r, smem_out_last, fb_first_word_r,
                    smem_out_data[63:0], smem_out_data[127:64]);
            end
        end
    end
`endif

endmodule
