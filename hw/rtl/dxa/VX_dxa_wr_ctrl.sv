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

// DXA Write Controller (refactored): thin SMEM write adapter.
// Accepts packed SMEM words with explicit addresses from cl2smem,
// buffers in FIFO, drains to smem_bank_wr_if, tracks completion.
// Sequential address counter removed — addresses come from cl2smem.

`include "VX_define.vh"

/* verilator lint_off UNUSEDPARAM */
module VX_dxa_wr_ctrl import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter WR_QUEUE_DEPTH   = 16,
    parameter SMEM_BYTES       = DXA_SMEM_WORD_SIZE,
    parameter SMEM_DATAW       = SMEM_BYTES * 8,
    parameter SMEM_OFF_BITS    = `CLOG2(SMEM_BYTES),
    parameter SMEM_ADDR_WIDTH  = DXA_SMEM_ADDR_WIDTH
) (
/* verilator lint_on UNUSEDPARAM */
    input  wire                        clk,
    input  wire                        reset,
`ifdef PERF_ENABLE
    output wire [31:0]                 perf_lmem_writes,
`endif
    input  wire                        transfer_active,
    input  wire                        transfer_start,

    // Params from setup.
    input  wire [31:0]                 total_smem_writes,
    input  wire [31:0]                 total_bytes,
    input  wire [`MEM_ADDR_WIDTH-1:0]  initial_smem_base,

    // SMEM word input (from cl2smem, valid/ready).
    input  wire                        smem_in_valid,
    output wire                        smem_in_ready,
    input  wire [SMEM_DATAW-1:0]       smem_in_data,
    input  wire [SMEM_BYTES-1:0]       smem_in_byteen,
    input  wire                        smem_in_last,

    // Explicit SMEM word address from cl2smem.
    input  wire [SMEM_ADDR_WIDTH-1:0]  smem_in_word_addr,

    // SMEM write output.
    output wire                        smem_wr_valid,
    output wire [SMEM_ADDR_WIDTH-1:0]  smem_wr_addr,
    output wire [SMEM_DATAW-1:0]       smem_wr_data,
    output wire [SMEM_BYTES-1:0]       smem_wr_byteen,
    input  wire                        smem_wr_ready,
    output wire                        smem_wr_last_pkt,

    // Completion.
    input  wire                        all_cls_done,    // from rd_ctrl: all CLs emitted
    input  wire                        cl2smem_idle,    // from cl2smem: fill buffer drained
    output wire                        transfer_done,
    output wire [31:0]                 wr_done_count,
    output wire                        smem_req_fire

`ifdef EXT_DXA_MULTICAST_ENABLE
    ,
    input  wire                        is_multicast,
    input  wire [`NUM_WARPS-1:0]       cta_mask,
    input  wire [31:0]                 smem_stride,
    input  wire [31:0]                 bar_stride,
    // Per-CTA done: pulses once for each CTA when its last SMEM word is written.
    output wire                        mc_cta_done,
    output wire [31:0]                 mc_cta_bar_offset
`endif
);
    // ---- Write queue ----
    localparam WRQ_DATAW = 1 + SMEM_ADDR_WIDTH + SMEM_DATAW + SMEM_BYTES;
    localparam WRQ_SIZEW = `CLOG2(WR_QUEUE_DEPTH + 1);

    `STATIC_ASSERT(`IS_POW2(WR_QUEUE_DEPTH), ("WR_QUEUE_DEPTH must be power of 2"))

    wire [WRQ_DATAW-1:0] wrq_data_in;
    wire [WRQ_DATAW-1:0] wrq_data_out;
    wire wrq_empty, wrq_full;
    wire wrq_alm_empty, wrq_alm_full;
    wire [WRQ_SIZEW-1:0] wrq_size;
    wire wrq_push, wrq_pop;

    VX_fifo_queue #(
        .DATAW   (WRQ_DATAW),
        .DEPTH   (WR_QUEUE_DEPTH),
        .OUT_REG (0),
        .LUTRAM  (1)
    ) wr_queue (
        .clk      (clk),
        .reset    (reset),
        .push     (wrq_push),
        .pop      (wrq_pop),
        .data_in  (wrq_data_in),
        .data_out (wrq_data_out),
        .empty    (wrq_empty),
        .alm_empty(wrq_alm_empty),
        .full     (wrq_full),
        .alm_full (wrq_alm_full),
        .size     (wrq_size)
    );

    // ════════════════════════════════════════════════════════════════════
    // Stage 1: Accept SMEM words from cl2smem with explicit addresses
    // ════════════════════════════════════════════════════════════════════

    // Push to write queue with the explicit address from cl2smem.
    assign smem_in_ready = ~wrq_full;
    assign wrq_push      = smem_in_valid && smem_in_ready;
    assign wrq_data_in   = {smem_in_last, smem_in_word_addr, smem_in_data, smem_in_byteen};

    // ════════════════════════════════════════════════════════════════════
    // Stage 2: SMEM write output from queue head
    // ════════════════════════════════════════════════════════════════════

    wire wrq_head_last;
    `UNUSED_VAR (wrq_head_last)
    wire [SMEM_ADDR_WIDTH-1:0] wrq_head_addr;
    wire [SMEM_DATAW-1:0] wrq_head_data;
    wire [SMEM_BYTES-1:0] wrq_head_byteen;

    assign {wrq_head_last, wrq_head_addr, wrq_head_data, wrq_head_byteen} = wrq_data_out;

`ifdef EXT_DXA_MULTICAST_ENABLE
    // ── Multicast replay state machine ──
    // For each queued SMEM word, iterate over all set bits in cta_mask.
    // CTA 0's write goes to the original address; CTA N's write goes to
    // addr + N * smem_stride_words. The write queue only pops after all
    // replay copies for the current word have been issued.
    localparam MC_NW_BITS = `CLOG2(`NUM_WARPS);
    wire [SMEM_ADDR_WIDTH-1:0] smem_stride_words = SMEM_ADDR_WIDTH'(smem_stride >> SMEM_OFF_BITS);

    // Track which CTAs still need a write for the current queue head.
    // Initialized to cta_mask when a new word appears; cleared one bit at a time.
    reg [`NUM_WARPS-1:0] replay_remaining_r;

    // Find the lowest set bit in replay_remaining_r — this is the next CTA to write.
    wire [MC_NW_BITS-1:0] replay_next_idx;
    wire replay_has_remaining;

    VX_priority_encoder #(
        .N (`NUM_WARPS)
    ) replay_pe (
        .data_in   (replay_remaining_r),
        .index_out (replay_next_idx),
        .valid_out (replay_has_remaining),
        `UNUSED_PIN (onehot_out)
    );

    // Compute replayed SMEM address: base + cta_index * stride
    wire [SMEM_ADDR_WIDTH-1:0] replay_addr_offset = SMEM_ADDR_WIDTH'(replay_next_idx) * smem_stride_words;
    wire [SMEM_ADDR_WIDTH-1:0] replay_addr = wrq_head_addr + replay_addr_offset;

    // Is this the last CTA for the current word?
    wire replay_is_last = replay_has_remaining
        && (replay_remaining_r == (`NUM_WARPS'(1) << replay_next_idx));

    // Write output logic
    wire mc_write_valid;
    wire mc_write_fire;
    wire mc_pop;

    if (1) begin : g_mc_out
        // When multicast: write is valid when queue has data AND replay has remaining CTAs.
        // When not multicast: write is valid when queue has data (normal path).
        assign mc_write_valid = transfer_active && ~wrq_empty
                             && (!is_multicast || replay_has_remaining);
        assign mc_write_fire = mc_write_valid && smem_wr_ready;
        // Pop when all CTAs written (multicast: last bit; normal: immediate)
        assign mc_pop = mc_write_fire && (!is_multicast || replay_is_last);
    end

    always @(posedge clk) begin
        if (reset || transfer_start) begin
            replay_remaining_r <= '0;
        end else if (transfer_active && is_multicast) begin
            if (replay_remaining_r == '0 && !wrq_empty) begin
                // New word from queue: load the full cta_mask
                replay_remaining_r <= cta_mask;
            end else if (mc_write_fire && replay_has_remaining) begin
                // Clear the bit we just wrote
                replay_remaining_r <= replay_remaining_r & ~(`NUM_WARPS'(1) << replay_next_idx);
            end
        end
    end

    assign smem_wr_valid    = mc_write_valid;
    assign smem_wr_addr     = is_multicast ? replay_addr : wrq_head_addr;
    assign smem_wr_data     = wrq_head_data;
    assign smem_wr_byteen   = wrq_head_byteen;
    assign wrq_pop          = mc_pop;
    assign smem_wr_last_pkt = mc_pop && wrq_head_last;
    assign smem_req_fire    = mc_write_fire;

    // Per-CTA done signal: when processing the last queue word, each CTA's
    // replay write triggers a done with bar_addr = base + cta_idx * bar_stride.
    // For non-multicast transfers, smem_wr_last_pkt handles done via the tag.
    assign mc_cta_done = is_multicast && mc_write_fire && wrq_head_last;
    assign mc_cta_bar_offset = 32'(replay_next_idx) * bar_stride;

`else
    // ── Normal (non-multicast) path ──
    // Always drain the queue, even between transfers. With per-CL drain,
    // the word count may exceed total_smem_writes, causing transfer_done to
    // fire before the queue is fully drained. Remaining words must still
    // be written to SMEM to prevent queue backup and pipeline deadlock.
    assign smem_wr_valid  = transfer_active && ~wrq_empty;
    assign smem_wr_addr   = wrq_head_addr;
    assign smem_wr_data   = wrq_head_data;
    assign smem_wr_byteen = wrq_head_byteen;
    assign wrq_pop        = smem_wr_valid && smem_wr_ready;
    assign smem_req_fire  = wrq_pop;

`endif


    // ════════════════════════════════════════════════════════════════════
    // Stage 3: Completion Tracking
    // ════════════════════════════════════════════════════════════════════
    // Count SMEM words written. Fire smem_wr_last_pkt when word count
    // reaches total_smem_writes. This is order-invariant -- correct with
    // OOO CL delivery.

    reg [31:0] wr_count_r;

    always @(posedge clk) begin
        if (reset || transfer_start) begin
            wr_count_r <= '0;
        end else if (transfer_active && wrq_pop) begin
            wr_count_r <= wr_count_r + 32'd1;
        end
    end

    assign wr_done_count = wr_count_r;

    // Completion: all three conditions must hold simultaneously:
    //   1. all_cls_done   — rd_ctrl confirms all CLs processed
    //   2. cl2smem_idle   — cl2smem fill buffer fully drained to wr_ctrl
    //   3. wrq_becoming_empty — this is the last SMEM write leaving wr_ctrl
    // Without cl2smem_idle, wrq can become momentarily empty while cl2smem
    // is still draining the last CL, causing premature transfer_done.
    wire wrq_becoming_empty = wrq_pop && (wrq_size == WRQ_SIZEW'(1));
`ifndef EXT_DXA_MULTICAST_ENABLE
    assign smem_wr_last_pkt = wrq_becoming_empty && all_cls_done && cl2smem_idle;
`endif
    assign transfer_done = transfer_active && smem_wr_last_pkt;


    `UNUSED_VAR (wrq_alm_empty)
    `UNUSED_VAR (wrq_alm_full)
    `UNUSED_VAR (total_smem_writes)
    `UNUSED_VAR (initial_smem_base)
    `UNUSED_VAR (total_bytes)
`ifdef EXT_DXA_MULTICAST_ENABLE
    `UNUSED_VAR (total_smem_writes)
`endif

`ifdef PERF_ENABLE
    // Lightweight write counter (no eff_bytes, no span, no back-to-back)
    reg [31:0] wrp_total_lmem_writes_r;
    always @(posedge clk) begin
        if (reset || transfer_start) begin
            wrp_total_lmem_writes_r <= '0;
        end else if (wrq_pop) begin
            wrp_total_lmem_writes_r <= wrp_total_lmem_writes_r + 32'd1;
        end
    end
    assign perf_lmem_writes = wrp_total_lmem_writes_r + 32'(wrq_pop);
`endif

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset) begin
            if (wrq_push) begin
                `TRACE(2, ("%t: wr_ctrl push: addr=0x%0h byteen=0x%0h last=%0b\n",
                    $time, smem_in_word_addr, smem_in_byteen, smem_in_last))
            end
            if (transfer_active && wrq_pop) begin
                `TRACE(2, ("%t: wr_ctrl pop: addr=0x%0h count=%0d last=%0b done=%0b\n",
                    $time, wrq_head_addr, wr_count_r + 32'd1, wrq_head_last, transfer_done))
            end
            // Structured SMEM write event for timeline visualization
            if (smem_req_fire) begin
`ifdef EXT_DXA_MULTICAST_ENABLE
                if (is_multicast) begin
                    // verilator lint_off WIDTHEXPAND
                    $write("DXA_PIPE,%0d,SMEM_WR,addr=0x%0h,byteen=0x%0h,cta=%0d,last=%0d\n",
                        $time, smem_wr_addr, smem_wr_byteen, replay_next_idx, smem_wr_last_pkt);
                    // verilator lint_on WIDTHEXPAND
                end else begin
                    $write("DXA_PIPE,%0d,SMEM_WR,addr=0x%0h,byteen=0x%0h,last=%0d\n",
                        $time, smem_wr_addr, smem_wr_byteen, smem_wr_last_pkt);
                end
`else
                $write("DXA_PIPE,%0d,SMEM_WR,addr=0x%0h,byteen=0x%0h,last=%0d\n",
                    $time, smem_wr_addr, smem_wr_byteen, smem_wr_last_pkt);
`endif
            end
        end
    end
`endif

endmodule
