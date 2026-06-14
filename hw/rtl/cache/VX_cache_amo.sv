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

`include "VX_cache_define.vh"

// Per-bank AMO engine, instantiated by the bank only when atomics are
// enabled. A bank plays exactly one of two roles:
//
//   IS_LLC=1 (commit): perform the read-modify-write on the line word
//     resident at S1, maintain the per-hart reservation table (via
//     VX_amo_unit), and inject the result back through the bank pipeline
//     as a single-outstanding synthetic writeback.
//
//   IS_LLC=0 (passthrough): forward the AMO downstream (the LLC does the
//     RMW), latch the returned result word, and replay it back to the
//     requester. Also enforces same-hart program order at the bank input.
//
// The two roles are mutually exclusive, so each ties off the other's
// outputs and the synthesizer keeps only the selected datapath.
module VX_cache_amo import VX_gpu_pkg::*; #(
    parameter IS_LLC          = 0,
    parameter NUM_RES_ENTRIES = 4,
    parameter LINE_ADDR_BITS  = 32,
    parameter WORD_WIDTH      = 32,
    parameter WORD_SIZE       = 4,
    parameter WORD_SEL_WIDTH  = 1,
    parameter TAG_WIDTH       = 1,
    parameter REQ_SEL_WIDTH   = 1,
    parameter ATTR_WIDTH      = 1,
    parameter MSHR_SIZE       = 1,
    parameter MSHR_ADDR_WIDTH = 1,
    parameter WORDS_PER_LINE  = 1
) (
    input  wire                          clk,
    input  wire                          reset,
    input  wire                          pipe_stall,

    // pipeline view
    input  amo_req_t                     amo_st0,
    input  wire                          valid_st0,
    input  wire                          is_creq_st0,
    input  wire                          is_hit_st0,
    input  wire                          is_replay_st0,
    input  amo_req_t                     amo_st1,
    input  wire                          valid_st1,
    input  wire                          is_creq_st1,
    input  wire                          is_hit_st1,
    input  wire                          is_replay_st1,
    input  wire                          do_write_st1,
    input  wire [WORD_WIDTH-1:0]         read_word_st1,
    input  wire [WORD_SIZE-1:0]          byteen_st1,
    input  wire [WORD_WIDTH-1:0]         write_word_st1,
    input  wire [WORD_SEL_WIDTH-1:0]     word_idx_st0,
    input  wire [WORD_SEL_WIDTH-1:0]     word_idx_st1,
    input  wire [LINE_ADDR_BITS-1:0]     addr_st0,
    input  wire [LINE_ADDR_BITS-1:0]     addr_st1,
    input  wire [TAG_WIDTH-1:0]          tag_st1,
    input  wire [REQ_SEL_WIDTH-1:0]      req_idx_st1,
    input  wire [ATTR_WIDTH-1:0]         attr_st1,

    // commit handshake: the bank grants the synthetic writeback this cycle
    input  wire                          wb_fire,

    // mshr / memory fill (passthrough)
    input  wire                          mshr_allocate_st0,
    input  wire [MSHR_ADDR_WIDTH-1:0]    mshr_alloc_id_st0,
    input  wire [MSHR_ADDR_WIDTH-1:0]    mshr_id_st1,
    input  wire                          mem_rsp_fire,
    input  wire [MSHR_ADDR_WIDTH-1:0]    mem_rsp_id,
    input  wire [WORDS_PER_LINE*WORD_WIDTH-1:0] mem_rsp_data,
    input  wire                          is_fill_sel,

    // input arbitration (passthrough age-ordering)
    input  wire                          core_req_valid,
    input  wire                          core_req_is_amo,
    input  wire                          core_req_rw,
    input  wire [LINE_ADDR_BITS-1:0]     core_req_addr,
    input  wire                          rw_st0,
    input  wire                          mshr_probe_pending_ld,
    input  wire                          mshr_probe_pending_amo,

    // commit outputs (tied off when IS_LLC=0)
    output wire                          amo_hit_st1,    // AMO commits locally at S1
    output wire                          commit_busy,    // commit in flight
    output wire                          chain_stall,    // pace same-line chained AMO
    output wire                          wb_pending,     // writeback request live
    output wire [WORD_WIDTH-1:0]         rsp_data,       // response word on amo_hit_st1
    output wire [LINE_ADDR_BITS-1:0]     wb_addr,
    output wire [WORD_SEL_WIDTH-1:0]     wb_word_idx,
    output wire [WORD_SIZE-1:0]          wb_byteen,
    output wire [WORD_WIDTH-1:0]         wb_data,
    output wire [TAG_WIDTH-1:0]          wb_tag,
    output wire [REQ_SEL_WIDTH-1:0]      wb_idx,
    output wire [ATTR_WIDTH-1:0]         wb_attr,

    // passthrough outputs (tied off when IS_LLC=1)
    output wire                          is_amo_fwd_st0,    // AMO first pass (S0)
    output wire                          is_amo_fwd_st1,    // AMO first pass (S1)
    output wire                          is_amo_replay_st1, // result replay
    output wire                          is_passthru_fill_sel,
    output wire [WORD_WIDTH-1:0]         amo_ptw_word_st1,
    output wire                          req_input_defer
);
    if (IS_LLC != 0) begin : g_commit
        // ----------------------------------------------------------------
        // LLC commit: RMW on the resident line + synthetic writeback
        // ----------------------------------------------------------------
        localparam BIT_OFF_BITS = `CLOG2(WORD_WIDTH);
        localparam AMO_OLD_BITS = (WORD_WIDTH < 64) ? WORD_WIDTH : 64;

        // Writeback queue (depth 2): a completed AMO pushes its result here
        // instead of overwriting a still-draining different-line writeback. The
        // head (slot 0) drains through the bank's synthetic-write path; pushes
        // never clobber a pending entry, so different-line AMOs pipeline without
        // stalling any replay (coalescer-safe) or the pipe (deadlock-free).
        localparam WBQ_SIZE = 2;
        localparam WBQ_CNTW = `CLOG2(WBQ_SIZE+1);
        localparam WBQ_IDXW = `CLOG2(WBQ_SIZE);
        reg [WBQ_CNTW-1:0]           wbq_count;
        reg [LINE_ADDR_BITS-1:0]     wbq_addr   [WBQ_SIZE];
        reg [WORD_SEL_WIDTH-1:0]     wbq_wsel   [WBQ_SIZE];
        reg [WORD_SIZE-1:0]          wbq_byteen [WBQ_SIZE];
        reg [WORD_WIDTH-1:0]         wbq_data   [WBQ_SIZE];
        reg [TAG_WIDTH-1:0]          wbq_tag    [WBQ_SIZE];
        reg [REQ_SEL_WIDTH-1:0]      wbq_idx    [WBQ_SIZE];
        reg [ATTR_WIDTH-1:0]         wbq_attr   [WBQ_SIZE];

        // Head aliases (slot 0 = oldest = the entry currently draining).
        wire                         wb_pending_r  = (wbq_count != '0);
        wire [LINE_ADDR_BITS-1:0]    wb_addr_r     = wbq_addr[0];
        wire [WORD_SEL_WIDTH-1:0]    wb_word_idx_r = wbq_wsel[0];
        wire [WORD_SIZE-1:0]         wb_byteen_r   = wbq_byteen[0];
        wire [WORD_WIDTH-1:0]        wb_data_r     = wbq_data[0];
        wire [TAG_WIDTH-1:0]         wb_tag_r      = wbq_tag[0];
        wire [REQ_SEL_WIDTH-1:0]     wb_idx_r      = wbq_idx[0];
        wire [ATTR_WIDTH-1:0]        wb_attr_r     = wbq_attr[0];

        // BRAM-settle window: a fired writeback takes a couple cycles to land
        // in cache_data; commit_busy stays high across it so the next AMO reads
        // the committed line. post_wb_{addr,data} hold the just-drained entry.
        reg [1:0]                    post_wb_age;
        reg [LINE_ADDR_BITS-1:0]     post_wb_addr;
        reg [WORD_WIDTH-1:0]         post_wb_data;
        wire                         post_wb_valid = (post_wb_age != 2'd0);

        // Compute stage: S1 latches the aligned operands, the RMW ALU + the
        // re-align shift run the next cycle, off the S1 critical path. AMO
        // commits are serialized by commit_busy (the bank holds off core
        // requests and replays), so the stage holds at most one operation and
        // each AMO reads the freshly written line (no operand forwarding).
        reg                          cmp_valid;
        reg [63:0]                   cmp_old, cmp_rhs;
        amo_op_e                     cmp_op;
        reg [1:0]                    cmp_width;
        reg                          cmp_unsigned;
        reg [BIT_OFF_BITS-1:0]       cmp_bit_off;
        reg [LINE_ADDR_BITS-1:0]     cmp_addr;
        reg [WORD_SIZE-1:0]          cmp_byteen;
        reg [WORD_SEL_WIDTH-1:0]     cmp_wsel;
        reg [TAG_WIDTH-1:0]          cmp_tag;
        reg [REQ_SEL_WIDTH-1:0]      cmp_idx;
        reg [ATTR_WIDTH-1:0]         cmp_attr;

        // Byte-offset alignment: shift the target down to bit 0 for compute,
        // and shift results back for response/writeback.
        wire [`UP(`CLOG2(WORD_SIZE))-1:0] byte_off_st1;
        VX_priority_encoder #(
            .N (WORD_SIZE)
        ) byte_off_enc (
            .data_in    (byteen_st1),
            .index_out  (byte_off_st1),
            `UNUSED_PIN (valid_out),
            `UNUSED_PIN (onehot_out)
        );
        wire [BIT_OFF_BITS-1:0] bit_off_st1 = BIT_OFF_BITS'({byte_off_st1, 3'b0});

        // Forward an in-flight (or just-fired) writeback on the same line back
        // into the operand: chained same-line AMOs are paced one cycle apart
        // (chain_stall) so the prior result already sits in wb_data_r while
        // read_word_st1 may still be stale.
        // Forward the newest in-flight value for this line: scan the queue
        // newest-first, then the just-drained (settling) entry, else the array.
        wire fwd_q1 = (wbq_count > 1) && (wbq_addr[1] == addr_st1);
        wire fwd_q0 = (wbq_count > 0) && (wbq_addr[0] == addr_st1);
        wire fwd_pw = post_wb_valid && (post_wb_addr == addr_st1);
        wire [WORD_WIDTH-1:0] line_word_st1 = fwd_q1 ? wbq_data[1]
                                            : fwd_q0 ? wbq_data[0]
                                            : fwd_pw ? post_wb_data
                                            : read_word_st1;
        wire [WORD_WIDTH-1:0] line_word_shifted_st1 = line_word_st1 >> bit_off_st1;
        wire [WORD_WIDTH-1:0] rhs_word_shifted_st1  = write_word_st1 >> bit_off_st1;

        // width from byteen popcount (.W -> 4 bytes, .D -> 8); operands top at .D.
        wire [1:0] width_st1 = ($countones(byteen_st1) == 8) ? 2'd3 : 2'd2;
        wire [63:0] rhs_st1 = (width_st1 == 2'd2)
                            ? 64'({32'h0, rhs_word_shifted_st1[31:0]})
                            : 64'(rhs_word_shifted_st1[AMO_OLD_BITS-1:0]);
        wire [63:0] old_st1 = (width_st1 == 2'd2)
                            ? 64'({32'h0, line_word_shifted_st1[31:0]})
                            : 64'(line_word_shifted_st1[AMO_OLD_BITS-1:0]);
        if (WORD_WIDTH > 64) begin : g_upper_unused
            `UNUSED_VAR (line_word_shifted_st1[WORD_WIDTH-1:64])
            `UNUSED_VAR (rhs_word_shifted_st1[WORD_WIDTH-1:64])
        end

        wire        res_check;

        // commit conditions (from the original AMO at S1; amo_st1.hart_id is
        // valid there, not on the compute/writeback cycle).
        wire amo_hit_w = amo_st1.amo_valid && is_hit_st1 && valid_st1 && is_creq_st1;
        wire sc_fail_st1 = (amo_st1.amo_op == AMO_OP_SC) && ~res_check;
        wire do_store_st1 = amo_hit_w && (amo_st1.amo_op != AMO_OP_LR) && ~sc_fail_st1;
        wire do_store_st0 = amo_st0.amo_valid && valid_st0 && is_creq_st0 && is_hit_st0
                         && (amo_st0.amo_op != AMO_OP_LR);

        wire res_reserve    = amo_hit_w && (amo_st1.amo_op == AMO_OP_LR);
        wire res_clear      = amo_hit_w && (amo_st1.amo_op == AMO_OP_SC);
        // any committed write to the line breaks other harts' reservations;
        // AMOs ride the load path (rw=0) so do_write_st1 is plain stores only.
        wire res_invalidate = do_store_st1 || do_write_st1;

        // RMW ALU runs on the registered compute-stage operands (off the S1
        // path); the reservation table is driven from S1 so the SC outcome is
        // ready for the response. ret_word is unused — the response old value
        // comes straight from S1 (no ALU).
        wire [63:0] new_word;
        wire [63:0] ret_word_unused;
        VX_amo_unit #(
            .NUM_RES_ENTRIES (NUM_RES_ENTRIES),
            .LINE_ADDR_BITS  (LINE_ADDR_BITS),
            .DATA_WIDTH      (AMO_OLD_BITS)  // 32-bit word cache -> 32-bit RMW datapath
        ) amo_unit (
            .clk           (clk),
            .reset         (reset),
            .compute_op    (cmp_op),
            .compute_unsigned (cmp_unsigned),
            .compute_width (cmp_width),
            .compute_old   (cmp_old),
            .compute_rhs   (cmp_rhs),
            .compute_new_word (new_word),
            .compute_ret_word (ret_word_unused),
            .res_reserve   (res_reserve),
            .res_clear     (res_clear),
            .res_invalidate(res_invalidate),
            .res_hart_id   (amo_st1.hart_id),
            .res_line_addr (addr_st1),
            .res_check     (res_check)
        );
        `UNUSED_VAR (ret_word_unused)

        // place the computed word at its byte offset within the cache word
        wire [WORD_WIDTH-1:0] wb_data_w = WORD_WIDTH'(new_word) << cmp_bit_off;

        // Compute finished this cycle (result ready to enqueue): the compute
        // stage is occupied and not being reloaded by a fresh latch.
        wire wb_push = cmp_valid && ~(do_store_st1 && ~pipe_stall);
        // A same-line result coalesces into its existing entry (only the latest
        // value must reach the array; earlier ones are forwarded), so a same-line
        // burst stays at a single entry. A new-line result enqueues at the tail.
        // The head cannot be coalesced into the cycle it drains.
        wire wb_coal0    = (wbq_count > 0) && (wbq_addr[0] == cmp_addr) && ~wb_fire;
        wire wb_coal1    = (wbq_count > 1) && (wbq_addr[1] == cmp_addr);
        wire wb_coalesce = wb_coal1 || wb_coal0;
        wire [WBQ_IDXW-1:0] wb_coal_idx = wb_coal1 ? WBQ_IDXW'(1) : WBQ_IDXW'(0);
        // New entry lands at the post-pop tail; a coalesce slot shifts down on a pop.
        wire [WBQ_IDXW-1:0] wb_new_idx  = WBQ_IDXW'(wb_fire ? (wbq_count - WBQ_CNTW'(1)) : wbq_count);
        wire [WBQ_IDXW-1:0] wb_slot     = wb_coalesce ? WBQ_IDXW'(wb_fire ? (wb_coal_idx - WBQ_IDXW'(1)) : wb_coal_idx)
                                                      : wb_new_idx;

        always @(posedge clk) begin
            if (reset) begin
                cmp_valid   <= 1'b0;
                wbq_count   <= '0;
                post_wb_age <= 2'd0;
            end else begin
                if (wb_fire) begin
                    post_wb_age  <= 2'd2;
                    post_wb_addr <= wbq_addr[0];
                    post_wb_data <= wbq_data[0];
                end else if (post_wb_valid) begin
                    post_wb_age <= post_wb_age - 2'd1;
                end

                // Compute stage (single): latch a new AMO, else retire the result.
                if (do_store_st1 && ~pipe_stall) begin
                    cmp_valid    <= 1'b1;
                    cmp_old      <= old_st1;
                    cmp_rhs      <= rhs_st1;
                    cmp_op       <= amo_st1.amo_op;
                    cmp_width    <= width_st1;
                    cmp_unsigned <= amo_st1.amo_unsigned;
                    cmp_bit_off  <= bit_off_st1;
                    cmp_addr     <= addr_st1;
                    cmp_byteen   <= byteen_st1;
                    cmp_wsel     <= word_idx_st1;
                    cmp_tag      <= tag_st1;
                    cmp_idx      <= req_idx_st1;
                    cmp_attr     <= attr_st1;
                end else if (cmp_valid) begin
                    cmp_valid <= 1'b0;
                end

                // Writeback queue: a drain (wb_fire) shifts the head out; a
                // completed compute (wb_push) enqueues at the tail. The push is
                // written after the shift so it wins when both hit the same slot.
                if (wb_fire) begin
                    wbq_addr[0]   <= wbq_addr[1];
                    wbq_wsel[0]   <= wbq_wsel[1];
                    wbq_byteen[0] <= wbq_byteen[1];
                    wbq_data[0]   <= wbq_data[1];
                    wbq_tag[0]    <= wbq_tag[1];
                    wbq_idx[0]    <= wbq_idx[1];
                    wbq_attr[0]   <= wbq_attr[1];
                end
                if (wb_push) begin
                    wbq_addr[wb_slot]   <= cmp_addr;
                    wbq_wsel[wb_slot]   <= cmp_wsel;
                    wbq_byteen[wb_slot] <= cmp_byteen;
                    wbq_data[wb_slot]   <= wb_data_w;
                    wbq_tag[wb_slot]    <= cmp_tag;
                    wbq_idx[wb_slot]    <= cmp_idx;
                    wbq_attr[wb_slot]   <= cmp_attr;
                end
                // Count grows only on a new (non-coalescing) enqueue; a coalesce
                // updates in place. Pop removes the head.
                if (wb_push && ~wb_coalesce && ~wb_fire)
                    wbq_count <= wbq_count + WBQ_CNTW'(1);
                else if (~(wb_push && ~wb_coalesce) && wb_fire)
                    wbq_count <= wbq_count - WBQ_CNTW'(1);
            end
        end

        // response (fired at S1): SC -> 0/1; other -> old value (LSU sexts).
        // The old value is available at S1 directly, no ALU needed.
        wire [63:0] rsp_word = (amo_st1.amo_op == AMO_OP_SC) ? {63'h0, sc_fail_st1} : old_st1;
        if (WORD_WIDTH < 64) begin : g_rsp_upper_unused
            `UNUSED_VAR (rsp_word[63:WORD_WIDTH])
        end

        assign amo_hit_st1 = amo_hit_w;
        assign rsp_data    = WORD_WIDTH'(rsp_word) << bit_off_st1;
        // Commit in flight: holds off new core-request admission from the S0
        // prediction through the compute stage and the writeback. Replays are
        // NOT blocked (the MSHR streams coalesced same-line AMOs back to back);
        // those are paced instead by chain_stall.
        assign commit_busy = do_store_st0 || do_store_st1 || cmp_valid || wb_pending_r;
        // Pace any same-line request sitting behind an in-flight compute by one
        // cycle, so the result lands in wb_data_r and forwards cleanly. Gated on
        // cmp_valid (an AMO is computing), so it never fires for baseline traffic.
        assign chain_stall = cmp_valid && valid_st1 && is_creq_st1 && (cmp_addr == addr_st1);

        // Invariants: a store-bearing AMO is only ever accepted into a free
        // compute stage (the queue absorbs different-line writebacks behind it),
        // and the writeback queue must never overflow.
        `RUNTIME_ASSERT (~(do_store_st1 && ~pipe_stall && cmp_valid),
            ("%t: AMO compute-stage overwrite (addr=0x%0h)", $time, addr_st1))
        `RUNTIME_ASSERT (~(wb_push && ~wb_coalesce && ~wb_fire && (wbq_count == WBQ_CNTW'(WBQ_SIZE))),
            ("%t: AMO writeback queue overflow (addr=0x%0h)", $time, cmp_addr))
        assign wb_pending  = wb_pending_r;
        assign wb_addr     = wb_addr_r;
        assign wb_word_idx = wb_word_idx_r;
        assign wb_byteen   = wb_byteen_r;
        assign wb_data     = wb_data_r;
        assign wb_tag      = wb_tag_r;
        assign wb_idx      = wb_idx_r;
        assign wb_attr     = wb_attr_r;

        // passthrough outputs unused in this role
        assign is_amo_fwd_st0       = 1'b0;
        assign is_amo_fwd_st1       = 1'b0;
        assign is_amo_replay_st1    = 1'b0;
        assign is_passthru_fill_sel = 1'b0;
        assign amo_ptw_word_st1     = '0;
        assign req_input_defer      = 1'b0;

        `UNUSED_VAR (amo_st0) // only amo_valid/amo_op are consumed at S0
        `UNUSED_VAR (is_replay_st0)
        `UNUSED_VAR (is_replay_st1)
        `UNUSED_VAR (word_idx_st0)
        `UNUSED_VAR (addr_st0)
        `UNUSED_VAR (mshr_allocate_st0)
        `UNUSED_VAR (mshr_alloc_id_st0)
        `UNUSED_VAR (mshr_id_st1)
        `UNUSED_VAR (mem_rsp_fire)
        `UNUSED_VAR (mem_rsp_id)
        `UNUSED_VAR (mem_rsp_data)
        `UNUSED_VAR (is_fill_sel)
        `UNUSED_VAR (core_req_valid)
        `UNUSED_VAR (core_req_is_amo)
        `UNUSED_VAR (core_req_rw)
        `UNUSED_VAR (core_req_addr)
        `UNUSED_VAR (rw_st0)
        `UNUSED_VAR (mshr_probe_pending_ld)
        `UNUSED_VAR (mshr_probe_pending_amo)
    end else begin : g_passthru
        // ----------------------------------------------------------------
        // Non-LLC passthrough: forward downstream, replay the result word
        // ----------------------------------------------------------------
        assign is_amo_fwd_st0    = amo_st0.amo_valid && valid_st0 && is_creq_st0 && ~is_replay_st0;
        assign is_amo_fwd_st1    = amo_st1.amo_valid && valid_st1 && is_creq_st1 && ~is_replay_st1;
        assign is_amo_replay_st1 = amo_st1.amo_valid && valid_st1 && is_creq_st1 && is_replay_st1;

        reg [MSHR_SIZE-1:0]      ptw_flag;   // entry awaits a passthru fill
        reg [WORD_SEL_WIDTH-1:0] ptw_wsel [MSHR_SIZE];
        reg [WORD_WIDTH-1:0]     ptw_word [MSHR_SIZE];

        wire [WORDS_PER_LINE-1:0][WORD_WIDTH-1:0] mem_rsp_words = mem_rsp_data;

        assign is_passthru_fill_sel = is_fill_sel && ptw_flag[mem_rsp_id];
        assign amo_ptw_word_st1     = ptw_word[mshr_id_st1];

        always @(posedge clk) begin
            if (reset) begin
                ptw_flag <= '0;
            end else begin
                // mark the AMO's MSHR entry on allocation
                if (is_amo_fwd_st0 && mshr_allocate_st0 && ~pipe_stall) begin
                    ptw_flag[mshr_alloc_id_st0] <= 1'b1;
                    ptw_wsel[mshr_alloc_id_st0] <= word_idx_st0;
                end
                // latch the result word on the passthru fill, clear the flag
                if (mem_rsp_fire && ptw_flag[mem_rsp_id]) begin
                    ptw_word[mem_rsp_id] <= mem_rsp_words[ptw_wsel[mem_rsp_id]];
                    ptw_flag[mem_rsp_id] <= 1'b0;
                end
            end
        end

        // catch a same-line request mid-allocation at S0 (not yet visible to
        // the MSHR probe in the window between admit and allocate).
        wire alloc_same_line = mshr_allocate_st0 && ~pipe_stall && (addr_st0 == core_req_addr);
        wire st0_ld_alloc    = alloc_same_line && ~amo_st0.amo_valid && ~rw_st0;
        wire st0_amo_alloc   = alloc_same_line &&  amo_st0.amo_valid;

        wire amo_input_defer  = core_req_valid && core_req_is_amo
                             && (mshr_probe_pending_ld || st0_ld_alloc);
        wire load_input_defer = core_req_valid && ~core_req_is_amo && ~core_req_rw
                             && (mshr_probe_pending_amo || st0_amo_alloc);
        assign req_input_defer = amo_input_defer || load_input_defer;

        // commit outputs unused in this role
        assign amo_hit_st1 = 1'b0;
        assign commit_busy = 1'b0;
        assign chain_stall = 1'b0;
        assign wb_pending  = 1'b0;
        assign rsp_data    = '0;
        assign wb_addr     = '0;
        assign wb_word_idx = '0;
        assign wb_byteen   = '0;
        assign wb_data     = '0;
        assign wb_tag      = '0;
        assign wb_idx      = '0;
        assign wb_attr     = '0;

        `UNUSED_VAR (amo_st0) // only amo_valid gates the passthru path
        `UNUSED_VAR (amo_st1)
        `UNUSED_VAR (is_hit_st0)
        `UNUSED_VAR (is_hit_st1)
        `UNUSED_VAR (do_write_st1)
        `UNUSED_VAR (read_word_st1)
        `UNUSED_VAR (byteen_st1)
        `UNUSED_VAR (write_word_st1)
        `UNUSED_VAR (word_idx_st1)
        `UNUSED_VAR (addr_st1)
        `UNUSED_VAR (tag_st1)
        `UNUSED_VAR (req_idx_st1)
        `UNUSED_VAR (attr_st1)
        `UNUSED_VAR (wb_fire)
    end

endmodule
