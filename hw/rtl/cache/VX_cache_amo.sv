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

        reg                          wb_pending_r;
        reg [LINE_ADDR_BITS-1:0]     wb_addr_r;
        reg [WORD_SEL_WIDTH-1:0]     wb_word_idx_r;
        reg [WORD_SIZE-1:0]          wb_byteen_r;
        reg [WORD_WIDTH-1:0]         wb_data_r;
        reg [TAG_WIDTH-1:0]          wb_tag_r;
        reg [REQ_SEL_WIDTH-1:0]      wb_idx_r;
        reg [ATTR_WIDTH-1:0]         wb_attr_r;
        // forwarding window: cache_data updates at the writeback's S0, so an
        // AMO whose S0 read pre-dates it reads stale data for two more cycles.
        reg [1:0]                    post_wb_age;
        wire                         post_wb_valid = (post_wb_age != 2'd0);

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

        // forward an in-flight (or just-fired) writeback on the same line back
        // into the operand, since cache_data has not yet absorbed it.
        wire fwd_active_st1 = (wb_pending_r || post_wb_valid) && (wb_addr_r == addr_st1);
        wire [WORD_WIDTH-1:0] line_word_st1 = fwd_active_st1 ? wb_data_r : read_word_st1;
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

        wire [63:0] new_word;
        wire [63:0] ret_word;
        wire        res_check;

        // commit conditions (from the original AMO at S1; amo_st1.hart_id is
        // valid there, not on the writeback cycle).
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

        VX_amo_unit #(
            .NUM_RES_ENTRIES (NUM_RES_ENTRIES),
            .LINE_ADDR_BITS  (LINE_ADDR_BITS)
        ) amo_unit (
            .clk           (clk),
            .reset         (reset),
            .compute_op    (amo_st1.amo_op),
            .compute_amo_unsigned (amo_st1.amo_unsigned),
            .compute_width (width_st1),
            .compute_old   (old_st1),
            .compute_rhs   (rhs_st1),
            .compute_new_word (new_word),
            .compute_ret_word (ret_word),
            .res_reserve   (res_reserve),
            .res_clear     (res_clear),
            .res_invalidate(res_invalidate),
            .res_hart_id   (amo_st1.hart_id),
            .res_line_addr (addr_st1),
            .res_check     (res_check)
        );

        // place the computed word at the target byte offset within the word
        wire [WORD_WIDTH-1:0] wb_data_w = WORD_WIDTH'(new_word) << bit_off_st1;

        always @(posedge clk) begin
            if (reset) begin
                wb_pending_r <= 1'b0;
                post_wb_age  <= 2'd0;
            end else begin
                if (wb_fire) begin
                    post_wb_age <= 2'd2;
                end else if (post_wb_valid) begin
                    post_wb_age <= post_wb_age - 2'd1;
                end

                if (do_store_st1 && ~pipe_stall && ~wb_pending_r && ~fwd_active_st1) begin
                    // fresh writeback: no in-flight or recent one on this line
                    wb_pending_r  <= 1'b1;
                    wb_addr_r     <= addr_st1;
                    wb_word_idx_r <= word_idx_st1;
                    wb_byteen_r   <= byteen_st1;
                    wb_data_r     <= wb_data_w;
                    wb_tag_r      <= tag_st1;
                    wb_idx_r      <= req_idx_st1;
                    wb_attr_r     <= attr_st1;
                end else if (do_store_st1 && ~pipe_stall && fwd_active_st1) begin
                    // chain onto the in-flight writeback: new_word folded this
                    // AMO in, so overwrite the data and re-arm so it commits.
                    wb_data_r    <= wb_data_w;
                    wb_pending_r <= 1'b1;
                end else if (wb_fire) begin
                    wb_pending_r <= 1'b0;
                end
            end
        end

        // response: SC -> 0 (success) / 1 (fail); other -> old value (LSU sexts).
        wire [63:0] rsp_word = (amo_st1.amo_op == AMO_OP_SC) ? {63'h0, sc_fail_st1} : ret_word;
        if (WORD_WIDTH < 64) begin : g_rsp_upper_unused
            `UNUSED_VAR (rsp_word[63:WORD_WIDTH])
        end

        assign amo_hit_st1 = amo_hit_w;
        assign rsp_data    = WORD_WIDTH'(rsp_word) << bit_off_st1;
        assign commit_busy = wb_pending_r || do_store_st1 || do_store_st0;
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
