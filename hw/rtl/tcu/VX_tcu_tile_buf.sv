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

`include "VX_define.vh"

//
// WGMMA tile stream-buffer (per-warp slot cache).
//
// Prefetches A, B, and optional sparse metadata tiles from local memory
// using a dedicated bank-parallel read port that bypasses the LSU crossbar.
// All LMEM banks are read simultaneously each cycle, yielding NUM_BANKS
// words/cycle throughput.
//
// Each active warp gets its own slot (direct-mapped by wid LSBs).  The
// sender FSM scans for the lowest slot that is valid but not yet fetched
// and services it.  Slots are allocated on the first µop of a new tile
// (step_m==step_n==step_k==0) and evicted implicitly when the same slot
// index is re-allocated for a different descriptor.
//
// Metadata extraction for WGMMA_SP is performed here (moved from
// VX_tcu_sp_mux) so the per-(i,j) FEDP instances only see a narrow
// vld_meta_block slice identical to what VX_tcu_meta produces for WMMA_SP.
//
// Assumptions:
//   - Tile base addresses (descriptor[15:0]) are bank-aligned.
//   - Tiles are stored row-major and packed (ldm == tile column count).
//

module VX_tcu_tile_buf import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID     = "",
    parameter         TCU_TBUF_SIZE   = `NUM_WARPS,
    parameter         NUM_BANKS       = 4,
    parameter         BANK_ADDR_WIDTH = 12
) (
    input  wire clk,
    input  wire reset,

`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] tbuf_fetch_stalls,
`endif

    // Execute-side observation
    input  wire                     req_valid,
    input  wire [NW_WIDTH-1:0]      req_wid,
    input  wire                     req_is_sparse,
    input  wire [3:0]               req_step_m,
    input  wire [3:0]               req_step_n,
    input  wire [3:0]               req_step_k,
    input  wire [3:0]               req_fmt_s,
    input  wire [`XLEN-1:0]         req_desc_a,
    input  wire [`XLEN-1:0]         req_desc_b,

    // LMEM read port
    VX_tcu_lmem_if.master           tcu_lmem_if,

    // Tile buffer outputs
    output wire [TCU_BLOCK_CAP-1:0][`XLEN-1:0] tbuf_rs1_data,
    output wire [TCU_BLOCK_CAP-1:0][`XLEN-1:0] tbuf_rs2_data,
`ifdef TCU_SPARSE_ENABLE
    output wire [TCU_MAX_META_BLOCK_WIDTH-1:0] tbuf_sp_meta,
`endif
    output wire                     tbuf_ready
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // -----------------------------------------------------------------------
    // Derived constants
    // -----------------------------------------------------------------------

    localparam SLOT_W        = `UP($clog2(TCU_TBUF_SIZE));
    localparam BANK_SEL_BITS = $clog2(NUM_BANKS);
    localparam LMEM_ADDR_WIDTH = BANK_ADDR_WIDTH + BANK_SEL_BITS;

    // WG tile dimensions (fp32-word level)
    localparam TILE_M = TCU_WG_TILE_M;
    localparam TILE_N = TCU_WG_TILE_N;
    localparam TILE_K = TCU_WG_TILE_K;

    localparam A_TOTAL    = TILE_M * TILE_K;
    localparam B_TOTAL    = TILE_K * TILE_N;

    // Bank-row counts for each tile phase
    localparam A_BANK_ROWS = (A_TOTAL + NUM_BANKS - 1) / NUM_BANKS;
    localparam B_BANK_ROWS = (B_TOTAL + NUM_BANKS - 1) / NUM_BANKS;

`ifdef TCU_SPARSE_ENABLE
    localparam A_TOTAL_SP    = A_TOTAL / 2;    // 2:4 compressed A has half the K columns
    localparam A_BANK_ROWS_SP = (A_TOTAL_SP + NUM_BANKS - 1) / NUM_BANKS;

    // Per-format metadata stride (words per wg_bank slot)
    localparam SP_I_RATIO_16B = 2;
    localparam SP_I_RATIO_8B  = 4;

    localparam META_ROW_BITS_16B = TCU_TC_K * 2 * SP_I_RATIO_16B;
    localparam META_ROW_BITS_8B  = TCU_TC_K * 2 * SP_I_RATIO_8B;

    localparam META_STRIDE_16B = (TCU_TC_M * META_ROW_BITS_16B + 31) / 32;
    localparam META_STRIDE_8B  = (TCU_TC_M * META_ROW_BITS_8B  + 31) / 32;

    localparam META_STRIDE_MAX = META_STRIDE_8B; // worst case

    localparam WG_HALF_K       = TCU_WG_K_STEPS / 2;
    localparam WG_META_BANKS   = TCU_WG_M_STEPS * WG_HALF_K;

    // Per-format total metadata words
    localparam META_TOTAL_16B  = WG_META_BANKS * META_STRIDE_16B;
    localparam META_TOTAL_8B   = WG_META_BANKS * META_STRIDE_8B;
    localparam META_TOTAL_MAX  = META_TOTAL_8B;

    // Bank-row counts for metadata
    localparam META_ROWS_16B = (META_TOTAL_16B + NUM_BANKS - 1) / NUM_BANKS;
    localparam META_ROWS_8B  = (META_TOTAL_8B  + NUM_BANKS - 1) / NUM_BANKS;
    localparam META_ROWS_MAX = META_ROWS_8B;
`else
    localparam META_TOTAL_MAX  = 1;
    localparam META_ROWS_MAX   = 1;
    `UNUSED_PARAM (META_TOTAL_MAX)
`endif

    localparam MAX_BANK_ROWS = (A_BANK_ROWS > B_BANK_ROWS)
                             ? (A_BANK_ROWS > META_ROWS_MAX ? A_BANK_ROWS : META_ROWS_MAX)
                             : (B_BANK_ROWS > META_ROWS_MAX ? B_BANK_ROWS : META_ROWS_MAX);
    localparam FETCH_CTR_W   = `CLOG2(MAX_BANK_ROWS + 1);

    // Output mux lane offsets (same as VX_tcu_core)
    localparam LG_A_BS = $clog2(TCU_A_BLOCK_SIZE);
    localparam LG_B_BS = $clog2(TCU_B_BLOCK_SIZE);
    localparam OFF_W   = $clog2(TCU_BLOCK_CAP);

`ifdef TCU_SPARSE_ENABLE
    localparam LG_B_BS_SP = $clog2(TCU_B_BLOCK_SIZE_SP);
`endif

    // -----------------------------------------------------------------------
    // is_sparse helper
    // -----------------------------------------------------------------------

`ifdef TCU_SPARSE_ENABLE
    wire is_sparse = req_is_sparse;
`else
    wire is_sparse = 1'b0;
    `UNUSED_VAR (is_sparse)
    `UNUSED_VAR (req_is_sparse)
`endif

    // -----------------------------------------------------------------------
    // Per-warp slot arrays
    // -----------------------------------------------------------------------

    logic                       slot_valid      [TCU_TBUF_SIZE];
    logic [NW_WIDTH-1:0]        slot_wid        [TCU_TBUF_SIZE];
    logic [`XLEN-1:0]           slot_desc_a     [TCU_TBUF_SIZE];
    logic [`XLEN-1:0]           slot_desc_b     [TCU_TBUF_SIZE];
    logic                       slot_fetch_done [TCU_TBUF_SIZE];
    // One-shot alloc suppressor: set when alloc_en fires, cleared when
    // the first uop fires (tbuf_ready && req_valid && is_first_uop).
    // Prevents alloc_en from re-triggering every cycle after a fetch
    // completes, which would cause a livelock.
    logic                       warp_alloc_pending [TCU_TBUF_SIZE];
    logic [BANK_ADDR_WIDTH-1:0] slot_a_row_base [TCU_TBUF_SIZE];
    logic [BANK_ADDR_WIDTH-1:0] slot_b_row_base [TCU_TBUF_SIZE];

    logic [31:0] slot_a_buf [TCU_TBUF_SIZE][0:A_TOTAL-1];
    logic [31:0] slot_b_buf [TCU_TBUF_SIZE][0:B_TOTAL-1];

`ifdef TCU_SPARSE_ENABLE
    logic [BANK_ADDR_WIDTH-1:0] slot_meta_row_base  [TCU_TBUF_SIZE];
    logic [FETCH_CTR_W-1:0]     slot_meta_bank_rows [TCU_TBUF_SIZE];
    logic [3:0]                 slot_meta_stride    [TCU_TBUF_SIZE];
    logic                       slot_is_sparse      [TCU_TBUF_SIZE];
    logic [31:0]                slot_meta_buf [TCU_TBUF_SIZE][0:META_TOTAL_MAX-1];
`endif

    // -----------------------------------------------------------------------
    // Sender FSM states
    // -----------------------------------------------------------------------

    typedef enum logic [1:0] {
        SEND_IDLE     = 2'd0,
        SEND_FETCH_A  = 2'd1,
        SEND_FETCH_B  = 2'd2,
        SEND_FETCH_META = 2'd3
    } send_state_e;

    send_state_e       send_state_r;
    logic [SLOT_W-1:0] send_slot_r;

    wire in_fetch_a    = (send_state_r == SEND_FETCH_A);
    wire in_fetch_b    = (send_state_r == SEND_FETCH_B);
    wire in_fetch_meta = (send_state_r == SEND_FETCH_META);
    wire in_fetch      = in_fetch_a || in_fetch_b || in_fetch_meta;

    // -----------------------------------------------------------------------
    // Fetch counters (shared across phases, reset at each phase transition)
    // -----------------------------------------------------------------------

    logic [FETCH_CTR_W-1:0] req_ctr_r;
    logic [FETCH_CTR_W-1:0] rsp_ctr_r;
    logic                   req_inflight_r;

    // -----------------------------------------------------------------------
    // Phase termination thresholds
    // -----------------------------------------------------------------------

    logic [FETCH_CTR_W-1:0] phase_total_rows;
    always_comb begin
        case (send_state_r)
            // Sparse A (2:4 compressed) has half as many K columns as dense A.
            SEND_FETCH_A: begin
            `ifdef TCU_SPARSE_ENABLE
                phase_total_rows = FETCH_CTR_W'(slot_is_sparse[send_slot_r]
                                    ? A_BANK_ROWS_SP : A_BANK_ROWS);
            `else
                phase_total_rows = FETCH_CTR_W'(A_BANK_ROWS);
            `endif
            end
            SEND_FETCH_B:    phase_total_rows = FETCH_CTR_W'(B_BANK_ROWS);
        `ifdef TCU_SPARSE_ENABLE
            SEND_FETCH_META: phase_total_rows = slot_meta_bank_rows[send_slot_r];
        `endif
            default:         phase_total_rows = '0;
        endcase
    end

    wire all_requested = (req_ctr_r >= phase_total_rows);
    wire all_received  = (rsp_ctr_r >= phase_total_rows);
    `UNUSED_VAR (all_received)

    // -----------------------------------------------------------------------
    // Bank-parallel read port handshake (pipelined: 1 request/cycle)
    // -----------------------------------------------------------------------

    wire can_issue = in_fetch && !all_requested
                  && (!req_inflight_r || tcu_lmem_if.rsp_valid);

    assign tcu_lmem_if.req_valid = can_issue;

    // Address mux: bank-row base for current slot's current phase + counter
    logic [BANK_ADDR_WIDTH-1:0] phase_row_base;
    always_comb begin
        case (send_state_r)
            SEND_FETCH_A:    phase_row_base = slot_a_row_base[send_slot_r];
            SEND_FETCH_B:    phase_row_base = slot_b_row_base[send_slot_r];
        `ifdef TCU_SPARSE_ENABLE
            SEND_FETCH_META: phase_row_base = slot_meta_row_base[send_slot_r];
        `endif
            default:         phase_row_base = '0;
        endcase
    end

    assign tcu_lmem_if.req_addr = phase_row_base + BANK_ADDR_WIDTH'(req_ctr_r);

    // -----------------------------------------------------------------------
    // Descriptor parsing (combinational) — used only during slot allocation
    // -----------------------------------------------------------------------

    wire [LMEM_ADDR_WIDTH-1:0] desc_a_word_base = LMEM_ADDR_WIDTH'(req_desc_a[15:0] >> 2);
    wire [LMEM_ADDR_WIDTH-1:0] desc_b_word_base = LMEM_ADDR_WIDTH'(req_desc_b[15:0] >> 2);

    wire [BANK_ADDR_WIDTH-1:0] desc_a_row_base  = desc_a_word_base[BANK_SEL_BITS +: BANK_ADDR_WIDTH];
    wire [BANK_ADDR_WIDTH-1:0] desc_b_row_base  = desc_b_word_base[BANK_SEL_BITS +: BANK_ADDR_WIDTH];

`ifndef TCU_SPARSE_ENABLE
    if (BANK_SEL_BITS > 0) begin : g_unused_word_base_lsbs
        `UNUSED_VAR (desc_a_word_base[BANK_SEL_BITS-1:0])
        `UNUSED_VAR (desc_b_word_base[BANK_SEL_BITS-1:0])
    end
`endif

`ifdef TCU_SPARSE_ENABLE
    // Sparse A (2:4 compressed) occupies A_TOTAL_SP fp32 words starting at
    // desc_a_word_base; metadata immediately follows at +A_TOTAL_SP.
    // Dense A occupies A_TOTAL words (no metadata after it).
    wire [LMEM_ADDR_WIDTH-1:0] desc_meta_word_base = desc_a_word_base
        + LMEM_ADDR_WIDTH'(is_sparse ? A_TOTAL_SP : A_TOTAL);
    wire [BANK_ADDR_WIDTH-1:0] desc_meta_row_base  = desc_meta_word_base[BANK_SEL_BITS +: BANK_ADDR_WIDTH];
    if (BANK_SEL_BITS > 0) begin : g_unused_sparse_word_base_lsbs
        `UNUSED_VAR (desc_b_word_base[BANK_SEL_BITS-1:0])
        `UNUSED_VAR (desc_meta_word_base[BANK_SEL_BITS-1:0])
    end

    logic [3:0]             init_meta_stride;
    logic [FETCH_CTR_W-1:0] init_meta_rows;
    always_comb begin
        case (tcu_fmt_width(req_fmt_s))
            16: begin
                init_meta_stride = 4'(META_STRIDE_16B);
                init_meta_rows   = FETCH_CTR_W'(META_ROWS_16B);
            end
            8: begin
                init_meta_stride = 4'(META_STRIDE_8B);
                init_meta_rows   = FETCH_CTR_W'(META_ROWS_8B);
            end
            default: begin // 8-bit (fp8/bf8/int8/uint8)
                init_meta_stride = 4'(META_STRIDE_8B);
                init_meta_rows   = FETCH_CTR_W'(META_ROWS_8B);
            end
        endcase
    end
`endif

    // -----------------------------------------------------------------------
    // Slot lookup (combinational)
    // -----------------------------------------------------------------------

    wire is_first_uop = (req_step_m == '0) && (req_step_n == '0) && (req_step_k == '0);

    logic                  tbuf_hit;
    logic [SLOT_W-1:0]     hit_slot;

    always_comb begin
        tbuf_hit = 1'b0;
        hit_slot = '0;
        for (int s = 0; s < TCU_TBUF_SIZE; ++s) begin
            if (slot_valid[s]
                    && (slot_wid[s] == req_wid)
                    && slot_fetch_done[s]
                    && (!is_first_uop
                        || (slot_desc_a[s] == req_desc_a && slot_desc_b[s] == req_desc_b))) begin
                if (!tbuf_hit) begin
                    tbuf_hit = 1'b1;
                    hit_slot = SLOT_W'(s);
                end
            end
        end
    end

    assign tbuf_ready = tbuf_hit && !alloc_en;

    // -----------------------------------------------------------------------
    // Slot allocation
    // -----------------------------------------------------------------------

    wire [SLOT_W-1:0] alloc_idx = SLOT_W'(req_wid);

    // slot_in_progress: a slot for this (wid, desc_a, desc_b) exists and its
    // fetch has NOT yet completed.  Used to suppress re-allocation only while
    // a fetch is in progress — if alloc_en fired every cycle (because tbuf_hit
    // requires slot_fetch_done), the alloc write would cancel fetch_done_set_r
    // every cycle and tbuf_ready would never assert.
    //
    // Critically, once fetch_done=1 (slot ready), a new is_first_uop IS allowed
    // to re-allocate the slot so that fresh SMEM content is re-fetched for the
    // next K-tile iteration (the descriptor addresses are fixed, but the SMEM
    // content changes between outer-loop iterations).
    logic slot_in_progress;
    always_comb begin
        slot_in_progress = 1'b0;
        for (int s = 0; s < TCU_TBUF_SIZE; ++s) begin
            if (slot_valid[s] && (slot_wid[s] == req_wid)
                    && (slot_desc_a[s] == req_desc_a) && (slot_desc_b[s] == req_desc_b)
                    && !slot_fetch_done[s])
                slot_in_progress = 1'b1;
        end
    end

    wire alloc_en = req_valid && is_first_uop && !slot_in_progress
                 && !warp_alloc_pending[alloc_idx];

    // -----------------------------------------------------------------------
    // Sender FSM
    // -----------------------------------------------------------------------
    //
    // fetch_done_set/fetch_done_slot are combinational signals that the FSM
    // asserts on the last response of a fetch sequence.  The slot-state
    // always_ff below consumes both alloc_en (clear fetch_done) and
    // fetch_done_set (set fetch_done) so that slot_fetch_done has exactly
    // one driver.
    //
    // Priority: alloc_en (clear) wins over fetch_done_set for the same slot
    // index in the same cycle — a new alloc that coincides with a just-
    // finished fetch immediately starts fresh.

    logic              fetch_done_set_r;    // registered: fetch completed last cycle
    logic [SLOT_W-1:0] fetch_done_slot_r;  // which slot finished (captured at last rsp)

    // Combinational: "last response of a fetch sequence is being received now"
    // Sparse A (2:4 compressed) uses A_BANK_ROWS_SP instead of A_BANK_ROWS.
`ifdef TCU_SPARSE_ENABLE
    wire last_rsp_a = in_fetch_a && tcu_lmem_if.rsp_valid
                   && (rsp_ctr_r == (slot_is_sparse[send_slot_r]
                       ? FETCH_CTR_W'(A_BANK_ROWS_SP - 1)
                       : FETCH_CTR_W'(A_BANK_ROWS    - 1)));
`else
    wire last_rsp_a = in_fetch_a && tcu_lmem_if.rsp_valid
                   && (rsp_ctr_r == FETCH_CTR_W'(A_BANK_ROWS - 1));
`endif
    wire last_rsp_b = in_fetch_b && tcu_lmem_if.rsp_valid
                   && (rsp_ctr_r == FETCH_CTR_W'(B_BANK_ROWS - 1));
`ifdef TCU_SPARSE_ENABLE
    wire last_rsp_meta = in_fetch_meta && tcu_lmem_if.rsp_valid
                      && (rsp_ctr_r == slot_meta_bank_rows[send_slot_r] - FETCH_CTR_W'(1));
    wire fetch_done_now = last_rsp_b && !slot_is_sparse[send_slot_r]
                       || last_rsp_meta;
`else
    wire fetch_done_now = last_rsp_b;
`endif

    always_ff @(posedge clk) begin
        if (reset) begin
            send_state_r      <= SEND_IDLE;
            send_slot_r       <= '0;
            req_ctr_r         <= '0;
            rsp_ctr_r         <= '0;
            req_inflight_r    <= 1'b0;
            fetch_done_set_r  <= 1'b0;
            fetch_done_slot_r <= '0;
        end else begin
            fetch_done_set_r  <= fetch_done_now;
            if (fetch_done_now)
                fetch_done_slot_r <= send_slot_r;

            // In-flight tracking
            if (tcu_lmem_if.rsp_valid)
                req_inflight_r <= 1'b0;
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                req_inflight_r <= 1'b1;
            // Note: if both fire in same cycle, last-assignment wins → stays 1

            case (send_state_r)
            // ---------------------------------------------------------
            SEND_IDLE: begin
                // Priority scan: lowest slot index that needs fetching.
                // alloc_en clears fetch_done in the slot always_ff below;
                // because both run on posedge clk, the newly allocated slot
                // will be visible to the scan on the NEXT cycle, which is
                // correct — we cannot start fetching until the bases are
                // registered.
                for (int s = TCU_TBUF_SIZE-1; s >= 0; s--) begin
                    if (slot_valid[s] && !slot_fetch_done[s]) begin
                        send_slot_r    <= SLOT_W'(s);
                        req_ctr_r      <= '0;
                        rsp_ctr_r      <= '0;
                        req_inflight_r <= 1'b0;
                        send_state_r   <= SEND_FETCH_A;
                    end
                end
            end
            // ---------------------------------------------------------
            SEND_FETCH_A: begin
                if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                    req_ctr_r <= req_ctr_r + FETCH_CTR_W'(1);
                if (last_rsp_a) begin
                    req_ctr_r      <= '0;
                    rsp_ctr_r      <= '0;
                    req_inflight_r <= 1'b0;
                    send_state_r   <= SEND_FETCH_B;
                end else if (tcu_lmem_if.rsp_valid) begin
                    rsp_ctr_r <= rsp_ctr_r + FETCH_CTR_W'(1);
                end
            end
            // ---------------------------------------------------------
            SEND_FETCH_B: begin
                if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                    req_ctr_r <= req_ctr_r + FETCH_CTR_W'(1);
                if (last_rsp_b) begin
                    rsp_ctr_r      <= '0;
                    req_ctr_r      <= '0;
                    req_inflight_r <= 1'b0;
                `ifdef TCU_SPARSE_ENABLE
                    send_state_r <= slot_is_sparse[send_slot_r]
                                  ? SEND_FETCH_META : SEND_IDLE;
                `else
                    send_state_r <= SEND_IDLE;
                `endif
                end else if (tcu_lmem_if.rsp_valid) begin
                    rsp_ctr_r <= rsp_ctr_r + FETCH_CTR_W'(1);
                end
            end
            // ---------------------------------------------------------
        `ifdef TCU_SPARSE_ENABLE
            SEND_FETCH_META: begin
                if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                    req_ctr_r <= req_ctr_r + FETCH_CTR_W'(1);
                if (last_rsp_meta) begin
                    req_ctr_r      <= '0;
                    rsp_ctr_r      <= '0;
                    req_inflight_r <= 1'b0;
                    send_state_r   <= SEND_IDLE;
                end else if (tcu_lmem_if.rsp_valid) begin
                    rsp_ctr_r <= rsp_ctr_r + FETCH_CTR_W'(1);
                end
            end
        `endif
            // ---------------------------------------------------------
            default: send_state_r <= SEND_IDLE;
            endcase
        end
    end

    // -----------------------------------------------------------------------
    // Slot state: single always_ff for slot_valid and slot_fetch_done
    // -----------------------------------------------------------------------

    always_ff @(posedge clk) begin
        if (reset) begin
            for (int s = 0; s < TCU_TBUF_SIZE; ++s) begin
                slot_valid[s]         <= 1'b0;
                slot_fetch_done[s]    <= 1'b0;
                warp_alloc_pending[s] <= 1'b0;
            end
        end else begin
            // warp_alloc_pending: set on alloc_en, cleared when first uop fires.
            // Priority: set on alloc_en (since tbuf_ready=0 when alloc_en=1,
            // the clearing condition can never be true simultaneously).
            if (alloc_en)
                warp_alloc_pending[alloc_idx] <= 1'b1;
            if (tbuf_ready && req_valid && is_first_uop)
                warp_alloc_pending[alloc_idx] <= 1'b0;
            // fetch_done_set_r: set fetch_done for the slot the sender finished
            if (fetch_done_set_r)
                slot_fetch_done[fetch_done_slot_r] <= 1'b1;
            // alloc_en: allocate / re-allocate a slot (clears fetch_done)
            // Priority: alloc wins over fetch_done_set for the same index
            if (alloc_en) begin
                slot_valid      [alloc_idx] <= 1'b1;
                slot_wid        [alloc_idx] <= req_wid;
                slot_desc_a     [alloc_idx] <= req_desc_a;
                slot_desc_b     [alloc_idx] <= req_desc_b;
                slot_fetch_done [alloc_idx] <= 1'b0;
                slot_a_row_base [alloc_idx] <= desc_a_row_base;
                slot_b_row_base [alloc_idx] <= desc_b_row_base;
            `ifdef TCU_SPARSE_ENABLE
                slot_is_sparse      [alloc_idx] <= is_sparse;
                slot_meta_row_base  [alloc_idx] <= desc_meta_row_base;
                slot_meta_stride    [alloc_idx] <= init_meta_stride;
                slot_meta_bank_rows [alloc_idx] <= init_meta_rows;
            `endif
            end
        end
    end

    // -----------------------------------------------------------------------
    // Data capture: write NUM_BANKS words per cycle on tcu_lmem_if.rsp_valid
    // -----------------------------------------------------------------------

    always_ff @(posedge clk) begin
        if (tcu_lmem_if.rsp_valid) begin
            for (int b = 0; b < NUM_BANKS; ++b) begin
                automatic int idx = int'(rsp_ctr_r) * NUM_BANKS + b;
                if (in_fetch_a && idx < A_TOTAL)
                    slot_a_buf[send_slot_r][idx] <= tcu_lmem_if.rsp_data[b * `XLEN +: `XLEN];
                if (in_fetch_b && idx < B_TOTAL)
                    slot_b_buf[send_slot_r][idx] <= tcu_lmem_if.rsp_data[b * `XLEN +: `XLEN];
            `ifdef TCU_SPARSE_ENABLE
                if (in_fetch_meta && idx < META_TOTAL_MAX)
                    slot_meta_buf[send_slot_r][idx] <= tcu_lmem_if.rsp_data[b * `XLEN +: `XLEN];
            `endif
            end
        end
    end

    // -----------------------------------------------------------------------
    // Output mux: A operand (rs1_data)
    // -----------------------------------------------------------------------
    //   slot_a_buf layout: [row * TILE_K + col]
    //   row = step_m * TC_M + i,  col = step_k * TC_K + k
    //   Lane index: a_off + i * TC_K + k

    wire [OFF_W-1:0] a_off_w = (OFF_W'(req_step_m) & OFF_W'(TCU_A_SUB_BLOCKS-1)) << LG_A_BS;

    logic [TCU_BLOCK_CAP-1:0][`XLEN-1:0] rs1_mux;
    always_comb begin
        rs1_mux = '0;
        for (int i = 0; i < TCU_TC_M; ++i) begin
            for (int k = 0; k < TCU_TC_K; ++k) begin
                automatic int tile_row = int'(req_step_m) * TCU_TC_M + i;
                automatic int tile_col = int'(req_step_k) * TCU_TC_K + k;
                automatic int lane     = int'(a_off_w) + i * TCU_TC_K + k;
            `ifdef TCU_SPARSE_ENABLE
                if (slot_is_sparse[hit_slot]) begin
                    // Compressed A: row stride = TILE_K/2 (half as many K cols stored).
                    rs1_mux[lane] = `XLEN'(slot_a_buf[hit_slot][tile_row * (TILE_K/2) + tile_col]);
                end else
            `endif
                begin
                    rs1_mux[lane] = `XLEN'(slot_a_buf[hit_slot][tile_row * TILE_K + tile_col]);
                end
            end
        end
    end

    assign tbuf_rs1_data = rs1_mux;

    // -----------------------------------------------------------------------
    // Output mux: B operand (rs2_data)
    // -----------------------------------------------------------------------
    //   slot_b_buf layout: [row * TILE_N + col]  (row-major: K rows × N cols)
    //   Dense:  row = step_k * TC_K + k,  col = step_n * TC_N + j
    //   Sparse: B is dense, but each sparse step_k covers 2× the K range.
    //           Pairs of consecutive B words are placed for sp_mux gather.

`ifdef TCU_SPARSE_ENABLE
    wire [OFF_W-1:0] b_off_w = is_sparse
        ? (OFF_W'(req_step_n) & OFF_W'(TCU_B_SUB_BLOCKS_SP-1)) << LG_B_BS_SP
        : (OFF_W'(req_step_n) & OFF_W'(TCU_B_SUB_BLOCKS-1))    << LG_B_BS;
`else
    wire [OFF_W-1:0] b_off_w = (OFF_W'(req_step_n) & OFF_W'(TCU_B_SUB_BLOCKS-1)) << LG_B_BS;
`endif

    logic [TCU_BLOCK_CAP-1:0][`XLEN-1:0] rs2_mux;
    always_comb begin
        rs2_mux = '0;

    `ifdef TCU_SPARSE_ENABLE
        if (slot_is_sparse[hit_slot]) begin
            // Sparse: format-aware K-gather for candidate pairs.
            for (int j = 0; j < TCU_TC_N; ++j) begin
                automatic int j_sp = SYM_SPARSE ? (j % (TCU_TC_N / 2)) : j;
                for (int k = 0; k < TCU_TC_K; ++k) begin
                    automatic int b_k0  = int'(req_step_k) * TCU_TC_K * 2 + k * 2;
                    automatic int b_k1  = b_k0 + 1;
                    automatic int b_col = int'(req_step_n) * TCU_TC_N + j;
                    automatic int lane0 = int'(b_off_w) + j_sp * TCU_TC_K * 2 + k * 2;
                    automatic int lane1 = lane0 + 1;
                    if (b_k0 < TILE_K) begin
                        case (tcu_fmt_width(req_fmt_s))
                            32: begin
                                rs2_mux[lane0] = `XLEN'(slot_b_buf[hit_slot][b_k0 * TILE_N + b_col]);
                            end
                            16: begin
                                rs2_mux[lane0][ 0+:16] = slot_b_buf[hit_slot][(b_k0*2+0)*(TILE_N/2) + b_col/2][(b_col%2)*16+:16];
                                rs2_mux[lane0][16+:16] = slot_b_buf[hit_slot][(b_k0*2+1)*(TILE_N/2) + b_col/2][(b_col%2)*16+:16];
                            end
                            default: begin // fp8 / bf8 / int8 / uint8: i_ratio=4, stride=TILE_N/4
                                rs2_mux[lane0][ 0+:8] = slot_b_buf[hit_slot][(b_k0*4+0)*(TILE_N/4) + b_col/4][(b_col%4)*8+:8];
                                rs2_mux[lane0][ 8+:8] = slot_b_buf[hit_slot][(b_k0*4+1)*(TILE_N/4) + b_col/4][(b_col%4)*8+:8];
                                rs2_mux[lane0][16+:8] = slot_b_buf[hit_slot][(b_k0*4+2)*(TILE_N/4) + b_col/4][(b_col%4)*8+:8];
                                rs2_mux[lane0][24+:8] = slot_b_buf[hit_slot][(b_k0*4+3)*(TILE_N/4) + b_col/4][(b_col%4)*8+:8];
                            end
                        endcase
                    end
                    if (b_k1 < TILE_K) begin
                        case (tcu_fmt_width(req_fmt_s))
                            32: begin
                                rs2_mux[lane1] = `XLEN'(slot_b_buf[hit_slot][b_k1 * TILE_N + b_col]);
                            end
                            16: begin
                                rs2_mux[lane1][ 0+:16] = slot_b_buf[hit_slot][(b_k1*2+0)*(TILE_N/2) + b_col/2][(b_col%2)*16+:16];
                                rs2_mux[lane1][16+:16] = slot_b_buf[hit_slot][(b_k1*2+1)*(TILE_N/2) + b_col/2][(b_col%2)*16+:16];
                            end
                            default: begin // fp8 / bf8 / int8 / uint8: i_ratio=4, stride=TILE_N/4
                                rs2_mux[lane1][ 0+:8] = slot_b_buf[hit_slot][(b_k1*4+0)*(TILE_N/4) + b_col/4][(b_col%4)*8+:8];
                                rs2_mux[lane1][ 8+:8] = slot_b_buf[hit_slot][(b_k1*4+1)*(TILE_N/4) + b_col/4][(b_col%4)*8+:8];
                                rs2_mux[lane1][16+:8] = slot_b_buf[hit_slot][(b_k1*4+2)*(TILE_N/4) + b_col/4][(b_col%4)*8+:8];
                                rs2_mux[lane1][24+:8] = slot_b_buf[hit_slot][(b_k1*4+3)*(TILE_N/4) + b_col/4][(b_col%4)*8+:8];
                            end
                        endcase
                    end
                end
            end
        end else
    `endif
        begin
            // Dense: format-aware K-gather.
            for (int j = 0; j < TCU_TC_N; ++j) begin
                for (int k = 0; k < TCU_TC_K; ++k) begin
                    automatic int b_row = int'(req_step_k) * TCU_TC_K + k;
                    automatic int n_col = int'(req_step_n) * TCU_TC_N + j;
                    automatic int lane  = int'(b_off_w) + j * TCU_TC_K + k;
                    case (tcu_fmt_width(req_fmt_s))
                        32: begin // fp32 / i32 / tf32: i_ratio=1, stride=TILE_N
                            rs2_mux[lane] = `XLEN'(slot_b_buf[hit_slot][b_row * TILE_N + n_col]);
                        end
                        16: begin // fp16 / bf16: i_ratio=2, stride=TILE_N/2
                            rs2_mux[lane][ 0+:16] = slot_b_buf[hit_slot][(b_row*2+0)*(TILE_N/2) + n_col/2][(n_col%2)*16+:16];
                            rs2_mux[lane][16+:16] = slot_b_buf[hit_slot][(b_row*2+1)*(TILE_N/2) + n_col/2][(n_col%2)*16+:16];
                        end
                        default: begin // fp8 / bf8 / int8 / uint8: i_ratio=4, stride=TILE_N/4
                            rs2_mux[lane][ 0+:8] = slot_b_buf[hit_slot][(b_row*4+0)*(TILE_N/4) + n_col/4][(n_col%4)*8+:8];
                            rs2_mux[lane][ 8+:8] = slot_b_buf[hit_slot][(b_row*4+1)*(TILE_N/4) + n_col/4][(n_col%4)*8+:8];
                            rs2_mux[lane][16+:8] = slot_b_buf[hit_slot][(b_row*4+2)*(TILE_N/4) + n_col/4][(n_col%4)*8+:8];
                            rs2_mux[lane][24+:8] = slot_b_buf[hit_slot][(b_row*4+3)*(TILE_N/4) + n_col/4][(n_col%4)*8+:8];
                        end
                    endcase
                end
            end
        end
    end

    assign tbuf_rs2_data = rs2_mux;

    // -----------------------------------------------------------------------
    // Metadata extraction for WGMMA_SP
    // -----------------------------------------------------------------------
    //   slot_meta_buf is stored in SMEM order: WG_META_BANKS consecutive slots,
    //   each slot = slot_meta_stride[hit_slot] words.
    //
    //   For a given (step_m, step_k):
    //     wg_bank   = step_m * WG_HALF_K + step_k
    //     word_base = wg_bank * slot_meta_stride[hit_slot]
    //     Extract slot_meta_stride[hit_slot] consecutive words → bit vector
    //
    //   Output is zero-padded to MAX_META_BLOCK_WIDTH (matches VX_tcu_meta).

`ifdef TCU_SPARSE_ENABLE
    logic [META_STRIDE_MAX*32-1:0] extracted_meta;

    always_comb begin
        extracted_meta = '0;
        begin
            automatic int wg_bank = int'(req_step_m) * WG_HALF_K + int'(req_step_k);
            case (slot_meta_stride[hit_slot])
                4'(META_STRIDE_16B): begin
                    for (int w = 0; w < META_STRIDE_16B; ++w) begin
                        automatic int idx = wg_bank * META_STRIDE_16B + w;
                        if (idx < META_TOTAL_MAX)
                            extracted_meta[w*32 +: 32] = slot_meta_buf[hit_slot][idx];
                    end
                end
                default: begin // 8-bit (fp8/bf8/int8/uint8)
                    for (int w = 0; w < META_STRIDE_8B; ++w) begin
                        automatic int idx = wg_bank * META_STRIDE_8B + w;
                        if (idx < META_TOTAL_MAX)
                            extracted_meta[w*32 +: 32] = slot_meta_buf[hit_slot][idx];
                    end
                end
            endcase
        end
    end

    assign tbuf_sp_meta = TCU_MAX_META_BLOCK_WIDTH'(extracted_meta);
`endif

    // -----------------------------------------------------------------------
    // Performance counter
    // -----------------------------------------------------------------------

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] fetch_stall_ctr_r;
    always_ff @(posedge clk) begin
        if (reset)
            fetch_stall_ctr_r <= '0;
        else if (in_fetch)
            fetch_stall_ctr_r <= fetch_stall_ctr_r + PERF_CTR_BITS'(1);
    end
    assign tbuf_fetch_stalls = fetch_stall_ctr_r;
`endif

    // -----------------------------------------------------------------------
    // Debug trace
    // -----------------------------------------------------------------------

`ifdef DBG_TRACE_TCU
    always @(posedge clk) begin
        if (!reset) begin
            if (alloc_en)
                `TRACE(3, ("%t: %s tile-buf: alloc slot=%0d, wid=%0d, desc_a=0x%0h, desc_b=0x%0h, sparse=%0b\n",
                    $time, INSTANCE_ID, alloc_idx, req_wid, req_desc_a, req_desc_b, is_sparse))
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                `TRACE(3, ("%t: %s tile-buf: rd_req slot=%0d, addr=0x%0h, state=%0d, req_ctr=%0d\n",
                    $time, INSTANCE_ID, send_slot_r, tcu_lmem_if.req_addr, send_state_r, req_ctr_r))
            if (tcu_lmem_if.rsp_valid)
                `TRACE(3, ("%t: %s tile-buf: rd_rsp slot=%0d, data[0]=0x%0h, state=%0d, rsp_ctr=%0d\n",
                    $time, INSTANCE_ID, send_slot_r, tcu_lmem_if.rsp_data[0 +: `XLEN], send_state_r, rsp_ctr_r))
            if (fetch_done_now)
                `TRACE(3, ("%t: %s tile-buf: slot=%0d READY (fetch_done)\n",
                    $time, INSTANCE_ID, send_slot_r))
        end
    end
`endif

endmodule
