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

`ifdef TCU_WGMMA_ENABLE

//
// Tile-buffer fetch engine (single-slot + FSM + LMEM capture).
//
// Manages a single tile-buffer slot shared across all warps.
// Runs a three-phase LMEM fetch FSM: FETCH_A -> FETCH_B -> [FETCH_META].
// Exposes slot contents as combinational read ports for VX_tcu_tbuf_gather.
//

module VX_tcu_tbuf_fetch import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID  = "",
    parameter         NUM_BANKS       = 4,
    parameter         BANK_ADDR_WIDTH = 12,
    // Tile buffer sizes (in 32-bit words); passed from VX_tcu_tbuf.
    parameter         A_TOTAL        = 1,
    parameter         B_TOTAL        = 1,
    parameter         C_TOTAL        = 1
`ifdef TCU_SPARSE_ENABLE
   ,parameter         META_TOTAL_MAX = 1
`endif
) (
    input  wire clk,
    input  wire reset,

`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] tbuf_stalls,      // cycles: req_valid && !tbuf_ready
    output wire [PERF_CTR_BITS-1:0] tbuf_cache_hits,  // B tile reuse hits from tile buffer cache
    output wire [PERF_CTR_BITS-1:0] lmem_reads,       // total LMEM transactions accepted
    output wire [PERF_CTR_BITS-1:0] tbuf_tile_fetches, // alloc_en events (tile misses)
    output wire [PERF_CTR_BITS-1:0] tbuf_fetch_cycles, // cycles FSM active (not IDLE)
    output wire [PERF_CTR_BITS-1:0] fetch_b_cycles,   // cycles FSM in FETCH_B phase
    output wire [PERF_CTR_BITS-1:0] lmem_reads_a,     // LMEM reads in FETCH_A phase
    output wire [PERF_CTR_BITS-1:0] lmem_reads_b,     // LMEM reads in FETCH_B phase
    output wire [PERF_CTR_BITS-1:0] lmem_reads_meta,  // LMEM reads in FETCH_META phase
    output wire [PERF_CTR_BITS-1:0] lmem_rsp_stalls,  // cycles: FSM in-flight, no rsp yet
`endif

    // Execute-side observation
    input  wire [NW_WIDTH-1:0]      req_wid,
    input  wire                     req_valid,
    input  wire                     req_fire,   // execute consumed current uop
    input  wire                     req_is_sparse,
    input  wire [3:0]               req_step_m,
    input  wire [3:0]               req_step_n,
    input  wire [3:0]               req_step_k,
    input  wire [3:0]               req_fmt_s,
    input  wire [1:0]               req_cd_nregs,      // NRC selector: 0→8, 1→16, 2→32
    input  wire [`XLEN-1:0]         req_desc_a,
    input  wire [`XLEN-1:0]         req_desc_b,
    input  wire                     req_a_is_smem,
    input  wire [`XLEN-1:0]         req_desc_cd,       // C/D lmem descriptor (valid when cd_from_lmem)
    input  wire                     req_cd_from_lmem,  // 1=C/D accumulator in lmem

    // Prefetch-B sideband: fire-and-forget descriptor for next B tile
    input  wire                     prefetch_b_valid,
    input  wire [`XLEN-1:0]         prefetch_b_desc,

    // C LUTRAM write-back from VX_tcu_core (FEDP output, lmem-accumulator mode)
    input  wire                     c_wb_valid,
    input  wire [C_TOTAL-1:0]       c_wb_wren,
    input  wire [C_TOTAL-1:0][31:0] c_wb_data,

    // Trigger STORE_D after final k-step FEDP outputs land in C LUTRAM
    input  wire                     c_all_done,

    // LMEM bank-parallel read port (1-cycle latency, pipelined)
    VX_mem_bus_if.master            tcu_lmem_if,

    // Hit status
    output wire                     tbuf_hit,
    output wire                     tbuf_ready,

    // Per-row readiness (dense SS only; used by core to fire uops early)
    output wire                          b_ready,
    output wire [TCU_WG_M_STEPS-1:0]    a_row_ready,

    // C tile ready (lmem-accumulator mode: C fetched from lmem)
    output wire                     c_ready,
    // Pulses one cycle after all C_BANK_ROWS writes in STORE_D complete
    output wire                     store_d_done,

    // Slot data (combinational read)
    output wire [A_TOTAL-1:0][31:0] hit_a_buf,
    output wire [B_TOTAL-1:0][31:0] hit_b_buf,
    output wire [C_TOTAL-1:0][31:0] hit_c_buf
`ifdef TCU_SPARSE_ENABLE
   ,output wire                     hit_is_sparse,
    output wire [META_TOTAL_MAX-1:0][31:0] hit_meta_buf,
    output wire [3:0]               hit_meta_stride
`endif
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // -----------------------------------------------------------------------
    // Derived constants
    // -----------------------------------------------------------------------

    localparam BANK_SEL_BITS   = $clog2(NUM_BANKS);
    localparam LMEM_ADDR_W     = BANK_ADDR_WIDTH + BANK_SEL_BITS;
    // Correct word-address shift for any XLEN (2 for 32-bit, 3 for 64-bit).
    localparam WORD_SIZE_LOG2  = $clog2(`XLEN / 8);

    localparam A_BANK_ROWS    = (A_TOTAL + NUM_BANKS - 1) / NUM_BANKS;
    localparam B_BANK_ROWS    = (B_TOTAL + NUM_BANKS - 1) / NUM_BANKS;
    localparam C_BANK_ROWS    = (C_TOTAL + NUM_BANKS - 1) / NUM_BANKS;
    // Half-tile threshold: words 0..A_TOTAL/2-1 cover m_step 0 (A stored row-major).
    localparam A_BANK_ROWS_M0 = (A_TOTAL / 2 + NUM_BANKS - 1) / NUM_BANKS;

`ifdef TCU_SPARSE_ENABLE
    localparam A_TOTAL_SP      = A_TOTAL / 2;
    localparam A_BANK_ROWS_SP  = (A_TOTAL_SP + NUM_BANKS - 1) / NUM_BANKS;

    localparam SP_I_RATIO_32B    = 1;  // tf32/fp32
    localparam SP_I_RATIO_16B    = 2;  // fp16/bf16
    localparam SP_I_RATIO_8B     = 4;  // fp8/bf8/int8/uint8
    localparam META_ROW_BITS_32B = TCU_TC_K * 2 * SP_I_RATIO_32B;
    localparam META_ROW_BITS_16B = TCU_TC_K * 2 * SP_I_RATIO_16B;
    localparam META_ROW_BITS_8B  = TCU_TC_K * 2 * SP_I_RATIO_8B;
    localparam META_STRIDE_32B   = (TCU_TC_M * META_ROW_BITS_32B + 31) / 32;
    localparam META_STRIDE_16B   = (TCU_TC_M * META_ROW_BITS_16B + 31) / 32;
    localparam META_STRIDE_8B    = (TCU_TC_M * META_ROW_BITS_8B  + 31) / 32;
    localparam WG_HALF_K         = TCU_WG_K_STEPS / 2;
    localparam WG_META_BANKS     = TCU_WG_M_STEPS * WG_HALF_K;
    localparam META_TOTAL_32B    = WG_META_BANKS * META_STRIDE_32B;
    localparam META_TOTAL_16B    = WG_META_BANKS * META_STRIDE_16B;
    localparam META_ROWS_MAX     = (META_TOTAL_MAX + NUM_BANKS - 1) / NUM_BANKS;

    localparam MAX_BANK_ROWS_AB = A_BANK_ROWS > B_BANK_ROWS
                             ? (A_BANK_ROWS > META_ROWS_MAX ? A_BANK_ROWS : META_ROWS_MAX)
                             : (B_BANK_ROWS > META_ROWS_MAX ? B_BANK_ROWS : META_ROWS_MAX);
`else
    localparam MAX_BANK_ROWS_AB = A_BANK_ROWS > B_BANK_ROWS ? A_BANK_ROWS : B_BANK_ROWS;
`endif
    localparam MAX_BANK_ROWS = MAX_BANK_ROWS_AB > C_BANK_ROWS ? MAX_BANK_ROWS_AB : C_BANK_ROWS;

    localparam FETCH_CTR_W = `CLOG2(MAX_BANK_ROWS + 1);

    // Single-slot design: fu_lock prevents warp interleaving during WGMMA
    // expansion, so only one warp uses the tile buffer at a time. With
    // per-block (per-slice) tbufs, warps in a slice come from different CTAs
    // and have different desc_b, so there is no cross-warp B reuse; each
    // WGMMA call fetches A (if SS) and B anew.
    localparam SLOT_DEPTH = 1;
    localparam SLOT_ADDRW = 1;
    wire [SLOT_ADDRW-1:0] cur_slot = 1'b0;

`ifndef DBG_TCU_PERF
    `UNUSED_VAR (req_wid)
`endif
`ifndef TCU_SPARSE_ENABLE
    wire is_sparse = 1'b0;
    `UNUSED_VAR (is_sparse)
    `UNUSED_VAR (req_is_sparse)
    `UNUSED_VAR (req_fmt_s)
`else
    wire is_sparse = req_is_sparse;
`endif

    // -----------------------------------------------------------------------
    // Per-slot state (SLOT_DEPTH entries)
    // -----------------------------------------------------------------------

    logic [SLOT_DEPTH-1:0]                            slot_valid;
    logic [SLOT_DEPTH-1:0][`XLEN-1:0]                 slot_desc_a;
    logic [SLOT_DEPTH-1:0][`XLEN-1:0]                 slot_desc_b;
    logic [SLOT_DEPTH-1:0]                            slot_fetch_done;
    logic [SLOT_DEPTH-1:0]                            warp_alloc_pending;
    logic [SLOT_DEPTH-1:0]                            slot_a_from_smem;
    logic [SLOT_DEPTH-1:0]                            slot_cd_from_lmem;
    logic [SLOT_DEPTH-1:0][BANK_ADDR_WIDTH-1:0]       slot_a_row_base;
    logic [SLOT_DEPTH-1:0][BANK_ADDR_WIDTH-1:0]       slot_b_row_base;
    logic [SLOT_DEPTH-1:0][BANK_ADDR_WIDTH-1:0]       slot_cd_row_base;
    logic [SLOT_ADDRW-1:0]                            fetch_slot;

`ifdef TCU_SPARSE_ENABLE
    logic [SLOT_DEPTH-1:0]                            slot_is_sparse;
    logic [SLOT_DEPTH-1:0][BANK_ADDR_WIDTH-1:0]       slot_meta_row_base;
    logic [SLOT_DEPTH-1:0][3:0]                       slot_meta_stride;
    logic [SLOT_DEPTH-1:0][FETCH_CTR_W-1:0]           slot_meta_bank_rows;
`endif

    // -----------------------------------------------------------------------
    // Slot lookup
    // -----------------------------------------------------------------------

    wire is_first_uop = (req_step_m == '0) && (req_step_n == '0) && (req_step_k == '0);

    // Descriptor match: validate against current slot contents.
    // RS mode (a_from_smem=0): A comes from registers, only check B descriptor.
    wire desc_match = (slot_desc_b[cur_slot] == req_desc_b)
                   && (req_a_is_smem ? (slot_desc_a[cur_slot] == req_desc_a) : 1'b1);

    // Hit: slot is valid, data ready, and (on first uop) descriptors match.
    // Non-first uops of a WGMMA expansion share the descriptor from first uop;
    // rs1/rs2 may be gated off via used_rs, so req_desc_* is only valid on first uop.
    assign tbuf_hit = slot_valid[cur_slot] && slot_fetch_done[cur_slot]
                   && (!is_first_uop || desc_match);

    // alloc_en and tbuf_ready are defined after the FSM state declaration (below)
    wire alloc_en;
    assign tbuf_ready = tbuf_hit && !alloc_en;

    // -----------------------------------------------------------------------
    // Descriptor parsing -> row bases (combinational, used only at alloc)
    // -----------------------------------------------------------------------

    wire [LMEM_ADDR_W-1:0] desc_a_word_base  = LMEM_ADDR_W'(req_desc_a[15:0]  >> WORD_SIZE_LOG2);
    wire [LMEM_ADDR_W-1:0] desc_b_word_base  = LMEM_ADDR_W'(req_desc_b[15:0]  >> WORD_SIZE_LOG2);
    wire [LMEM_ADDR_W-1:0] desc_cd_word_base = LMEM_ADDR_W'(req_desc_cd[15:0] >> WORD_SIZE_LOG2);

    wire [BANK_ADDR_WIDTH-1:0] desc_a_row_base  = desc_a_word_base [BANK_SEL_BITS +: BANK_ADDR_WIDTH];
    wire [BANK_ADDR_WIDTH-1:0] desc_b_row_base  = desc_b_word_base [BANK_SEL_BITS +: BANK_ADDR_WIDTH];
    wire [BANK_ADDR_WIDTH-1:0] desc_cd_row_base = desc_cd_word_base[BANK_SEL_BITS +: BANK_ADDR_WIDTH];

    // PREFETCH_B descriptor parsing (combinational; used only when prefetch_b_valid fires)
    wire [LMEM_ADDR_W-1:0]     pb_word_base_c = LMEM_ADDR_W'(prefetch_b_desc[15:0] >> WORD_SIZE_LOG2);
    wire [BANK_ADDR_WIDTH-1:0] pb_row_base_c  = pb_word_base_c[BANK_SEL_BITS +: BANK_ADDR_WIDTH];

    if (BANK_SEL_BITS > 0) begin : g_word_base_lsbs_unused
        `UNUSED_VAR (desc_a_word_base[BANK_SEL_BITS-1:0])
        `UNUSED_VAR (desc_b_word_base[BANK_SEL_BITS-1:0])
        `UNUSED_VAR (desc_cd_word_base[BANK_SEL_BITS-1:0])
        `UNUSED_VAR (pb_word_base_c[BANK_SEL_BITS-1:0])
    end
    `UNUSED_VAR (req_desc_cd[`XLEN-1:16])

`ifdef TCU_SPARSE_ENABLE
    wire [LMEM_ADDR_W-1:0] desc_meta_word_base = desc_a_word_base
        + LMEM_ADDR_W'(is_sparse ? A_TOTAL_SP : A_TOTAL);
    wire [BANK_ADDR_WIDTH-1:0] desc_meta_row_base = desc_meta_word_base[BANK_SEL_BITS +: BANK_ADDR_WIDTH];
    if (BANK_SEL_BITS > 0) begin : g_meta_word_base_lsbs_unused
        `UNUSED_VAR (desc_meta_word_base[BANK_SEL_BITS-1:0])
    end

    logic [3:0]             init_meta_stride;
    logic [FETCH_CTR_W-1:0] init_meta_rows;
    always_comb begin
        case (tcu_fmt_width(req_fmt_s))
            32: begin  // tf32/fp32: i_ratio=1
                init_meta_stride = 4'(META_STRIDE_32B);
                init_meta_rows   = FETCH_CTR_W'((META_TOTAL_32B + NUM_BANKS - 1) / NUM_BANKS);
            end
            16: begin  // fp16/bf16: i_ratio=2
                init_meta_stride = 4'(META_STRIDE_16B);
                init_meta_rows   = FETCH_CTR_W'((META_TOTAL_16B + NUM_BANKS - 1) / NUM_BANKS);
            end
            default: begin  // fp8/bf8/int8/uint8: i_ratio=4
                init_meta_stride = 4'(META_STRIDE_8B);
                init_meta_rows   = FETCH_CTR_W'(META_ROWS_MAX);
            end
        endcase
    end
`endif

    // -----------------------------------------------------------------------
    // Sender FSM states
    // -----------------------------------------------------------------------

    typedef enum logic [2:0] {
        SEND_IDLE       = 3'd0,
        SEND_FETCH_C    = 3'd1,
        SEND_FETCH_B    = 3'd2,
        SEND_FETCH_A    = 3'd3,
        SEND_FETCH_META = 3'd4,
        SEND_STORE_D    = 3'd5
    } send_state_e;

    (* fsm_encoding = "one_hot" *) send_state_e send_state_r;

    // Prefetch-B queue (1-entry) and related state
    reg                        pb_valid_r;        // queued PREFETCH_B descriptor
    reg [`XLEN-1:0]            pb_desc_r;         // queued b_desc
    reg [BANK_ADDR_WIDTH-1:0]  pb_row_base_r;     // pre-computed LMEM row base for pb_desc_r
    reg [`XLEN-1:0]            b_buf_desc_r;      // which desc_b is currently in B LUTRAM
    reg                        is_b_prefetch_r;   // current SEND_FETCH_B is a standalone prefetch
    reg                        b_prefetch_hit_r;  // alloc hit a prefetched B (skip SEND_FETCH_B)

    wire in_fetch_c    = (send_state_r == SEND_FETCH_C);
    wire in_fetch_a    = (send_state_r == SEND_FETCH_A);
    wire in_fetch_b    = (send_state_r == SEND_FETCH_B);
    wire in_store_d    = (send_state_r == SEND_STORE_D);
`ifdef TCU_SPARSE_ENABLE
    wire in_fetch_meta = (send_state_r == SEND_FETCH_META);
`endif
    wire in_fetch      = in_fetch_c || in_fetch_b || in_fetch_a
`ifdef TCU_SPARSE_ENABLE
                      || in_fetch_meta
`endif
                      ;

    // Allocate (or re-allocate) a slot on the first uop of each new tile.
    // req_desc_a/req_desc_b may be stale on non-first uops (used_rs gates RF
    // reads), so alloc/desc_match checks must be gated on is_first_uop.
    wire slot_in_progress = slot_valid[cur_slot] && !slot_fetch_done[cur_slot];
    assign alloc_en = req_valid && is_first_uop && !slot_in_progress
                   && !warp_alloc_pending[cur_slot];

    // Prefetch hit: B tile is already in the LUTRAM with the right descriptor.
    // Only valid on first uop (alloc_en cycle); suppresses SEND_FETCH_B phase.
    wire b_prefetch_hit_alloc = alloc_en && b_ready_r && (req_desc_b == b_buf_desc_r);

    // Immediate fetch-done for RS + prefetch-hit + no-lmem-accum: no fetch needed at all.
    wire immediate_fetch_done = alloc_en && b_prefetch_hit_alloc
                             && !req_a_is_smem && !req_cd_from_lmem;

    // -----------------------------------------------------------------------
    // Fetch counters (req: issued requests; rsp: received responses)
    // -----------------------------------------------------------------------

    logic [FETCH_CTR_W-1:0] req_ctr_r;
    logic [FETCH_CTR_W-1:0] rsp_ctr_r;
    logic                   req_inflight_r;

    // Per-phase termination threshold and row base.
    logic [FETCH_CTR_W-1:0]     phase_total_rows;
    logic [BANK_ADDR_WIDTH-1:0] phase_row_base;
    always_comb begin
        case (send_state_r)
            SEND_FETCH_C: begin
                phase_total_rows = FETCH_CTR_W'(C_BANK_ROWS);
                phase_row_base   = slot_cd_row_base[fetch_slot];
            end
            SEND_FETCH_A: begin
`ifdef TCU_SPARSE_ENABLE
                phase_total_rows = FETCH_CTR_W'(slot_is_sparse[fetch_slot]
                                   ? A_BANK_ROWS_SP : A_BANK_ROWS);
`else
                phase_total_rows = FETCH_CTR_W'(A_BANK_ROWS);
`endif
                phase_row_base   = slot_a_row_base[fetch_slot];
            end
            SEND_FETCH_B: begin
                phase_total_rows = FETCH_CTR_W'(B_BANK_ROWS);
                phase_row_base   = is_b_prefetch_r ? pb_row_base_r : slot_b_row_base[fetch_slot];
            end
`ifdef TCU_SPARSE_ENABLE
            SEND_FETCH_META: begin
                phase_total_rows = slot_meta_bank_rows[fetch_slot];
                phase_row_base   = slot_meta_row_base[fetch_slot];
            end
`endif
            SEND_STORE_D: begin
                phase_total_rows = FETCH_CTR_W'(C_BANK_ROWS);
                phase_row_base   = slot_cd_row_base[fetch_slot];
            end
            default: begin
                phase_total_rows = '0;
                phase_row_base   = '0;
            end
        endcase
    end

    wire all_requested = (req_ctr_r >= phase_total_rows);

    // -----------------------------------------------------------------------
    // LMEM request handshake
    // -----------------------------------------------------------------------

    // Reads use req_inflight_r to ensure one in-flight at a time (rsp_valid clears it).
    // Writes (STORE_D) have no response, so bypass req_inflight_r entirely.
    wire can_issue = (in_fetch   && !all_requested && (!req_inflight_r || tcu_lmem_if.rsp_valid))
                  || (in_store_d && !all_requested);

    // Write data from C LUTRAM for STORE_D phase
    logic [NUM_BANKS*`XLEN-1:0] store_d_wdata;
    logic [NUM_BANKS*(`XLEN/8)-1:0] store_d_byteen;
    always_comb begin
        store_d_wdata  = '0;
        store_d_byteen = '0;
        for (int b = 0; b < NUM_BANKS; ++b) begin
            automatic int idx = int'(req_ctr_r) * NUM_BANKS + b;
            if (idx < C_TOTAL) begin
                store_d_wdata [b * `XLEN +: `XLEN]         = `XLEN'(hit_c_buf[idx]);
                store_d_byteen[b * (`XLEN/8) +: (`XLEN/8)] = '1;
            end
        end
    end

    assign tcu_lmem_if.req_valid       = can_issue;
    assign tcu_lmem_if.req_data.rw     = in_store_d;
    assign tcu_lmem_if.req_data.addr   = LMEM_DMA_ADDR_WIDTH'(phase_row_base + BANK_ADDR_WIDTH'(req_ctr_r));
    assign tcu_lmem_if.req_data.data   = in_store_d ? store_d_wdata : '0;
    assign tcu_lmem_if.req_data.byteen = in_store_d ? store_d_byteen : '0;
    assign tcu_lmem_if.req_data.flags  = '0;
    assign tcu_lmem_if.req_data.tag    = '0;
    assign tcu_lmem_if.rsp_ready       = 1'b1;
    `UNUSED_VAR (tcu_lmem_if.rsp_data.tag)

    // -----------------------------------------------------------------------
    // Phase-done detection
    // -----------------------------------------------------------------------

    wire last_rsp_c = in_fetch_c && tcu_lmem_if.rsp_valid
                   && (rsp_ctr_r == FETCH_CTR_W'(C_BANK_ROWS - 1));
`ifdef TCU_SPARSE_ENABLE
    wire last_rsp_a = in_fetch_a && tcu_lmem_if.rsp_valid
                   && (rsp_ctr_r == (slot_is_sparse[fetch_slot]
                       ? FETCH_CTR_W'(A_BANK_ROWS_SP - 1)
                       : FETCH_CTR_W'(A_BANK_ROWS    - 1)));
`else
    wire last_rsp_a = in_fetch_a && tcu_lmem_if.rsp_valid
                   && (rsp_ctr_r == FETCH_CTR_W'(A_BANK_ROWS - 1));
`endif
    wire last_rsp_b = in_fetch_b && tcu_lmem_if.rsp_valid
                   && (rsp_ctr_r == FETCH_CTR_W'(B_BANK_ROWS - 1));
    // STORE_D completes on last write request accepted (writes have no rsp).
    wire last_store_d = in_store_d && tcu_lmem_if.req_valid && tcu_lmem_if.req_ready
                     && (req_ctr_r == FETCH_CTR_W'(C_BANK_ROWS - 1));

    // Fires when m_step 0's A words (first half of A tile) have all arrived.
    wire last_rsp_a_m0 = in_fetch_a && tcu_lmem_if.rsp_valid
                      && (rsp_ctr_r == FETCH_CTR_W'(A_BANK_ROWS_M0 - 1));

`ifdef TCU_SPARSE_ENABLE
    wire last_rsp_meta = in_fetch_meta && tcu_lmem_if.rsp_valid
                      && (rsp_ctr_r == slot_meta_bank_rows[fetch_slot] - FETCH_CTR_W'(1));
    // SS sparse tiles finish after META; SS dense after A; RS after B (or after C for prefetch-hit).
    wire is_ss_sparse = slot_is_sparse[fetch_slot] && slot_a_from_smem[fetch_slot];
    wire fetch_done_now = is_ss_sparse ? last_rsp_meta
                        : (slot_a_from_smem[fetch_slot] ? last_rsp_a
                         : (b_prefetch_hit_r ? last_rsp_c : last_rsp_b));
`else
    // When b_prefetch_hit_r: RS lmem-accum skips FETCH_B and completes on last_rsp_c.
    // RS no-lmem-accum completes via immediate_fetch_done below (no LMEM needed).
    wire fetch_done_now = slot_a_from_smem[fetch_slot] ? last_rsp_a
                        : (b_prefetch_hit_r ? last_rsp_c : last_rsp_b);
`endif

    // -----------------------------------------------------------------------
    // Sender FSM
    // -----------------------------------------------------------------------

    always_ff @(posedge clk) begin
        if (reset) begin
            send_state_r    <= SEND_IDLE;
            req_ctr_r       <= '0;
            rsp_ctr_r       <= '0;
            req_inflight_r  <= 1'b0;
            is_b_prefetch_r <= 1'b0;
        end else begin
            if (tcu_lmem_if.rsp_valid)
                req_inflight_r <= 1'b0;
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                req_inflight_r <= 1'b1;

            case (send_state_r)
            // -----------------------------------------------------------------
            SEND_IDLE: begin
                // STORE_D triggered by c_all_done (higher priority than new fetch)
                if (c_all_done && slot_cd_from_lmem[cur_slot]) begin
                    fetch_slot      <= cur_slot;
                    req_ctr_r       <= '0;
                    rsp_ctr_r       <= '0;
                    req_inflight_r  <= 1'b0;
                    send_state_r    <= SEND_STORE_D;
                end else if (pb_valid_r && !b_ready_r && !slot_in_progress) begin
                    // PREFETCH_B: start standalone B fetch from queued descriptor
                    fetch_slot      <= cur_slot;
                    req_ctr_r       <= '0;
                    rsp_ctr_r       <= '0;
                    req_inflight_r  <= 1'b0;
                    is_b_prefetch_r <= 1'b1;
                    send_state_r    <= SEND_FETCH_B;
                end else begin
                    for (int s = SLOT_DEPTH-1; s >= 0; s--) begin
                        if (slot_valid[s] && !slot_fetch_done[s]) begin
                            fetch_slot      <= SLOT_ADDRW'(s);
                            req_ctr_r       <= '0;
                            rsp_ctr_r       <= '0;
                            req_inflight_r  <= 1'b0;
                            is_b_prefetch_r <= 1'b0;
                            // If B was prefetched for this alloc, skip SEND_FETCH_B.
                            // Use b_prefetch_hit_r: registered from alloc_en cycle (T),
                            // valid here at cycle T+1 when slot_valid first becomes 1.
                            if (b_prefetch_hit_r) begin
                                send_state_r <= slot_cd_from_lmem[s] ? SEND_FETCH_C
                                             : (slot_a_from_smem[s]  ? SEND_FETCH_A
                                                                      : SEND_IDLE);
                            end else begin
                                // Normal: Fetch C first (lmem-accum), then B, then A (SS only)
                                send_state_r <= slot_cd_from_lmem[s] ? SEND_FETCH_C : SEND_FETCH_B;
                            end
                        end
                    end
                end
            end
            // -----------------------------------------------------------------
            SEND_FETCH_C: begin
                if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                    req_ctr_r <= req_ctr_r + FETCH_CTR_W'(1);
                if (last_rsp_c) begin
                    req_ctr_r      <= '0;
                    rsp_ctr_r      <= '0;
                    req_inflight_r <= 1'b0;
                    // If B was prefetched, skip FETCH_B
                    if (b_prefetch_hit_r)
                        send_state_r <= slot_a_from_smem[fetch_slot] ? SEND_FETCH_A : SEND_IDLE;
                    else
                        send_state_r <= SEND_FETCH_B;
                end else if (tcu_lmem_if.rsp_valid) begin
                    rsp_ctr_r <= rsp_ctr_r + FETCH_CTR_W'(1);
                end
            end
            // -----------------------------------------------------------------
            SEND_FETCH_B: begin
                if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                    req_ctr_r <= req_ctr_r + FETCH_CTR_W'(1);
                if (last_rsp_b) begin
                    req_ctr_r      <= '0;
                    rsp_ctr_r      <= '0;
                    req_inflight_r <= 1'b0;
                    is_b_prefetch_r <= 1'b0;
                    // Standalone prefetch: done (no alloc yet, so a_from_smem unknown)
                    // Normal alloc-triggered: SS → FETCH_A; RS → IDLE
                    send_state_r <= (is_b_prefetch_r || !slot_a_from_smem[fetch_slot])
                                  ? SEND_IDLE : SEND_FETCH_A;
                end else if (tcu_lmem_if.rsp_valid) begin
                    rsp_ctr_r <= rsp_ctr_r + FETCH_CTR_W'(1);
                end
            end
            // -----------------------------------------------------------------
            SEND_FETCH_A: begin
                if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                    req_ctr_r <= req_ctr_r + FETCH_CTR_W'(1);
                if (last_rsp_a) begin
                    req_ctr_r      <= '0;
                    rsp_ctr_r      <= '0;
                    req_inflight_r <= 1'b0;
                `ifdef TCU_SPARSE_ENABLE
                    // Only SS sparse needs LMEM metadata fetch; RS sparse uses VX_tcu_meta.
                    send_state_r <= (slot_is_sparse[fetch_slot] && slot_a_from_smem[fetch_slot])
                                  ? SEND_FETCH_META : SEND_IDLE;
                `else
                    send_state_r <= SEND_IDLE;
                `endif
                end else if (tcu_lmem_if.rsp_valid) begin
                    rsp_ctr_r <= rsp_ctr_r + FETCH_CTR_W'(1);
                end
            end
            // -----------------------------------------------------------------
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
            // -----------------------------------------------------------------
            SEND_STORE_D: begin
                if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                    req_ctr_r <= req_ctr_r + FETCH_CTR_W'(1);
                if (last_store_d) begin
                    req_ctr_r      <= '0;
                    rsp_ctr_r      <= '0;
                    req_inflight_r <= 1'b0;
                    send_state_r   <= SEND_IDLE;
                end
            end
            // -----------------------------------------------------------------
            default: send_state_r <= SEND_IDLE;
            endcase
        end
    end

    // -----------------------------------------------------------------------
    // Slot state updates
    // -----------------------------------------------------------------------

    always_ff @(posedge clk) begin
        if (reset) begin
            slot_valid         <= '0;
            slot_fetch_done    <= '0;
            warp_alloc_pending <= '0;
            fetch_slot         <= '0;
        end else begin
            if (fetch_done_now)
                slot_fetch_done[fetch_slot] <= 1'b1;
            // RS + prefetch-hit + no-lmem-accum: done on alloc, no LMEM phase needed
            if (immediate_fetch_done)
                slot_fetch_done[cur_slot] <= 1'b1;

            if (alloc_en) begin
                slot_valid[cur_slot]          <= 1'b1;
                slot_fetch_done[cur_slot]     <= 1'b0;
                slot_a_from_smem[cur_slot]    <= req_a_is_smem;
                slot_cd_from_lmem[cur_slot]   <= req_cd_from_lmem;
                slot_desc_a[cur_slot]         <= req_desc_a;
                slot_desc_b[cur_slot]         <= req_desc_b;
                slot_a_row_base[cur_slot]     <= desc_a_row_base;
                slot_b_row_base[cur_slot]     <= desc_b_row_base;
                slot_cd_row_base[cur_slot]    <= desc_cd_row_base;
            `ifdef TCU_SPARSE_ENABLE
                slot_is_sparse[cur_slot]      <= is_sparse;
                slot_meta_row_base[cur_slot]  <= desc_meta_row_base;
                slot_meta_stride[cur_slot]    <= init_meta_stride;
                slot_meta_bank_rows[cur_slot] <= init_meta_rows;
            `endif
            end

            if (alloc_en)
                warp_alloc_pending[cur_slot] <= 1'b1;
            if (req_fire && is_first_uop)
                warp_alloc_pending[cur_slot] <= 1'b0;
        end
    end

    // -----------------------------------------------------------------------
    // Per-row readiness tracking
    // -----------------------------------------------------------------------
    // b_ready_r: set when B tile is fully fetched.
    // a_rows_done_r[m]: set when A words for m_step m are in the LUTRAM.
    // Both cleared on alloc_en (new tile started) OR when the last uop fires,
    // so the next tile's alloc starts from a clean state and prefetch can
    // overlap the inter-tile issue-pipeline gap.

    // Effective N steps depends on cd_from_lmem (forces full NRC=32 tile).
    wire [1:0] eff_cd_nregs_ready = req_cd_from_lmem ? 2'd2 : req_cd_nregs;
    reg last_n_step;
    always_comb begin
        case (eff_cd_nregs_ready)
            2'd0:    last_n_step = (req_step_n == 4'd3);
            2'd1:    last_n_step = (req_step_n == 4'd7);
            default: last_n_step = (req_step_n == 4'd15);
        endcase
    end
    wire last_uop_fire = req_fire
                      && (req_step_m == 4'(TCU_WG_M_STEPS - 1))
                      && last_n_step
                      && (req_step_k == 4'(TCU_WG_K_STEPS - 1));

    reg                       b_ready_r;
    reg [TCU_WG_M_STEPS-1:0] a_rows_done_r;

    always_ff @(posedge clk) begin
        if (reset) begin
            b_ready_r     <= 1'b0;
            a_rows_done_r <= '0;
        // On alloc: clear only if B was not prefetched for this tile (otherwise keep).
        // On last_uop_fire: always clear (tile consumed, ready for next prefetch).
        end else if (last_uop_fire || (alloc_en && !b_prefetch_hit_alloc)) begin
            b_ready_r     <= 1'b0;
            a_rows_done_r <= '0;
        end else begin
            if (last_rsp_b)    b_ready_r              <= 1'b1;
            if (last_rsp_a_m0) a_rows_done_r[0]       <= 1'b1;
            if (last_rsp_a)    a_rows_done_r           <= {TCU_WG_M_STEPS{1'b1}};
        end
    end

    assign b_ready     = b_ready_r     && !alloc_en;
    assign a_row_ready = a_rows_done_r & {TCU_WG_M_STEPS{!alloc_en}};

    // -----------------------------------------------------------------------
    // Prefetch-B queue and B-buffer descriptor tracking
    // -----------------------------------------------------------------------

    always_ff @(posedge clk) begin
        if (reset) begin
            pb_valid_r    <= 1'b0;
            pb_desc_r     <= '0;
            pb_row_base_r <= '0;
        end else begin
            if (prefetch_b_valid) begin
                // Always accept latest PREFETCH_B; overwrite stale entry
                pb_valid_r    <= 1'b1;
                pb_desc_r     <= prefetch_b_desc;
                pb_row_base_r <= pb_row_base_c;
            end else if (alloc_en || (last_rsp_b && is_b_prefetch_r)) begin
                // Consumed: by alloc (hit or miss) or by completing the prefetch fetch
                pb_valid_r <= 1'b0;
            end
        end
    end

    always_ff @(posedge clk) begin
        if (reset)
            b_buf_desc_r <= '0;
        else if (last_rsp_b)
            b_buf_desc_r <= is_b_prefetch_r ? pb_desc_r : slot_desc_b[fetch_slot];
    end

    always_ff @(posedge clk) begin
        if (reset)
            b_prefetch_hit_r <= 1'b0;
        else if (alloc_en)
            b_prefetch_hit_r <= b_prefetch_hit_alloc;
    end

    // -----------------------------------------------------------------------
    // C tile readiness tracking (lmem-accumulator mode)
    // -----------------------------------------------------------------------
    reg c_ready_r;
    always_ff @(posedge clk) begin
        if (reset)
            c_ready_r <= 1'b0;
        else if (alloc_en)
            c_ready_r <= 1'b0;
        else if (last_rsp_c)
            c_ready_r <= 1'b1;
    end
    assign c_ready = c_ready_r && !alloc_en;

    // -----------------------------------------------------------------------
    // A data capture — single-entry LUTRAM (SIZE=1, async read)
    // -----------------------------------------------------------------------

    logic [A_TOTAL*32-1:0] a_wdata;
    logic [A_TOTAL-1:0]    a_wren;

    always_comb begin
        a_wdata = {A_TOTAL{32'b0}};
        a_wren  = {A_TOTAL{1'b0}};
        if (tcu_lmem_if.rsp_valid && in_fetch_a) begin
            for (int b = 0; b < NUM_BANKS; ++b) begin
                if (int'(rsp_ctr_r) * NUM_BANKS + b < A_TOTAL) begin
                    a_wren[int'(rsp_ctr_r) * NUM_BANKS + b]               = 1'b1;
                    a_wdata[(int'(rsp_ctr_r) * NUM_BANKS + b) * 32 +: 32] =
                        tcu_lmem_if.rsp_data.data[b * `XLEN +: `XLEN];
                end
            end
        end
    end

    VX_dp_ram #(
        .DATAW   (A_TOTAL * 32),
        .SIZE    (SLOT_DEPTH),
        .WRENW   (A_TOTAL),
        .LUTRAM  (1),
        .OUT_REG (0),
        .RDW_MODE("W")
    ) slot_a_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (tcu_lmem_if.rsp_valid && in_fetch_a),
        .wren  (a_wren),
        .waddr (fetch_slot),
        .wdata (a_wdata),
        .raddr (cur_slot),
        .rdata (hit_a_buf)
    );

    // -----------------------------------------------------------------------
    // B data capture — single-entry LUTRAM (SIZE=1, async read)
    // -----------------------------------------------------------------------

    logic [B_TOTAL*32-1:0] b_wdata;
    logic [B_TOTAL-1:0]    b_wren;

    always_comb begin
        b_wdata = {B_TOTAL{32'b0}};
        b_wren  = {B_TOTAL{1'b0}};
        if (tcu_lmem_if.rsp_valid && in_fetch_b) begin
            for (int b = 0; b < NUM_BANKS; ++b) begin
                if (int'(rsp_ctr_r) * NUM_BANKS + b < B_TOTAL) begin
                    b_wren[int'(rsp_ctr_r) * NUM_BANKS + b]               = 1'b1;
                    b_wdata[(int'(rsp_ctr_r) * NUM_BANKS + b) * 32 +: 32] =
                        tcu_lmem_if.rsp_data.data[b * `XLEN +: `XLEN];
                end
            end
        end
    end

    VX_dp_ram #(
        .DATAW   (B_TOTAL * 32),
        .SIZE    (1),
        .WRENW   (B_TOTAL),
        .LUTRAM  (1),
        .OUT_REG (0),
        .RDW_MODE("W")
    ) shared_b_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (tcu_lmem_if.rsp_valid && in_fetch_b),
        .wren  (b_wren),
        .waddr (1'b0),
        .wdata (b_wdata),
        .raddr (1'b0),
        .rdata (hit_b_buf)
    );

    // -----------------------------------------------------------------------
    // C data capture — single-entry LUTRAM (lmem-accumulator mode)
    // Written by: (a) LMEM reads during FETCH_C; (b) VX_tcu_core FEDP output via c_wb_*
    // Read by: VX_tcu_tbuf_gather (combinational) and STORE_D FSM via store_d_wdata
    // -----------------------------------------------------------------------

    logic [C_TOTAL*32-1:0] c_wdata;
    logic [C_TOTAL-1:0]    c_wren;
    logic                  c_write_en;

    always_comb begin
        c_wdata    = '0;
        c_wren     = '0;
        c_write_en = 1'b0;
        if (tcu_lmem_if.rsp_valid && in_fetch_c) begin
            // Capture from lmem response (NUM_BANKS words per cycle)
            c_write_en = 1'b1;
            for (int b = 0; b < NUM_BANKS; ++b) begin
                if (int'(rsp_ctr_r) * NUM_BANKS + b < C_TOTAL) begin
                    c_wren[int'(rsp_ctr_r) * NUM_BANKS + b]               = 1'b1;
                    c_wdata[(int'(rsp_ctr_r) * NUM_BANKS + b) * 32 +: 32] =
                        tcu_lmem_if.rsp_data.data[b * `XLEN +: `XLEN];
                end
            end
        end else if (c_wb_valid) begin
            // Capture from VX_tcu_core FEDP write-back
            c_write_en = 1'b1;
            for (int i = 0; i < C_TOTAL; ++i) begin
                c_wren[i]           = c_wb_wren[i];
                c_wdata[i * 32 +: 32] = c_wb_data[i];
            end
        end
    end

    VX_dp_ram #(
        .DATAW   (C_TOTAL * 32),
        .SIZE    (1),
        .WRENW   (C_TOTAL),
        .LUTRAM  (1),
        .OUT_REG (0),
        .RDW_MODE("W")
    ) shared_c_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (c_write_en),
        .wren  (c_wren),
        .waddr (1'b0),
        .wdata (c_wdata),
        .raddr (1'b0),
        .rdata (hit_c_buf)
    );

    // store_d_done: combinational pulse on last STORE_D acknowledgment
    assign store_d_done = last_store_d;

    // -----------------------------------------------------------------------
    // Metadata capture — single-entry LUTRAM (sparse only)
    // -----------------------------------------------------------------------

`ifdef TCU_SPARSE_ENABLE
    logic [META_TOTAL_MAX*32-1:0] meta_wdata;
    logic [META_TOTAL_MAX-1:0]    meta_wren;

    always_comb begin
        meta_wdata = '0;
        meta_wren  = '0;
        if (tcu_lmem_if.rsp_valid && in_fetch_meta) begin
            for (int b = 0; b < NUM_BANKS; ++b) begin
                if (int'(rsp_ctr_r) * NUM_BANKS + b < META_TOTAL_MAX) begin
                    meta_wren[int'(rsp_ctr_r) * NUM_BANKS + b]               = 1'b1;
                    meta_wdata[(int'(rsp_ctr_r) * NUM_BANKS + b) * 32 +: 32] =
                        tcu_lmem_if.rsp_data.data[b * `XLEN +: `XLEN];
                end
            end
        end
    end

    VX_dp_ram #(
        .DATAW   (META_TOTAL_MAX * 32),
        .SIZE    (SLOT_DEPTH),
        .WRENW   (META_TOTAL_MAX),
        .LUTRAM  (1),
        .OUT_REG (0),
        .RDW_MODE("W")
    ) slot_meta_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (tcu_lmem_if.rsp_valid && in_fetch_meta),
        .wren  (meta_wren),
        .waddr (fetch_slot),
        .wdata (meta_wdata),
        .raddr (cur_slot),
        .rdata (hit_meta_buf)
    );

    assign hit_is_sparse   = slot_is_sparse[cur_slot];
    assign hit_meta_stride = slot_meta_stride[cur_slot];
`endif

    // -----------------------------------------------------------------------
    // Performance counters
    // -----------------------------------------------------------------------

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] fetch_stall_ctr_r;
    reg [PERF_CTR_BITS-1:0] lmem_reads_ctr_r;
    reg [PERF_CTR_BITS-1:0] tbuf_tile_fetches_ctr_r;
    reg [PERF_CTR_BITS-1:0] tbuf_fetch_cycles_ctr_r;
    reg [PERF_CTR_BITS-1:0] fetch_b_cycles_ctr_r;
    reg [PERF_CTR_BITS-1:0] lmem_reads_a_ctr_r;
    reg [PERF_CTR_BITS-1:0] lmem_reads_b_ctr_r;
    reg [PERF_CTR_BITS-1:0] lmem_reads_meta_ctr_r;
    reg [PERF_CTR_BITS-1:0] lmem_rsp_stalls_ctr_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            fetch_stall_ctr_r       <= '0;
            lmem_reads_ctr_r        <= '0;
            tbuf_tile_fetches_ctr_r <= '0;
            tbuf_fetch_cycles_ctr_r <= '0;
            fetch_b_cycles_ctr_r    <= '0;
            lmem_reads_a_ctr_r      <= '0;
            lmem_reads_b_ctr_r      <= '0;
            lmem_reads_meta_ctr_r   <= '0;
            lmem_rsp_stalls_ctr_r   <= '0;
        end else begin
            if (req_valid && !tbuf_ready)
                fetch_stall_ctr_r <= fetch_stall_ctr_r + PERF_CTR_BITS'(1);
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                lmem_reads_ctr_r <= lmem_reads_ctr_r + PERF_CTR_BITS'(1);
            if (alloc_en)
                tbuf_tile_fetches_ctr_r <= tbuf_tile_fetches_ctr_r + PERF_CTR_BITS'(1);
            if (in_fetch)
                tbuf_fetch_cycles_ctr_r <= tbuf_fetch_cycles_ctr_r + PERF_CTR_BITS'(1);
            if (in_fetch_b)
                fetch_b_cycles_ctr_r <= fetch_b_cycles_ctr_r + PERF_CTR_BITS'(1);
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready && in_fetch_a)
                lmem_reads_a_ctr_r <= lmem_reads_a_ctr_r + PERF_CTR_BITS'(1);
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready && in_fetch_b)
                lmem_reads_b_ctr_r <= lmem_reads_b_ctr_r + PERF_CTR_BITS'(1);
`ifdef TCU_SPARSE_ENABLE
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready && in_fetch_meta)
                lmem_reads_meta_ctr_r <= lmem_reads_meta_ctr_r + PERF_CTR_BITS'(1);
`endif
            if (in_fetch && req_inflight_r && !tcu_lmem_if.rsp_valid)
                lmem_rsp_stalls_ctr_r <= lmem_rsp_stalls_ctr_r + PERF_CTR_BITS'(1);
        end
    end
    assign tbuf_stalls      = fetch_stall_ctr_r;
    assign tbuf_cache_hits  = '0;
    assign lmem_reads       = lmem_reads_ctr_r;
    assign tbuf_tile_fetches = tbuf_tile_fetches_ctr_r;
    assign tbuf_fetch_cycles = tbuf_fetch_cycles_ctr_r;
    assign fetch_b_cycles   = fetch_b_cycles_ctr_r;
    assign lmem_reads_a     = lmem_reads_a_ctr_r;
    assign lmem_reads_b     = lmem_reads_b_ctr_r;
    assign lmem_reads_meta  = lmem_reads_meta_ctr_r;
    assign lmem_rsp_stalls  = lmem_rsp_stalls_ctr_r;
`endif

    // -----------------------------------------------------------------------
    // Debug trace
    // -----------------------------------------------------------------------

`ifdef DBG_TRACE_TCU
    always @(posedge clk) begin
        if (!reset) begin
            if (alloc_en)
                `TRACE(3, ("%t: %s tbuf-fetch: alloc desc_a=0x%0h desc_b=0x%0h sparse=%0b cd_from_lmem=%0b desc_cd=0x%0h\n",
                    $time, INSTANCE_ID, req_desc_a, req_desc_b, is_sparse, req_cd_from_lmem, req_desc_cd))
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                `TRACE(3, ("%t: %s tbuf-fetch: rd_req addr=0x%0h phase=%0d req_ctr=%0d\n",
                    $time, INSTANCE_ID, tcu_lmem_if.req_data.addr, send_state_r, req_ctr_r))
            if (tcu_lmem_if.rsp_valid)
                `TRACE(3, ("%t: %s tbuf-fetch: rd_rsp data[0]=0x%0h phase=%0d rsp_ctr=%0d\n",
                    $time, INSTANCE_ID, tcu_lmem_if.rsp_data.data[0 +: `XLEN], send_state_r, rsp_ctr_r))
            if (fetch_done_now)
                `TRACE(3, ("%t: %s tbuf-fetch: READY\n", $time, INSTANCE_ID))
            if (last_rsp_c)
                `TRACE(3, ("%t: %s tbuf-fetch: C-FETCH-DONE (cd_from_lmem hit)\n", $time, INSTANCE_ID))
            if (c_all_done && slot_cd_from_lmem[cur_slot])
                `TRACE(3, ("%t: %s tbuf-fetch: STORE-D triggered (cd_from_lmem hit)\n", $time, INSTANCE_ID))
            if (last_store_d)
                `TRACE(3, ("%t: %s tbuf-fetch: STORE-D done\n", $time, INSTANCE_ID))
        end
    end
`endif

`ifdef DBG_TRACE_TCU_PREFETCH
    always @(posedge clk) begin
        if (!reset) begin
            if (prefetch_b_valid)
                `TRACE(2, ("%t: %s tbuf-fetch: PREFETCH_B queued desc=0x%0h row_base=%0d\n",
                    $time, INSTANCE_ID, prefetch_b_desc, pb_row_base_c))
            if (pb_valid_r && !b_ready_r && !slot_in_progress && (send_state_r == SEND_IDLE))
                `TRACE(2, ("%t: %s tbuf-fetch: PREFETCH_B standalone fetch started desc=0x%0h\n",
                    $time, INSTANCE_ID, pb_desc_r))
            if (last_rsp_b && is_b_prefetch_r)
                `TRACE(2, ("%t: %s tbuf-fetch: PREFETCH_B fetch done b_buf_desc=0x%0h\n",
                    $time, INSTANCE_ID, pb_desc_r))
            if (alloc_en && b_prefetch_hit_alloc)
                `TRACE(2, ("%t: %s tbuf-fetch: PREFETCH_B hit at alloc desc_b=0x%0h (skipping FETCH_B)\n",
                    $time, INSTANCE_ID, req_desc_b))
            if (alloc_en && !b_prefetch_hit_alloc && b_ready_r)
                `TRACE(2, ("%t: %s tbuf-fetch: PREFETCH_B miss at alloc req_desc_b=0x%0h b_buf_desc=0x%0h\n",
                    $time, INSTANCE_ID, req_desc_b, b_buf_desc_r))
        end
    end
`endif

`ifdef DBG_TCU_PERF
    always @(posedge clk) if (!reset && req_valid) begin
        $display("TCUPERF_FETCH,%0t,wid=%0d,cslot=%0d,fslot=%0d,fsm=%0d,sv=%b,sfd=%b,wap=%b,hit=%b,rdy=%b,alloc=%b,rq=%0d,rs=%0d,lrv=%b,lrr=%b,lsv=%b,fdn=%b,first=%b,dm=%b",
            $time, req_wid, cur_slot, fetch_slot, send_state_r,
            slot_valid, slot_fetch_done, warp_alloc_pending,
            tbuf_hit, tbuf_ready, alloc_en, req_ctr_r, rsp_ctr_r,
            tcu_lmem_if.req_valid, tcu_lmem_if.req_ready,
            tcu_lmem_if.rsp_valid, fetch_done_now, is_first_uop, desc_match);
    end
`endif

endmodule

`endif // TCU_WGMMA_ENABLE
