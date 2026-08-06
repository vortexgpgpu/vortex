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

`ifdef VX_CFG_TCU_WGMMA_ENABLE

//
// TB-shared B buffer (block-major SMEM, 1 bank-row storage).
//
// Single instance per VX_tcu_unit. Holds the bank-row of B that contains
// the current (step_k, step_n) block. All Q tcu_cores read from the
// same buffer (structural fan-out, not arbitrated).
//
// For canonical configs where TC_K * TC_N < NUM_BANKS, one bank-row holds
// B_SUB_BLOCKS = NUM_BANKS / (TC_K * TC_N) consecutive (k,n) blocks.
// Refill key = {desc_b, bank_row_index} where
//   bank_row_index = (step_k * N_STEPS + step_n) >> LG_B_SUB_BLOCKS.
//
// The bus to tcu_core carries the whole bank-row; tcu_core's b_off
// (= step_n & (B_SUB_BLOCKS-1) << LG_B_BS) selects within.
//
// Block-major within-block layout:
//   B_smem[(k*N_STEPS+n) * BLOCK_WORDS + j*(TC_K*i_ratio) + k_in_elem]
// Each 32-bit word packs i_ratio K-elements at one (j, k_word) cell.
// This matches tcu_core's `b_col[k] = rs2_data[b_off + j*TC_K + k]` indexing,
// so bbuf is word pass-through (no format-aware extraction needed here).
//

module VX_tcu_bbuf import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID     = "",
    parameter         NUM_BANKS       = 4,
    parameter         BANK_ADDR_WIDTH = 12
) (
    input  wire clk,
    input  wire reset,

`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] bbuf_stalls,
    output wire [PERF_CTR_BITS-1:0] bbuf_cache_hits,
    output wire [PERF_CTR_BITS-1:0] lmem_reads,
`endif

    // TB-level uop observation (req_valid is already gated to WGMMA at wrapper)
    input  wire                     req_valid,
    input  wire                     req_setup,
    input  wire                     req_is_first_uop,
    input  wire                     req_is_sparse,
    input  wire [3:0]               req_step_m,
    input  wire [3:0]               req_step_k,
    input  wire [3:0]               req_step_n,
    input  wire [1:0]               req_cd_nregs,
    input  wire [NCTA_WIDTH-1:0]    req_cta_id,
    input  wire [`VX_CFG_XLEN-1:0]  req_desc_b,
    input  wire [UUID_WIDTH-1:0]    req_uuid,

    // LMEM bank-parallel read port
    VX_mem_bus_if.master            tcu_lmem_if,

    // Outputs (broadcast to all Q tcu_cores)
    output wire                     bbuf_ready,
    output wire [TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0] bbuf_rs2_data
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    localparam BANK_SEL_BITS      = $clog2(NUM_BANKS);
    localparam WORD_SIZE_LOG2     = $clog2(`VX_CFG_XLEN / 8);
    localparam B_BLOCK_WORDS      = TCU_WG_FEDP_K * TCU_TC_N;
    localparam B_BLOCK_WORDS_SP   = TCU_TC_K * TCU_TC_N;
    localparam B_BUF_WORDS        = NUM_BANKS;             // storage per slot
                                                           // (= 1 logical 32-bit bank-row)
    localparam LG_B_SUB_BLOCKS    = $clog2(TCU_WG_B_SUB_BLOCKS);
    // XLEN ratio: each physical LMEM bank-row carries (XLEN/32) logical
    // 32-bit bank-rows side-by-side.
    localparam XLEN_RATIO         = `VX_CFG_XLEN / 32;
    localparam LG_XLEN_RATIO      = (XLEN_RATIO > 1) ? $clog2(XLEN_RATIO) : 0;

    localparam LOGICAL_ROWS_PER_BLK = TCU_WG_B_BLOCK_SIZE_SP / NUM_BANKS;
    `STATIC_ASSERT (LOGICAL_ROWS_PER_BLK == 1 || LOGICAL_ROWS_PER_BLK == 2,
                    ("flat sparse-B supports 1 or 2 logical bank-rows per block"))
    localparam bit SPARSE_TWO_SLOT = (LOGICAL_ROWS_PER_BLK == 2);
    localparam bit SPARSE_TWO_FETCH = SPARSE_TWO_SLOT && (XLEN_RATIO == 1);

    // Canonical-config invariant: 1 logical (32-bit-equivalent) bank-row
    // holds B_SUB_BLOCKS blocks (the smem layout is XLEN-independent).
    `STATIC_ASSERT (B_BLOCK_WORDS * TCU_WG_B_SUB_BLOCKS == NUM_BANKS,
                    ("VX_tcu_bbuf assumes one bank-row per B_SUB_BLOCKS blocks"))

    // K-major (= row-major SMEM where K runs along contiguous bytes; the
    // WGMMA SS-descriptor's canonical layout) fetch path. Engaged
    // when desc_b's stride field (bits [31:16]) is non-zero. Performs
    // TCU_TC_N per-N-row LMEM reads per (step_k, step_n) WGMMA uop, writing
    // TCU_WG_FEDP_K 32-bit words per row into storage at the b_off offset
    // tcu_core's `b_off + j*tcK + k` indexing will read. Mirrors
    // VX_tcu_abuf.sv's row-major fetch for A.
    localparam LDM_W              = 14;
    localparam BANK_ROW_WORDS     = NUM_BANKS * XLEN_RATIO;
    localparam BANK_ROW_WORDS_LOG2= $clog2(BANK_ROW_WORDS);
    localparam LG_B_BLOCK_WORDS   = $clog2(B_BLOCK_WORDS);
    localparam KM_CTR_W           = $clog2(TCU_TC_N + 1);
    // 32-bit-word offset width for K-major address arithmetic.
    //   step_n × TCU_TC_N + j has ≤ 4+4 bits; × ldm_words adds LDM_W.
    localparam KM_OFF_W           = LDM_W + 4 + 4;

    // -----------------------------------------------------------------------
    // Block-index compute (variable N_STEPS via cd_nregs).
    // K_STEPS=2 always; N_STEPS=4/8/16 for cd_nregs=0/1/2 (NRC=8/16/32).
    // -----------------------------------------------------------------------

    logic [4:0] block_index;
    always_comb begin
        case (req_cd_nregs)
            2'd0:    block_index = {2'b0, req_step_k[0], req_step_n[1:0]};   // N_STEPS=4
            2'd1:    block_index = {1'b0, req_step_k[0], req_step_n[2:0]};   // N_STEPS=8
            default: block_index = {req_step_k[0], req_step_n[3:0]};         // N_STEPS=16
        endcase
    end
    if (4 > 1) begin : g_step_k_upper_unused
        `UNUSED_VAR (req_step_k[3:1])
    end

    // LMEM bank-row offset.
    //
    //   Dense:  one logical 32-bit bank-row holds B_SUB_BLOCKS dense
    //           blocks; XLEN_RATIO of them fit in one physical bank-row.
    //           So advance the LMEM addr every
    //             (B_SUB_BLOCKS * XLEN_RATIO) blocks.
    // Sparse blocks use the flat producer layout and can span one or two
    // logical 32-bit bank rows.
    localparam TOTAL_SHIFT = LG_B_SUB_BLOCKS + LG_XLEN_RATIO;
    wire [4:0] dense_offset = (TOTAL_SHIFT == 0)
                            ? block_index
                            : 5'(block_index >> TOTAL_SHIFT);

    // Dense within-physical-bank-row selector (XLEN>32 only). Picks which
    // of the XLEN_RATIO logical 32-bit bank-rows to copy into slot A.
    localparam SUB_HALF_W = (LG_XLEN_RATIO == 0) ? 1 : LG_XLEN_RATIO;
    wire [SUB_HALF_W-1:0] dense_sub_half =
        (LG_XLEN_RATIO == 0)
        ? '0
        : SUB_HALF_W'(({27'b0, block_index} >> LG_B_SUB_BLOCKS) & ((1 << LG_XLEN_RATIO) - 1));

    localparam SP_LOGROW_W = 6;
    wire [SP_LOGROW_W-1:0] sp_logrow_a =
        SP_LOGROW_W'(block_index) * SP_LOGROW_W'(LOGICAL_ROWS_PER_BLK);
    wire [SP_LOGROW_W-1:0] sp_logrow_b = sp_logrow_a + SP_LOGROW_W'(1);
    wire [SUB_HALF_W-1:0] sp_sub_a = (LG_XLEN_RATIO == 0) ? '0
                                   : SUB_HALF_W'(sp_logrow_a & SP_LOGROW_W'((1 << LG_XLEN_RATIO) - 1));
    wire [SUB_HALF_W-1:0] sp_sub_b = (LG_XLEN_RATIO == 0) ? '0
                                   : SUB_HALF_W'(sp_logrow_b & SP_LOGROW_W'((1 << LG_XLEN_RATIO) - 1));

    // -----------------------------------------------------------------------
    // Address compute (block-major)
    // -----------------------------------------------------------------------

    localparam DESC_ADDR_W = BANK_ADDR_WIDTH + BANK_SEL_BITS;
    wire [DESC_ADDR_W-1:0]      desc_b_word_base = DESC_ADDR_W'(req_desc_b[15:0] >> WORD_SIZE_LOG2);
    wire [BANK_ADDR_WIDTH-1:0]  desc_b_row_base  = desc_b_word_base[BANK_SEL_BITS +: BANK_ADDR_WIDTH];
    // desc_b's upper 16 bits encode the per-row byte stride (WGMMA
    // SS-descriptor `ldm`). Non-zero stride selects the K-major fetch path.
    wire [LDM_W-1:0] desc_b_ldm_words = LDM_W'(req_desc_b[31:16] >> 2);
    if (`VX_CFG_XLEN > 32) begin : g_desc_b_upper_unused
        `UNUSED_VAR (req_desc_b[`VX_CFG_XLEN-1:32])
    end
    if (BANK_SEL_BITS > 0) begin : g_addr_lsb_unused
        `UNUSED_VAR (desc_b_word_base[BANK_SEL_BITS-1:0])
    end

    // -----------------------------------------------------------------------
    // Resident slots
    //   slot A: holds k_blk=0 bank-row (dense uses this exclusively)
    //   slot B: holds k_blk=1 bank-row (sparse only)
    // -----------------------------------------------------------------------

    logic                       slot_a_valid_r;
    logic [BANK_ADDR_WIDTH-1:0] slot_a_addr_r;
    logic [BANK_ADDR_WIDTH-1:0] slot_desc_b_row_base_r;
    logic                       slot_fetching_r;
    // Dense within-physical-bank-row half (XLEN>32 only).
    logic [SUB_HALF_W-1:0]      slot_a_sub_half_r;
    // Second slot for sparse blocks spanning two logical rows.
    logic                       slot_b_valid_r;
    logic [BANK_ADDR_WIDTH-1:0] slot_b_addr_r;
    logic [SUB_HALF_W-1:0]      slot_b_sub_half_r;
    // Mode the slot pair was filled under (sparse vs dense). On mode
    // transition for the same warpgroup we must refill.
    logic                       slot_is_sparse_r;
    // K-major mode + per-WGMMA latched fields. Descriptor fields are latched
    // on setup; per-compute step fields are latched when a fetch is allocated.
    logic                       slot_row_major_r;
    logic [LDM_W-1:0]           slot_ldm_words_r;
    logic [3:0]                 slot_step_k_r;
    logic [3:0]                 slot_step_n_r;
    logic [NCTA_WIDTH-1:0]      slot_cta_id_r;
    // K-major descriptors can point to storage rewritten between WGMMA instructions.
    logic                       refetched_for_first_uop_r;
    // K-major multi-fetch counters (count up to TCU_TC_N requests / responses
    // per WGMMA uop, one per N-row of the (step_k, step_n) block).
    logic [KM_CTR_W-1:0]        km_req_ctr_r;
    logic [KM_CTR_W-1:0]        km_rsp_ctr_r;

    // The WGMMA wrapper supplies the setup-latched descriptor on compute uops.
    `UNUSED_VAR (req_step_m)
    wire [BANK_ADDR_WIDTH-1:0] effective_desc_b_row_base =
        req_is_first_uop ? desc_b_row_base : slot_desc_b_row_base_r;
    // K-major slot fields (slot_row_major_r / slot_ldm_words_r /
    // slot_step_k_r / slot_step_n_r) are latched at alloc_en — see the
    // always_ff below. The K-major addressing arithmetic reads them
    // directly. desc_b_ldm_words on the bus is used only for setup.

    // Per-mode fetch addresses.
    wire [BANK_ADDR_WIDTH-1:0] fetch_addr_dense =
        effective_desc_b_row_base + BANK_ADDR_WIDTH'(dense_offset);
    wire [BANK_ADDR_WIDTH-1:0] fetch_addr_a_sparse =
        effective_desc_b_row_base + BANK_ADDR_WIDTH'(sp_logrow_a >> LG_XLEN_RATIO);
    wire [BANK_ADDR_WIDTH-1:0] fetch_addr_b_sparse =
        effective_desc_b_row_base + BANK_ADDR_WIDTH'(sp_logrow_b >> LG_XLEN_RATIO);

    wire [BANK_ADDR_WIDTH-1:0] fetch_addr_a =
        req_is_sparse ? fetch_addr_a_sparse : fetch_addr_dense;

    wire bank_row_resident_dense =
        slot_a_valid_r && !slot_is_sparse_r
        && (slot_a_addr_r == fetch_addr_dense)
        && (slot_a_sub_half_r == dense_sub_half);

    wire bank_row_resident_sparse =
        slot_a_valid_r && slot_is_sparse_r
        && (slot_a_addr_r == fetch_addr_a_sparse)
        && (slot_a_sub_half_r == sp_sub_a)
        && (!SPARSE_TWO_SLOT || (slot_b_valid_r
                                 && (slot_b_addr_r == fetch_addr_b_sparse)
                                 && (slot_b_sub_half_r == sp_sub_b)));

    // K-major residency (dense + sparse): same (step_k, step_n) already in
    // the slot pair. For sparse, both slots must be filled because sparse B
    // is split across the two dense candidate streams.
    wire bank_row_resident_kmajor =
        slot_a_valid_r && slot_row_major_r
        && (!slot_is_sparse_r || slot_b_valid_r)
        && (!req_is_first_uop || refetched_for_first_uop_r)
        && (slot_cta_id_r == req_cta_id)
        && (slot_step_k_r == req_step_k)
        && (slot_step_n_r == req_step_n);

    // Block-major residency is the existing dense/sparse branch; K-major
    // overrides it when the first compute descriptor or latched slot mode
    // selects K-major.
    wire req_wants_kmajor =
        req_is_first_uop ? (desc_b_ldm_words != '0) : slot_row_major_r;

    wire bank_row_resident = req_wants_kmajor
        ? bank_row_resident_kmajor
        : (req_is_sparse ? bank_row_resident_sparse : bank_row_resident_dense);
    wire need_fetch        = req_valid && !bank_row_resident;
    wire alloc_en          = need_fetch && !slot_fetching_r;

    assign bbuf_ready = !req_valid || bank_row_resident;
    wire fire = req_valid && bbuf_ready;

    // -----------------------------------------------------------------------
    // K-major address generation
    // -----------------------------------------------------------------------
    // For row r (= km_req_ctr_r) of the current (step_k, step_n) block:
    //   word_off = (step_n × TCU_TC_N + r) × ldm_words + step_k × k_words
    //   bank_row = base + (word_off >> log2(BANK_ROW_WORDS))
    //   lane     = word_off & (BANK_ROW_WORDS - 1)
    // k_words is fedpK for dense and tcK for sparse

    wire [3:0] km_k_words = slot_is_sparse_r ? 4'(TCU_TC_K) : 4'(TCU_WG_FEDP_K);

    wire [KM_OFF_W-1:0] km_word_off_req =
        KM_OFF_W'(slot_step_n_r) * KM_OFF_W'(TCU_TC_N) * KM_OFF_W'(slot_ldm_words_r)
      + KM_OFF_W'(km_req_ctr_r) * KM_OFF_W'(slot_ldm_words_r)
      + KM_OFF_W'(slot_step_k_r) * KM_OFF_W'(km_k_words);
    wire [BANK_ADDR_WIDTH-1:0] km_lmem_addr =
        slot_desc_b_row_base_r + BANK_ADDR_WIDTH'(km_word_off_req >> BANK_ROW_WORDS_LOG2);

    wire [KM_OFF_W-1:0] km_word_off_rsp =
        KM_OFF_W'(slot_step_n_r) * KM_OFF_W'(TCU_TC_N) * KM_OFF_W'(slot_ldm_words_r)
      + KM_OFF_W'(km_rsp_ctr_r) * KM_OFF_W'(slot_ldm_words_r)
      + KM_OFF_W'(slot_step_k_r) * KM_OFF_W'(km_k_words);
    wire [BANK_ROW_WORDS_LOG2:0] km_lane_rsp = (BANK_ROW_WORDS_LOG2+1)'(
        km_word_off_rsp & KM_OFF_W'(BANK_ROW_WORDS - 1));

    // Storage offset where this K-major block lands (matches tcu_core's
    // b_off = (step_n & (B_SUB_BLOCKS-1)) << LG_B_BLOCK_WORDS so the
    // FEDP's `rs2[b_off + j*tcK + k]` reads what we wrote).
    localparam BUF_OFF_W = $clog2(B_BUF_WORDS);
    wire [BUF_OFF_W:0] km_b_off;
    if (LG_B_SUB_BLOCKS > 0) begin : g_km_b_off
        assign km_b_off = (BUF_OFF_W+1)'(slot_step_n_r[LG_B_SUB_BLOCKS-1:0])
                          << LG_B_BLOCK_WORDS;
    end else begin : g_km_b_off_zero
        assign km_b_off = '0;
    end

    // -----------------------------------------------------------------------
    // Fetch FSM
    //   S_IDLE  → S_FETCH_A → S_IDLE                          (dense)
    //   S_IDLE  → S_FETCH_A → S_FETCH_B → S_IDLE              (sparse)
    // -----------------------------------------------------------------------

    typedef enum logic [1:0] {
        S_IDLE     = 2'b00,
        S_FETCH_A  = 2'b01,
        S_FETCH_B  = 2'b10
    } state_e;
    state_e fsm_state_r;

    wire in_fetch_a = (fsm_state_r == S_FETCH_A);
    wire in_fetch_b = (fsm_state_r == S_FETCH_B);
    wire in_fetch   = in_fetch_a || in_fetch_b;
    logic req_inflight_r;

    // K-major: counter-gated multi-fire (TC_N requests per uop, single
    // outstanding). Block-major: original single-fire-per-state behavior
    // (km_req_ctr_r is reset on alloc; goes 0 → 1 for one block-major fire).
    wire km_more_to_request =
        slot_row_major_r ? (km_req_ctr_r < KM_CTR_W'(TCU_TC_N))
                         : (km_req_ctr_r == '0);
    wire can_issue  = in_fetch && !req_inflight_r && km_more_to_request;
    wire km_final_rsp = slot_row_major_r
                     && tcu_lmem_if.rsp_valid
                     && (km_rsp_ctr_r == KM_CTR_W'(TCU_TC_N - 1));
    wire bm_final_rsp = !slot_row_major_r && tcu_lmem_if.rsp_valid;
    wire last_rsp  = in_fetch && (km_final_rsp || bm_final_rsp);

    // Issue address: K-major path uses km_lmem_addr (re-derived per
    // km_req_ctr_r); block-major path uses the original slot_a/b addr.
    wire [BANK_ADDR_WIDTH-1:0] active_lmem_addr =
        slot_row_major_r ? km_lmem_addr
                         : (in_fetch_b ? slot_b_addr_r : slot_a_addr_r);

    assign tcu_lmem_if.req_valid       = can_issue;
    assign tcu_lmem_if.req_data.rw     = 1'b0;
    assign tcu_lmem_if.req_data.addr   = active_lmem_addr;
    assign tcu_lmem_if.req_data.data   = '0;
    assign tcu_lmem_if.req_data.byteen = '0;
    assign tcu_lmem_if.req_data.attr   = '0;
    assign tcu_lmem_if.req_data.tag.uuid  = req_uuid;
    assign tcu_lmem_if.req_data.tag.value = '0;
    assign tcu_lmem_if.rsp_ready       = 1'b1;
    `UNUSED_VAR (tcu_lmem_if.rsp_data.tag)

    always_ff @(posedge clk) begin
        if (reset) begin
            fsm_state_r            <= S_IDLE;
            req_inflight_r         <= 1'b0;
            slot_a_valid_r         <= 1'b0;
            slot_b_valid_r         <= 1'b0;
            slot_fetching_r        <= 1'b0;
            slot_a_addr_r          <= '0;
            slot_b_addr_r          <= '0;
            slot_desc_b_row_base_r <= '0;
            slot_a_sub_half_r      <= '0;
            slot_b_sub_half_r      <= '0;
            slot_is_sparse_r       <= 1'b0;
            slot_row_major_r       <= 1'b0;
            slot_ldm_words_r       <= '0;
            slot_step_k_r          <= '0;
            slot_step_n_r          <= '0;
            slot_cta_id_r          <= '0;
            refetched_for_first_uop_r <= 1'b0;
            km_req_ctr_r           <= '0;
            km_rsp_ctr_r           <= '0;
        end else begin
            if (req_setup) begin
                slot_a_valid_r  <= 1'b0;
                slot_b_valid_r  <= 1'b0;
                slot_fetching_r <= 1'b0;
                fsm_state_r     <= S_IDLE;
                req_inflight_r  <= 1'b0;
                km_req_ctr_r    <= '0;
                km_rsp_ctr_r    <= '0;
                refetched_for_first_uop_r <= 1'b0;
            end else begin
                if (tcu_lmem_if.rsp_valid)
                    req_inflight_r <= 1'b0;
                if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                    req_inflight_r <= 1'b1;

                // Latch descriptor fields on the first compute uop.
                if (req_valid && req_is_first_uop) begin
                    slot_desc_b_row_base_r <= desc_b_row_base;
                    slot_ldm_words_r       <= desc_b_ldm_words;
                    slot_row_major_r       <= (desc_b_ldm_words != '0);
                end

                if (last_rsp && req_is_first_uop)
                    refetched_for_first_uop_r <= 1'b1;
                else if (fire && !req_is_first_uop)
                    refetched_for_first_uop_r <= 1'b0;

                // K-major req/rsp counters advance independently of FSM state
                // (single-outstanding still enforced via req_inflight_r).
                if (slot_row_major_r && tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                    km_req_ctr_r <= km_req_ctr_r + KM_CTR_W'(1);
                if (slot_row_major_r && tcu_lmem_if.rsp_valid && !km_final_rsp)
                    km_rsp_ctr_r <= km_rsp_ctr_r + KM_CTR_W'(1);

                case (fsm_state_r)
                    S_IDLE: begin
                        if (alloc_en) begin
                            fsm_state_r         <= S_FETCH_A;
                            slot_fetching_r     <= 1'b1;
                            slot_a_valid_r      <= 1'b0;
                            slot_b_valid_r      <= 1'b0;
                            slot_a_addr_r       <= fetch_addr_a;
                            slot_a_sub_half_r   <= req_is_sparse ? sp_sub_a : dense_sub_half;
                            slot_b_addr_r       <= fetch_addr_b_sparse;
                            slot_b_sub_half_r   <= sp_sub_b;
                            slot_is_sparse_r    <= req_is_sparse;
                            // K-major per-compute slot state for this
                            // (step_k, step_n) block.
                            slot_step_k_r       <= req_step_k;
                            slot_step_n_r       <= req_step_n;
                            slot_cta_id_r       <= req_cta_id;
                            // slot_row_major_r is latched from the setup
                            // descriptor and reused for all compute refills.
                            km_req_ctr_r        <= '0;
                            km_rsp_ctr_r        <= '0;
                            req_inflight_r      <= 1'b0;
                        end
                    end
                    S_FETCH_A: begin
                        if (last_rsp) begin
                            slot_a_valid_r <= 1'b1;
                            req_inflight_r <= 1'b0;
                            if (slot_is_sparse_r && slot_row_major_r) begin
                                slot_b_valid_r  <= 1'b1;
                                fsm_state_r     <= S_IDLE;
                                slot_fetching_r <= 1'b0;
                            end else if (slot_is_sparse_r && SPARSE_TWO_FETCH) begin
                                fsm_state_r <= S_FETCH_B;
                            end else begin
                                if (slot_is_sparse_r && SPARSE_TWO_SLOT)
                                    slot_b_valid_r <= 1'b1;
                                fsm_state_r     <= S_IDLE;
                                slot_fetching_r <= 1'b0;
                            end
                        end
                    end
                    S_FETCH_B: begin
                        if (last_rsp) begin
                            fsm_state_r     <= S_IDLE;
                            slot_fetching_r <= 1'b0;
                            slot_b_valid_r  <= 1'b1;
                            req_inflight_r  <= 1'b0;
                        end
                    end
                    default: fsm_state_r <= S_IDLE;
                endcase
            end
        end
    end

    // -----------------------------------------------------------------------
    // Storage (LUTRAM): two slots × NUM_BANKS 32-bit words.
    //   slot A is always written on FETCH_A response (dense + sparse).
    //   slot B is written on FETCH_B response (sparse only).
    //
    //   Dense write:  copies one logical 32-bit bank-row (NUM_BANKS words)
    //                 from the physical response — picked by sub_half at
    //                 XLEN>32, the lower NUM_BANKS otherwise. tcu_core's
    //                 b_off then picks within those words at execute time.
    //
    //   Sparse write: extracts the compressed candidate lanes from one or two
    //                 dense B blocks and stores them in the compact rs2 order.
    // -----------------------------------------------------------------------

    logic [B_BUF_WORDS*32-1:0] storage_a_wdata, storage_b_wdata;
    logic [B_BUF_WORDS-1:0]    storage_a_wren,  storage_b_wren;

    // Per-slot logical-row offsets within a physical LMEM response.
    localparam OFF_W = $clog2(NUM_BANKS * XLEN_RATIO) + 1;
    wire [OFF_W-1:0] a_off_words = OFF_W'(slot_a_sub_half_r) * OFF_W'(NUM_BANKS);
    wire [OFF_W-1:0] b_off_words = OFF_W'(slot_b_sub_half_r) * OFF_W'(NUM_BANKS);
    wire sparse_b_from_a = SPARSE_TWO_SLOT && !SPARSE_TWO_FETCH
                          && slot_is_sparse_r && !slot_row_major_r;

    always_comb begin
        storage_a_wdata = '0;
        storage_a_wren  = '0;
        storage_b_wdata = '0;
        storage_b_wren  = '0;
        if (tcu_lmem_if.rsp_valid) begin
            if (slot_row_major_r && slot_is_sparse_r) begin
                // K-major sparse: per N-row r (= km_rsp_ctr_r = column j) the
                // response carries this column's K dimension z-major with the
                // two sparse candidates adjacent — word (z*2 + cand) holds the
                // FEDP's bword{cand} for K-step z.
                // The FEDP reads rs2[k_idx*(TC_N*2) + j*2 + cand] with k_idx=z;
                // slot_a||slot_b splits at B_BLOCK_WORDS_SP (= TC_K*TC_N).
                // Drive each (z,cand) word to its exact flat target and route by slot.
                for (int z = 0; z < TCU_TC_K; ++z) begin
                    for (int c = 0; c < 2; ++c) begin
                        automatic int src = int'(km_lane_rsp) + z * 2 + c;
                        automatic int tgt = z * (TCU_TC_N * 2) + int'(km_rsp_ctr_r) * 2 + c;
                        if (src < (NUM_BANKS * XLEN_RATIO) && in_fetch_a) begin
                            if (tgt < int'(B_BLOCK_WORDS_SP)) begin
                                if (tgt < B_BUF_WORDS) begin
                                    storage_a_wren[tgt]             = 1'b1;
                                    storage_a_wdata[tgt * 32 +: 32] =
                                        tcu_lmem_if.rsp_data.data[src * 32 +: 32];
                                end
                            end else begin
                                automatic int tgt_b = tgt - int'(B_BLOCK_WORDS_SP);
                                if (tgt_b < B_BUF_WORDS) begin
                                    storage_b_wren[tgt_b]             = 1'b1;
                                    storage_b_wdata[tgt_b * 32 +: 32] =
                                        tcu_lmem_if.rsp_data.data[src * 32 +: 32];
                                end
                            end
                        end
                    end
                end
            end else if (slot_row_major_r) begin
                // K-major dense: write fedpK words for this row (km_rsp_ctr_r)
                // into storage[km_b_off + km_rsp_ctr_r * fedpK .. + fedpK),
                // sourced from the response at lane km_lane_rsp.
                for (int k = 0; k < TCU_WG_FEDP_K; ++k) begin
                    automatic int dst = int'(km_b_off)
                                      + int'(km_rsp_ctr_r) * TCU_WG_FEDP_K
                                      + k;
                    automatic int src = int'(km_lane_rsp) + k;
                    if (dst < B_BUF_WORDS && src < (NUM_BANKS * XLEN_RATIO) && in_fetch_a) begin
                        storage_a_wren[dst]             = 1'b1;
                        storage_a_wdata[dst * 32 +: 32] =
                            tcu_lmem_if.rsp_data.data[src * 32 +: 32];
                    end
                end
            end else begin
                for (int b = 0; b < B_BUF_WORDS; ++b) begin
                    if (in_fetch_a) begin
                        storage_a_wren[b]             = 1'b1;
                        storage_a_wdata[b * 32 +: 32] =
                            tcu_lmem_if.rsp_data.data[(int'(a_off_words) + b) * 32 +: 32];
                        if (sparse_b_from_a) begin
                            storage_b_wren[b]             = 1'b1;
                            storage_b_wdata[b * 32 +: 32] =
                                tcu_lmem_if.rsp_data.data[(int'(b_off_words) + b) * 32 +: 32];
                        end
                    end
                    if (in_fetch_b) begin
                        storage_b_wren[b]             = 1'b1;
                        storage_b_wdata[b * 32 +: 32] =
                            tcu_lmem_if.rsp_data.data[(int'(b_off_words) + b) * 32 +: 32];
                    end
                end
            end
        end
    end

    wire [B_BUF_WORDS-1:0][31:0] storage_a_rdata, storage_b_rdata;

    VX_dp_ram #(
        .DATAW   (B_BUF_WORDS * 32),
        .SIZE    (1),
        .WRENW   (B_BUF_WORDS),
        .LUTRAM  (1),
        .OUT_REG (0),
        .RDW_MODE("W")
    ) storage_a_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (in_fetch_a && tcu_lmem_if.rsp_valid),
        .wren  (storage_a_wren),
        .waddr (1'b0),
        .wdata (storage_a_wdata),
        .raddr (1'b0),
        .rdata (storage_a_rdata)
    );

    VX_dp_ram #(
        .DATAW   (B_BUF_WORDS * 32),
        .SIZE    (1),
        .WRENW   (B_BUF_WORDS),
        .LUTRAM  (1),
        .OUT_REG (0),
        .RDW_MODE("W")
    ) storage_b_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write ((in_fetch_b && tcu_lmem_if.rsp_valid)
             || (in_fetch_a && tcu_lmem_if.rsp_valid && slot_is_sparse_r
                 && (slot_row_major_r || sparse_b_from_a))),
        .wren  (storage_b_wren),
        .waddr (1'b0),
        .wdata (storage_b_wdata),
        .raddr (1'b0),
        .rdata (storage_b_rdata)
    );

    // -----------------------------------------------------------------------
    // Output mux.
    //   Dense:  rs2[0..NUM_BANKS-1] = storage_A (legacy). tcu_core's b_off
    //           picks within at execute time.
    //   Sparse K-major is written directly in FEDP order. Block-major keeps
    //   the flat producer layout and applies a fixed read permutation.
    // -----------------------------------------------------------------------

    logic [TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0] rs2_mux;
    always_comb begin
        rs2_mux = '0;
        for (int lane = 0; lane < TCU_WG_RS2_WIDTH; ++lane) begin
            if (slot_is_sparse_r) begin
                if (slot_row_major_r) begin
                    if (lane < int'(B_BLOCK_WORDS_SP)) begin
                        rs2_mux[lane] = `VX_CFG_XLEN'(storage_a_rdata[lane]);
                    end else if (lane < int'(2 * B_BLOCK_WORDS_SP)) begin
                        rs2_mux[lane] = `VX_CFG_XLEN'(
                            storage_b_rdata[lane - int'(B_BLOCK_WORDS_SP)]);
                    end
                end else begin
                    automatic int unsigned k_idx_l = lane / (TCU_TC_N * 2);
                    automatic int unsigned rem_l   = lane % (TCU_TC_N * 2);
                    automatic int unsigned n_in_l  = rem_l / 2;
                    automatic int unsigned cand_l  = rem_l % 2;
                    automatic int unsigned w_l =
                        (k_idx_l * 2 + cand_l) * TCU_TC_N + n_in_l;
                    if (w_l < int'(B_BUF_WORDS)) begin
                        rs2_mux[lane] = `VX_CFG_XLEN'(storage_a_rdata[w_l]);
                    end else if (w_l < int'(2 * B_BLOCK_WORDS)) begin
                        rs2_mux[lane] = `VX_CFG_XLEN'(
                            storage_b_rdata[(w_l - int'(B_BUF_WORDS)) & (B_BUF_WORDS-1)]);
                    end
                end
            end else if (lane < int'(B_BUF_WORDS)) begin
                rs2_mux[lane] = `VX_CFG_XLEN'(storage_a_rdata[lane]);
            end
        end
    end
    assign bbuf_rs2_data = rs2_mux;

    // -----------------------------------------------------------------------
    // Performance counters
    // -----------------------------------------------------------------------

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] stall_ctr_r;
    reg [PERF_CTR_BITS-1:0] hits_ctr_r;
    reg [PERF_CTR_BITS-1:0] reads_ctr_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            stall_ctr_r <= '0;
            hits_ctr_r  <= '0;
            reads_ctr_r <= '0;
        end else begin
            // Stall: a request is pending and the resident bank-row doesn't match.
            if (req_valid && !bbuf_ready)
                stall_ctr_r <= stall_ctr_r + PERF_CTR_BITS'(1);
            // Hit: a request is pending and the resident bank-row already serves
            // it (no LMEM refill triggered). Counts cycles of bbuf reuse — a
            // direct measure of CTA-internal B-tile sharing across warps.
            if (req_valid && bank_row_resident)
                hits_ctr_r <= hits_ctr_r + PERF_CTR_BITS'(1);
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                reads_ctr_r <= reads_ctr_r + PERF_CTR_BITS'(1);
        end
    end
    assign bbuf_stalls     = stall_ctr_r;
    assign bbuf_cache_hits = hits_ctr_r;
    assign lmem_reads      = reads_ctr_r;
`endif

    // -----------------------------------------------------------------------
    // Debug trace
    // -----------------------------------------------------------------------

`ifdef DBG_TRACE_TCU
    always @(posedge clk) begin
        if (!reset) begin
            if (alloc_en)
                `TRACE(3, ("%t: %s bbuf: alloc desc_b=0x%0h sparse=%0d step_k=%0d step_n=%0d addr_a=0x%0h addr_b=0x%0h sub_half=%0d\n",
                    $time, INSTANCE_ID, req_desc_b, req_is_sparse, req_step_k, req_step_n,
                    fetch_addr_a, fetch_addr_b_sparse, dense_sub_half))
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                `TRACE(3, ("%t: %s bbuf: rd_req addr=0x%0h\n",
                    $time, INSTANCE_ID, tcu_lmem_if.req_data.addr))
            if (tcu_lmem_if.rsp_valid)
                `TRACE(3, ("%t: %s bbuf: rd_rsp\n", $time, INSTANCE_ID))
            if (last_rsp)
                `TRACE(3, ("%t: %s bbuf: bank-row READY\n", $time, INSTANCE_ID))
        end
    end
`endif

endmodule

`endif // VX_CFG_TCU_WGMMA_ENABLE
