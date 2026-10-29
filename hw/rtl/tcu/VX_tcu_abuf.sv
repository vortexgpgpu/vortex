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
// Per-block A buffer (k-stripe storage, supports both block-major and
// row-major SMEM layouts).
//
// Holds the M_STEPS A-blocks for the current k-stripe of one warp slot.
// Storage is A_STRIPE_WORDS = M_STEPS × tcM × tcK 32-bit words, indexed
// by `row_idx * tcK + k_in` where row_idx ∈ [0, M_STEPS*tcM). The output
// mux is layout-agnostic — only the fetch path differs.
//
// Mode is selected by the descriptor's stride field (req_desc_a[31:16]):
//   stride == 0 → block-major. SMEM layout:
//     A_smem[(k_blk * M_STEPS + m_blk) * BLOCK_WORDS + i*TC_K + k_in]
//   stride != 0 → row-major (matches WGMMA SS descriptors and
//     DXA-loaded slabs). ldm_words = stride_bytes/4 = words per row.
//     A_smem[row * ldm_words + col_word].
//
// Block-major refill key = {desc_a, step_k}; one fetch reads M_STEPS
// adjacent bank-rows (one full stripe) starting at
//   base_a + step_k * M_STEPS * BLOCK_BANK_ROWS.
//
// Row-major refill key = {desc_a, step_k}; the FSM issues one LMEM read
// per row (M_STEPS*tcM total) at addr
//   base_a + (r * ldm_words + step_k * tcK) >> log2(BANK_ROW_WORDS)
// and extracts tcK consecutive words at lane
//   (r * ldm_words + step_k * tcK) & (BANK_ROW_WORDS-1).
//
// TMEM refill: req_desc_a is an absolute TMEM column. A's rows are this CTA's
// TCU_WG_TILE_M lanes, i.e. a whole bank word, so one bank read supplies
// TCU_TC_N of the stripe's FEDP_K columns and a refill is FEDP_K / TCU_TC_N
// reads (1, or 2 under FEDP2K).
//
// Output rs1_data carries one A-block per cycle, selected by step_m.
// Format-aware sub-word extraction (fp16/fp8 packing) happens
// downstream in the FEDP grid; abuf is word pass-through.
//

module VX_tcu_abuf import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID     = "",
    parameter         NUM_BANKS       = 4,
    parameter         BANK_ADDR_WIDTH = 12
) (
    input  wire clk,
    input  wire reset,

`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] abuf_stalls,
    output wire [PERF_CTR_BITS-1:0] lmem_reads,
`endif

    // Execute-side observation (req_valid is already gated to WGMMA)
    input  wire [NW_WIDTH-1:0]      req_wid,
    input  wire                     req_valid,
    input  wire                     req_setup,
    input  wire [1:0]               req_step_m,
    input  wire [6:0]               req_step_n,
    input  wire [1:0]               req_step_k,
    input  wire [`VX_CFG_XLEN-1:0]  req_desc_a,
    input  wire                     req_a_is_smem,
    input  wire                     req_is_sparse,
    input  wire [UUID_WIDTH-1:0]    req_uuid,

    // LMEM bank-parallel read port
    VX_mem_bus_if.master            tcu_lmem_if,

`ifdef VX_CFG_TCU_TMEM_ENABLE
    // TMEM bank read port. afill_grant arrives the cycle after the win with
    // afill_data.
    input  wire                     req_a_is_tmem,
    output wire                     afill_valid,
    output wire [TCU_TMEM_COL_BITS-1:0] afill_col_base,
    input  wire                     afill_grant,
    input  wire [TCU_WG_TILE_M-1:0][TCU_TC_N-1:0][31:0] afill_data,
`endif

    // Outputs
    output wire                     abuf_ready,
    output wire [TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] abuf_rs1_data
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (req_wid)

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    localparam BANK_SEL_BITS      = $clog2(NUM_BANKS);
    localparam WORD_SIZE_LOG2     = $clog2(`VX_CFG_XLEN / 8);
    localparam A_BLOCK_WORDS      = TCU_TC_M * TCU_WG_FEDP_K;
    localparam A_BLOCK_WORDS_SP   = TCU_TC_M * TCU_TC_K;
    localparam A_BLOCK_BANK_ROWS  = (A_BLOCK_WORDS + NUM_BANKS - 1) / NUM_BANKS;
    localparam A_BLOCK_BANK_ROWS_SP = (A_BLOCK_WORDS_SP + NUM_BANKS - 1) / NUM_BANKS;
    localparam BLOCK_WORDS_PADDED = A_BLOCK_BANK_ROWS * NUM_BANKS;
    localparam BLOCK_WORDS_PADDED_SP = A_BLOCK_BANK_ROWS_SP * NUM_BANKS;
    localparam A_STRIPE_BANK_ROWS = TCU_WG_M_STEPS * A_BLOCK_BANK_ROWS;
    localparam A_STRIPE_BANK_ROWS_SP = TCU_WG_M_STEPS * A_BLOCK_BANK_ROWS_SP;
    localparam A_STRIPE_WORDS     = A_STRIPE_BANK_ROWS * NUM_BANKS;
    // XLEN ratio: each physical LMEM bank-row carries XLEN_RATIO logical
    // 32-bit bank-rows side-by-side. Smem layout is XLEN-independent, so the
    // LMEM fetch count and stride must scale down by XLEN_RATIO.
    localparam XLEN_RATIO         = `VX_CFG_XLEN / 32;
    localparam BANK_ROW_WORDS     = NUM_BANKS * XLEN_RATIO; // 32-bit words per LMEM bank-row
    localparam BANK_ROW_WORDS_LOG2 = $clog2(BANK_ROW_WORDS);
    localparam A_STRIPE_LMEM_ROWS = (A_STRIPE_WORDS + BANK_ROW_WORDS - 1) / BANK_ROW_WORDS;
    localparam A_STRIPE_LMEM_ROWS_SP = (A_STRIPE_BANK_ROWS_SP * NUM_BANKS + BANK_ROW_WORDS - 1) / BANK_ROW_WORDS;
    // Row-major path issues one LMEM read per logical row (M_STEPS*TC_M rows).
    localparam A_TOTAL_ROWS       = TCU_WG_M_STEPS * TCU_TC_M;
    localparam FETCH_CTR_W_BM     = `CLOG2(A_STRIPE_LMEM_ROWS + 1);
    localparam FETCH_CTR_W_RM     = `CLOG2(A_TOTAL_ROWS + 1);
    localparam FETCH_CTR_W        = (FETCH_CTR_W_BM > FETCH_CTR_W_RM) ? FETCH_CTR_W_BM : FETCH_CTR_W_RM;
    localparam K_STEPS_W          = `CLOG2(TCU_WG_K_STEPS);
    // ldm field width: desc[31:16] is a 16-bit byte stride; >>2 gives the
    // per-row 32-bit-word stride. Cap at 14 bits (= 64KB smem / 4B).
    localparam LDM_W              = 14;

    // Canonical configs fit an A block in one bank-row normally and two
    // bank-rows with dense FEDP2K.

`ifdef VX_CFG_TCU_TMEM_ENABLE
    localparam AFILL_WORDS = TCU_WG_FEDP_K / TCU_TC_N;
    `STATIC_ASSERT ((TCU_WG_FEDP_K % TCU_TC_N) == 0,
        ("A-from-TMEM needs FEDP_K to be a whole number of TMEM bank words"))
    wire a_is_tmem = req_a_is_tmem;
`else
    wire a_is_tmem = 1'b0;
`endif

    // -----------------------------------------------------------------------
    // Resident slot state
    // -----------------------------------------------------------------------

    logic                       slot_valid_r;
    logic [BANK_ADDR_WIDTH-1:0] slot_desc_a_row_base_r;
    logic [`UP(K_STEPS_W)-1:0]  slot_step_k_r;
    logic                       slot_fetching_r;
    // Row-major mode: latched per-row stride. ldm_words=0 means block-major;
    // the fetch FSM falls through to the existing path. Latched at alloc_en
    // alongside fetch_base_r and slot_step_k_r.
    logic                       slot_row_major_r;
    logic [LDM_W-1:0]           slot_ldm_words_r;
    logic                       slot_is_sparse_r;
    // TMEM mode: the in-flight fetch reads TMEM rather than LMEM.
    logic                       slot_tmem_r;
`ifdef VX_CFG_TCU_TMEM_ENABLE
    logic [TCU_TMEM_COL_BITS-1:0] slot_tmem_col_r;   // a_col, latched on the first compute uop
    logic [TCU_TMEM_COL_BITS-1:0] fetch_tmem_col_r;  // in-flight stripe's first column
`endif

    wire [`UP(K_STEPS_W)-1:0]   req_step_k_trunc = `UP(K_STEPS_W)'(req_step_k);
    if (2 > K_STEPS_W) begin : g_step_k_upper_unused
        `UNUSED_VAR (req_step_k[1:`UP(K_STEPS_W)])
    end

    // The WGMMA wrapper supplies the setup-latched descriptor on compute uops.
    wire is_first_compute_uop = (req_step_m == '0) && (req_step_n == '0) && (req_step_k == '0);
    `UNUSED_VAR (req_step_n)  // only used in is_first_uop computation

    // Force a refetch on every WGMMA's first compute uop. A cooperative-load
    // pattern (K-tile loop rewrites A_warp_smem in place, issues a fresh
    // WGMMA with an unchanged descriptor) would incorrectly hit the cached
    // stripe and serve stale A data if residency were checked by desc_a alone.
    // refetched_for_first_compute_r gates stripe_resident: cleared until the
    // current WGMMA's first compute fetch completes, then set; cleared again
    // on later compute uops so the next WGMMA always triggers a fresh fetch.
    // TMEM-sourced A relies on this for correctness: the kernel rewrites A in
    // TMEM between UMMAs at an unchanged column.
    reg refetched_for_first_compute_r;
    wire stripe_resident = slot_valid_r
                        && (slot_step_k_r == req_step_k_trunc)
                        && (!is_first_compute_uop || refetched_for_first_compute_r);

    // WGMMA RS mode: A from registers, abuf bypassed -> always ready.
    wire uses_abuf  = req_a_is_smem || a_is_tmem;
    wire need_buf   = req_valid && uses_abuf;
    wire need_fetch = need_buf && !stripe_resident;
    wire alloc_en   = need_fetch && !slot_fetching_r;

    assign abuf_ready = !req_valid || !uses_abuf || stripe_resident;

    wire fire = req_valid && abuf_ready;

    // -----------------------------------------------------------------------
    // Address compute
    // -----------------------------------------------------------------------
    // base_a is the bank-row address of block(k=0, m=0). One k-stripe is
    // M_STEPS adjacent bank-rows starting at base + k * M_STEPS.

    localparam DESC_ADDR_W = BANK_ADDR_WIDTH + BANK_SEL_BITS;
    wire [DESC_ADDR_W-1:0]      desc_a_word_base = DESC_ADDR_W'(req_desc_a[15:0] >> WORD_SIZE_LOG2);
    wire [BANK_ADDR_WIDTH-1:0]  desc_a_row_base  = desc_a_word_base[BANK_SEL_BITS +: BANK_ADDR_WIDTH];
    if (BANK_SEL_BITS > 0) begin : g_addr_lsb_unused
        `UNUSED_VAR (desc_a_word_base[BANK_SEL_BITS-1:0])
    end
    // desc_a's lower 16 bits encode the smem offset; upper 16 bits carry the
    // per-row byte stride (matches WGMMA SS descriptor). Stride=0 keeps
    // the canonical block-major layout; non-zero selects the row-major path.
    wire [LDM_W-1:0] desc_a_ldm_words = LDM_W'(req_desc_a[31:16] >> 2);
    if (`VX_CFG_XLEN > 32) begin : g_desc_a_upper_unused
        `UNUSED_VAR (req_desc_a[`VX_CFG_XLEN-1:32])
    end

    localparam STRIPE_STRIDE_BANK_ROWS = A_STRIPE_LMEM_ROWS;

    // Compute uops use the setup-latched descriptor. Without this,
    // k-stripe-transition refills mid-WGMMA would compute the wrong fetch_base.
    wire [BANK_ADDR_WIDTH-1:0]  effective_desc_a_row_base =
        is_first_compute_uop ? desc_a_row_base : slot_desc_a_row_base_r;
    wire [LDM_W-1:0]            effective_ldm_words =
        is_first_compute_uop ? desc_a_ldm_words : slot_ldm_words_r;
    wire                        effective_row_major = (effective_ldm_words != '0);

    // Block-major stripe base (unchanged): one fetch covers M_STEPS A-blocks.
    wire [BANK_ADDR_WIDTH-1:0]  stripe_base =
        effective_desc_a_row_base
      + BANK_ADDR_WIDTH'(req_step_k_trunc) * BANK_ADDR_WIDTH'(STRIPE_STRIDE_BANK_ROWS);
    // M_STEPS, A_BLOCK_BANK_ROWS are compile-time; STRIPE_STRIDE_BANK_ROWS
    // is a power of 2 in canonical configs, so synth folds the multiply
    // into a shift.

`ifdef VX_CFG_TCU_TMEM_ENABLE
    wire [TCU_TMEM_COL_BITS-1:0] effective_tmem_col =
        is_first_compute_uop ? req_desc_a[TCU_TMEM_COL_BITS-1:0] : slot_tmem_col_r;
    wire [TCU_TMEM_COL_BITS-1:0] stripe_tmem_col = effective_tmem_col
        + TCU_TMEM_COL_BITS'(req_step_k_trunc) * TCU_TMEM_COL_BITS'(TCU_WG_FEDP_K);

    // A bank read addresses a whole word (col / TC_N), so a stripe that is not
    // word-aligned would read the wrong columns.
    `RUNTIME_ASSERT (~(alloc_en && a_is_tmem) || ((32'(stripe_tmem_col) % TCU_TC_N) == 0),
        ("%s: A-from-TMEM stripe column %0d is not aligned to TCU_TC_N (%0d)",
         INSTANCE_ID, stripe_tmem_col, TCU_TC_N))
    `RUNTIME_ASSERT (~(alloc_en && a_is_tmem) || ((32'(stripe_tmem_col) + TCU_WG_FEDP_K) <= TCU_TMEM_COLS),
        ("%s: A-from-TMEM stripe column %0d overruns TMEM (%0d columns)",
         INSTANCE_ID, stripe_tmem_col, TCU_TMEM_COLS))
`endif

    // -----------------------------------------------------------------------
    // Fetch FSM
    // -----------------------------------------------------------------------

    typedef enum logic {
        S_IDLE  = 1'b0,
        S_FETCH = 1'b1
    } state_e;
    state_e fsm_state_r;

    wire in_fetch = (fsm_state_r == S_FETCH);

    logic [FETCH_CTR_W-1:0]     req_ctr_r;
    logic [FETCH_CTR_W-1:0]     rsp_ctr_r;
    logic                       req_inflight_r;
    logic [BANK_ADDR_WIDTH-1:0] fetch_base_r;

    // Total fetches differs by mode:
    //   block-major: A_STRIPE_LMEM_ROWS adjacent bank-row reads
    //   row-major:   A_TOTAL_ROWS per-row reads (one per logical row)
    wire [FETCH_CTR_W-1:0] target_fetches = slot_row_major_r
        ? FETCH_CTR_W'(A_TOTAL_ROWS)
        : (slot_is_sparse_r ? FETCH_CTR_W'(A_STRIPE_LMEM_ROWS_SP)
                            : FETCH_CTR_W'(A_STRIPE_LMEM_ROWS));

    wire all_requested = (req_ctr_r >= target_fetches);
    wire lmem_fetch    = in_fetch && !slot_tmem_r;
    wire can_issue     = lmem_fetch && !all_requested
                      && (!req_inflight_r || tcu_lmem_if.rsp_valid);
    wire lmem_rsp      = lmem_fetch && tcu_lmem_if.rsp_valid;
    wire last_rsp      = lmem_rsp
                      && (rsp_ctr_r == target_fetches - FETCH_CTR_W'(1));

    // TMEM mode: one bank word per grant, req_ctr_r counting words taken. The
    // request stays up across the grant cycle and the TMEM side drops it, so
    // the address must not advance until the grant has been consumed.
`ifdef VX_CFG_TCU_TMEM_ENABLE
    wire afill_take = in_fetch && slot_tmem_r && afill_grant;
    wire last_afill = afill_take && (req_ctr_r == FETCH_CTR_W'(AFILL_WORDS - 1));
    assign afill_valid    = in_fetch && slot_tmem_r;
    assign afill_col_base = fetch_tmem_col_r
                          + TCU_TMEM_COL_BITS'(req_ctr_r) * TCU_TMEM_COL_BITS'(TCU_TC_N);
`else
    wire afill_take = 1'b0;
    wire last_afill = 1'b0;
`endif
    wire fetch_done = last_rsp || last_afill;

    // Per-row 32-bit-word offset (relative to fetch_base_r * BANK_ROW_WORDS).
    // Width = LDM (row stride) + counter + a bit for step_k*TC_K headroom.
    // The offset is linear in the fetch counter and the first issue is at
    // least one cycle after alloc: the step_k product is registered at alloc
    // and each request/response accumulates the row stride, keeping the
    // multiplier off the LMEM request-address and response-lane paths.
    localparam ROW_OFF_W = LDM_W + FETCH_CTR_W + 4;
    wire [3:0] row_k_words = slot_is_sparse_r ? 4'(TCU_TC_K) : 4'(TCU_WG_FEDP_K);
    wire [3:0] row_k_words_alloc = req_is_sparse ? 4'(TCU_TC_K) : 4'(TCU_WG_FEDP_K);

    wire [ROW_OFF_W-1:0] row_word_off_init =
        ROW_OFF_W'(req_step_k_trunc) * ROW_OFF_W'(row_k_words_alloc);

    logic [ROW_OFF_W-1:0] row_word_off_req_r;
    logic [ROW_OFF_W-1:0] row_word_off_rsp_r;

    wire [BANK_ADDR_WIDTH-1:0] row_lmem_addr =
        fetch_base_r + BANK_ADDR_WIDTH'(row_word_off_req_r >> BANK_ROW_WORDS_LOG2);

    // LMEM master driver
    assign tcu_lmem_if.req_valid       = can_issue;
    assign tcu_lmem_if.req_data.rw     = 1'b0;
    assign tcu_lmem_if.req_data.addr   = slot_row_major_r
        ? row_lmem_addr
        : (fetch_base_r + BANK_ADDR_WIDTH'(req_ctr_r));
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
            req_ctr_r              <= '0;
            rsp_ctr_r              <= '0;
            row_word_off_req_r     <= '0;
            row_word_off_rsp_r     <= '0;
            req_inflight_r         <= 1'b0;
            slot_valid_r           <= 1'b0;
            slot_fetching_r        <= 1'b0;
            slot_desc_a_row_base_r <= '0;
            slot_step_k_r          <= '0;
            fetch_base_r           <= '0;
            slot_row_major_r       <= 1'b0;
            slot_ldm_words_r       <= '0;
            slot_is_sparse_r       <= 1'b0;
            slot_tmem_r            <= 1'b0;
        `ifdef VX_CFG_TCU_TMEM_ENABLE
            slot_tmem_col_r        <= '0;
            fetch_tmem_col_r       <= '0;
        `endif
            refetched_for_first_compute_r <= 1'b0;
        end else begin
            if (req_setup) begin
                fsm_state_r      <= S_IDLE;
                req_ctr_r        <= '0;
                rsp_ctr_r        <= '0;
                req_inflight_r   <= 1'b0;
                slot_valid_r     <= 1'b0;
                slot_fetching_r  <= 1'b0;
                refetched_for_first_compute_r <= 1'b0;
            end else begin
                // Inflight tracker (single outstanding request at a time)
                if (tcu_lmem_if.rsp_valid)
                    req_inflight_r <= 1'b0;
                if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                    req_inflight_r <= 1'b1;

                // Latch descriptor fields on the first compute uop.
                if (req_valid && is_first_compute_uop) begin
                    slot_desc_a_row_base_r <= desc_a_row_base;
                    slot_ldm_words_r       <= desc_a_ldm_words;
                    slot_row_major_r       <= (desc_a_ldm_words != '0);
                `ifdef VX_CFG_TCU_TMEM_ENABLE
                    slot_tmem_col_r        <= req_desc_a[TCU_TMEM_COL_BITS-1:0];
                `endif
                end

                if (fetch_done && is_first_compute_uop)
                    refetched_for_first_compute_r <= 1'b1;
                else if (fire && !is_first_compute_uop)
                    refetched_for_first_compute_r <= 1'b0;

                case (fsm_state_r)
                    S_IDLE: begin
                        if (alloc_en) begin
                            fsm_state_r      <= S_FETCH;
                            slot_fetching_r  <= 1'b1;
                            slot_valid_r     <= 1'b0;
                            slot_step_k_r    <= req_step_k_trunc;
                            slot_is_sparse_r  <= req_is_sparse;
                            slot_tmem_r       <= a_is_tmem;
                        `ifdef VX_CFG_TCU_TMEM_ENABLE
                            fetch_tmem_col_r  <= stripe_tmem_col;
                        `endif
                            // Row-major base is the A_warp start bank-row (no
                            // step_k offset; per-row arithmetic derives the
                            // exact LMEM addr). Block-major base is the stripe
                            // origin (step_k already factored in).
                            fetch_base_r     <= effective_row_major
                                ? effective_desc_a_row_base
                                : stripe_base;
                            req_ctr_r        <= '0;
                            rsp_ctr_r        <= '0;
                            row_word_off_req_r <= row_word_off_init;
                            row_word_off_rsp_r <= row_word_off_init;
                            req_inflight_r   <= 1'b0;
                        end
                    end
                    S_FETCH: begin
                        if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready) begin
                            req_ctr_r <= req_ctr_r + FETCH_CTR_W'(1);
                            row_word_off_req_r <= row_word_off_req_r + ROW_OFF_W'(slot_ldm_words_r);
                        end
                        if (afill_take) begin
                            req_ctr_r <= req_ctr_r + FETCH_CTR_W'(1);
                        end
                        if (fetch_done) begin
                            fsm_state_r     <= S_IDLE;
                            slot_fetching_r <= 1'b0;
                            slot_valid_r    <= 1'b1;
                            req_inflight_r  <= 1'b0;
                        end else if (lmem_rsp) begin
                            rsp_ctr_r <= rsp_ctr_r + FETCH_CTR_W'(1);
                            row_word_off_rsp_r <= row_word_off_rsp_r + ROW_OFF_W'(slot_ldm_words_r);
                        end
                    end
                    default: fsm_state_r <= S_IDLE;
                endcase
            end
        end
    end

    // -----------------------------------------------------------------------
    // Storage (LUTRAM): A_STRIPE_BANK_ROWS × NUM_BANKS 32-bit words.
    // Each LMEM response writes one bank-row at offset rsp_ctr_r.
    // -----------------------------------------------------------------------

    logic [A_STRIPE_WORDS*32-1:0] storage_wdata;
    logic [A_STRIPE_WORDS-1:0]    storage_wren;

    // Row-major per-response decoding: the row being written is rsp_ctr_r,
    // and its tcK source words start at lane = row_word_off & (BANK_ROW_WORDS-1)
    // within the LMEM response (registered accumulator, see above).
    wire [BANK_ROW_WORDS_LOG2:0] row_lane_rsp = (BANK_ROW_WORDS_LOG2+1)'(
        row_word_off_rsp_r & ROW_OFF_W'(BANK_ROW_WORDS - 1));

    always_comb begin
        storage_wdata = '0;
        storage_wren  = '0;
    `ifdef VX_CFG_TCU_TMEM_ENABLE
        if (afill_take) begin
            // Bank word req_ctr_r carries stripe columns [req_ctr_r*TC_N, +TC_N)
            // of every row. Index through the per-block padding so the output
            // mux reads the same layout as the LMEM paths.
            for (int r = 0; r < TCU_WG_TILE_M; ++r) begin
                for (int c = 0; c < TCU_TC_N; ++c) begin
                    automatic int dst = (r / int'(TCU_TC_M)) * int'(BLOCK_WORDS_PADDED)
                                      + (r % int'(TCU_TC_M)) * int'(TCU_WG_FEDP_K)
                                      + int'(req_ctr_r) * int'(TCU_TC_N) + c;
                    if (dst < A_STRIPE_WORDS) begin
                        storage_wren[dst] = 1'b1;
                        storage_wdata[dst * 32 +: 32] = afill_data[r][c];
                    end
                end
            end
        end else
    `endif
        if (lmem_rsp) begin
            if (slot_row_major_r) begin
                // Row-major: write k_words of row rsp_ctr_r into
                // storage[rsp_ctr_r * k_words .. + k_words). Source words start
                // at row_lane_rsp inside the LMEM response.
                for (int k = 0; k < TCU_WG_FEDP_K; ++k) begin
                    automatic int dst = int'(rsp_ctr_r) * int'(row_k_words) + k;
                    automatic int src = int'(row_lane_rsp) + k;
                    if (k < int'(row_k_words) && dst < A_STRIPE_WORDS && src < BANK_ROW_WORDS) begin
                        storage_wren[dst] = 1'b1;
                        storage_wdata[dst * 32 +: 32] =
                            tcu_lmem_if.rsp_data.data[src * 32 +: 32];
                    end
                end
            end else begin
                // Block-major: one LMEM response carries BANK_ROW_WORDS
                // 32-bit words (= NUM_BANKS * XLEN_RATIO). Store all of them
                // starting at the current LMEM-row offset within the stripe.
                for (int b = 0; b < BANK_ROW_WORDS; ++b) begin
                    if (int'(rsp_ctr_r) * BANK_ROW_WORDS + b < A_STRIPE_WORDS) begin
                        storage_wren[int'(rsp_ctr_r) * BANK_ROW_WORDS + b] = 1'b1;
                        storage_wdata[(int'(rsp_ctr_r) * BANK_ROW_WORDS + b) * 32 +: 32] =
                            tcu_lmem_if.rsp_data.data[b * 32 +: 32];
                    end
                end
            end
        end
    end

    wire [A_STRIPE_WORDS-1:0][31:0] storage_rdata;

    VX_dp_ram #(
        .DATAW   (A_STRIPE_WORDS * 32),
        .SIZE    (1),
        .WRENW   (A_STRIPE_WORDS),
        .LUTRAM  (1),
        .OUT_REG (0),
        .RDW_MODE("W")
    ) storage_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (lmem_rsp || afill_take),
        .wren  (storage_wren),
        .waddr (1'b0),
        .wdata (storage_wdata),
        .raddr (1'b0),
        .rdata (storage_rdata)
    );

    // -----------------------------------------------------------------------
    // Output: select A-block based on step_m. Sparse data is stored compactly,
    // then expanded into FEDP_K row stride with the unavailable tail lanes zero.
    // -----------------------------------------------------------------------

    logic [TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] rs1_mux;
    always_comb begin
        rs1_mux = '0;
        for (int lane = 0; lane < TCU_WG_A_DATA_SIZE; ++lane) begin
            int src_idx;
            if (req_is_sparse) begin
                automatic int row = lane / int'(TCU_WG_FEDP_K);
                automatic int k   = lane % int'(TCU_WG_FEDP_K);
                if (row < int'(TCU_TC_M) && k < int'(TCU_TC_K)) begin
                    src_idx = int'(req_step_m) * int'(BLOCK_WORDS_PADDED_SP)
                            + row * int'(TCU_TC_K)
                            + k;
                    if (src_idx < int'(A_STRIPE_WORDS))
                        rs1_mux[lane] = `VX_CFG_XLEN'(storage_rdata[src_idx]);
                end
            end else if (lane < int'(A_BLOCK_WORDS)) begin
                src_idx = int'(req_step_m) * int'(BLOCK_WORDS_PADDED) + lane;
                if (src_idx < int'(A_STRIPE_WORDS))
                    rs1_mux[lane] = `VX_CFG_XLEN'(storage_rdata[src_idx]);
            end
        end
    end
    assign abuf_rs1_data = rs1_mux;

    // -----------------------------------------------------------------------
    // Performance counters
    // -----------------------------------------------------------------------

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] stall_ctr_r;
    reg [PERF_CTR_BITS-1:0] reads_ctr_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            stall_ctr_r <= '0;
            reads_ctr_r <= '0;
        end else begin
            if (req_valid && !abuf_ready)
                stall_ctr_r <= stall_ctr_r + PERF_CTR_BITS'(1);
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                reads_ctr_r <= reads_ctr_r + PERF_CTR_BITS'(1);
        end
    end
    assign abuf_stalls = stall_ctr_r;
    assign lmem_reads  = reads_ctr_r;
`endif

    // -----------------------------------------------------------------------
    // Debug trace
    // -----------------------------------------------------------------------

`ifdef DBG_TRACE_TCU
    always @(posedge clk) begin
        if (!reset) begin
            if (alloc_en)
                `TRACE(3, ("%t: %s abuf: alloc desc_a=0x%0h step_k=%0d base=0x%0h\n",
                    $time, INSTANCE_ID, req_desc_a, req_step_k, stripe_base))
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                `TRACE(3, ("%t: %s abuf: rd_req addr=0x%0h req_ctr=%0d\n",
                    $time, INSTANCE_ID, tcu_lmem_if.req_data.addr, req_ctr_r))
            if (tcu_lmem_if.rsp_valid)
                `TRACE(3, ("%t: %s abuf: rd_rsp rsp_ctr=%0d\n",
                    $time, INSTANCE_ID, rsp_ctr_r))
        `ifdef VX_CFG_TCU_TMEM_ENABLE
            if (afill_take)
                `TRACE(3, ("%t: %s abuf: tmem word=%0d col=0x%0h\n",
                    $time, INSTANCE_ID, req_ctr_r, afill_col_base))
        `endif
            if (fetch_done)
                `TRACE(3, ("%t: %s abuf: stripe READY\n", $time, INSTANCE_ID))
        end
    end
`endif

endmodule

`endif // VX_CFG_TCU_WGMMA_ENABLE
