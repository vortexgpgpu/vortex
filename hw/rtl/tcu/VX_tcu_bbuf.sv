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
// Block-major within-block layout (per docs/proposals/wgmma_simx_v3_addendum §3.2):
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
    input  wire                     req_is_first_uop,
    input  wire                     req_is_sparse,
    input  wire [3:0]               req_step_m,
    input  wire [3:0]               req_step_k,
    input  wire [3:0]               req_step_n,
    input  wire [1:0]               req_cd_nregs,
    input  wire [`VX_CFG_XLEN-1:0]         req_desc_b,

    // LMEM bank-parallel read port
    VX_mem_bus_if.master            tcu_lmem_if,

    // Outputs (broadcast to all Q tcu_cores)
    output wire                                       bbuf_ready,
    output wire [TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0]     bbuf_rs2_data
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    localparam BANK_SEL_BITS      = $clog2(NUM_BANKS);
    localparam WORD_SIZE_LOG2     = $clog2(`VX_CFG_XLEN / 8);
    localparam B_BLOCK_WORDS      = TCU_TC_K * TCU_TC_N;
    localparam B_BUF_WORDS        = NUM_BANKS;             // storage per slot
                                                           // (= 1 logical 32-bit bank-row)
    localparam LG_B_SUB_BLOCKS    = $clog2(TCU_WG_B_SUB_BLOCKS);
    // XLEN ratio: each physical LMEM bank-row carries (XLEN/32) logical
    // 32-bit bank-rows side-by-side.
    localparam XLEN_RATIO         = `VX_CFG_XLEN / 32;
    localparam LG_XLEN_RATIO      = (XLEN_RATIO > 1) ? $clog2(XLEN_RATIO) : 0;

    // How many dense blocks fit in one physical LMEM bank-row.
    //   = TCU_WG_B_SUB_BLOCKS × XLEN_RATIO
    localparam DENSE_BLOCKS_PER_ROW = TCU_WG_B_SUB_BLOCKS * XLEN_RATIO;
    localparam LG_DENSE_BLOCKS_PER_ROW = (DENSE_BLOCKS_PER_ROW > 1)
                                       ? $clog2(DENSE_BLOCKS_PER_ROW) : 0;

    // Canonical-config invariant: 1 logical (32-bit-equivalent) bank-row
    // holds B_SUB_BLOCKS blocks (the smem layout is XLEN-independent).
    `STATIC_ASSERT (B_BLOCK_WORDS * TCU_WG_B_SUB_BLOCKS == NUM_BANKS,
                    ("VX_tcu_bbuf assumes one bank-row per B_SUB_BLOCKS blocks"))

    // -----------------------------------------------------------------------
    // Block-index compute (variable N_STEPS via cd_nregs).
    // Vortex WGMMA uop expansion uses K_STEPS=2 always; N_STEPS=4/8/16
    // for cd_nregs=0/1/2 (NRC=8/16/32).
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
    //   Sparse: a single sparse step_n needs TC_K*TC_N*2 = 2*B_BLOCK_WORDS
    //           32-bit words, split across TWO dense K-blocks at the SAME
    //           n_blk. Those two blocks live in different physical bank-rows
    //           separated by sp_k_stride (in bank-rows). The two bbuf slots
    //           hold one block each; the position WITHIN each bank-row is
    //           selected by sparse_pos.
    localparam TOTAL_SHIFT = LG_B_SUB_BLOCKS + LG_XLEN_RATIO;
    wire [4:0] dense_offset = (TOTAL_SHIFT == 0)
                            ? block_index
                            : 5'(block_index >> TOTAL_SHIFT);

    // For sparse: step_n indexes n_blk; the shift to LMEM bank-row is the
    // same as dense (DENSE_BLOCKS_PER_ROW blocks per physical bank-row).
    wire [4:0] sparse_offset_a = (TOTAL_SHIFT == 0)
                               ? {1'b0, req_step_n}
                               : 5'({1'b0, req_step_n} >> TOTAL_SHIFT);

    // Sparse K-block stride in physical LMEM bank-rows.
    //   = n_steps * B_BLOCK_WORDS / (NUM_BANKS * XLEN_RATIO)
    // n_steps = NRC / 2 for the canonical (tcM*tcN == BLOCK_CAP) configs:
    //   NRC=8 → 4, NRC=16 → 8, NRC=32 → 16.
    // Old hardcoded table only matched NT=8; NT=16 has B_BLOCK_WORDS == NUM_BANKS
    // and so needs strides 2× larger. NT=32 happens to match NT=8.
    localparam SP_STRIDE_DEN = NUM_BANKS * XLEN_RATIO;
    localparam SP_STRIDE_NR8  = (4  * B_BLOCK_WORDS) / SP_STRIDE_DEN;
    localparam SP_STRIDE_NR16 = (8  * B_BLOCK_WORDS) / SP_STRIDE_DEN;
    localparam SP_STRIDE_NR32 = (16 * B_BLOCK_WORDS) / SP_STRIDE_DEN;
    logic [5:0] sp_k_stride;
    always_comb begin
        case (req_cd_nregs)
            2'd0:    sp_k_stride = 6'(SP_STRIDE_NR8);
            2'd1:    sp_k_stride = 6'(SP_STRIDE_NR16);
            default: sp_k_stride = 6'(SP_STRIDE_NR32);
        endcase
    end

    // Dense within-physical-bank-row selector (XLEN>32 only). Picks which
    // of the XLEN_RATIO logical 32-bit bank-rows to copy into slot A.
    localparam SUB_HALF_W = (LG_XLEN_RATIO == 0) ? 1 : LG_XLEN_RATIO;
    wire [SUB_HALF_W-1:0] dense_sub_half =
        (LG_XLEN_RATIO == 0)
        ? '0
        : SUB_HALF_W'(({27'b0, block_index} >> LG_B_SUB_BLOCKS) & ((1 << LG_XLEN_RATIO) - 1));

    // Sparse within-physical-bank-row selector. Picks which of the
    // DENSE_BLOCKS_PER_ROW dense blocks to extract (B_BLOCK_WORDS 32-bit
    // words at offset sparse_pos * B_BLOCK_WORDS in the LMEM response).
    localparam SPARSE_POS_W = (LG_DENSE_BLOCKS_PER_ROW == 0) ? 1 : LG_DENSE_BLOCKS_PER_ROW;
    wire [SPARSE_POS_W-1:0] sparse_pos_w =
        (LG_DENSE_BLOCKS_PER_ROW == 0)
        ? '0
        : SPARSE_POS_W'({27'b0, req_step_n} & ((1 << LG_DENSE_BLOCKS_PER_ROW) - 1));

    // -----------------------------------------------------------------------
    // Address compute (block-major)
    // -----------------------------------------------------------------------

    localparam DESC_ADDR_W = BANK_ADDR_WIDTH + BANK_SEL_BITS;
    wire [DESC_ADDR_W-1:0]      desc_b_word_base = DESC_ADDR_W'(req_desc_b[15:0] >> WORD_SIZE_LOG2);
    wire [BANK_ADDR_WIDTH-1:0]  desc_b_row_base  = desc_b_word_base[BANK_SEL_BITS +: BANK_ADDR_WIDTH];
    `UNUSED_VAR (req_desc_b[`VX_CFG_XLEN-1:16])
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
    // Sparse within-physical-bank-row block selector.
    logic [SPARSE_POS_W-1:0]    slot_a_sparse_pos_r;
    // Second slot for sparse k_blk=1 bank-row.
    logic                       slot_b_valid_r;
    logic [BANK_ADDR_WIDTH-1:0] slot_b_addr_r;
    logic [SPARSE_POS_W-1:0]    slot_b_sparse_pos_r;
    // Mode the slot pair was filled under (sparse vs dense). On mode
    // transition for the same warpgroup we must refill.
    logic                       slot_is_sparse_r;

    // The uop expander only reads rs2 (desc_b) on uop 0 of a WGMMA expansion
    // (VX_tcu_uops.sv:369, used_rs[1] = (wg_idx_ctr == '0)). On non-first uops,
    // req_desc_b is the bus's residual value (garbage). Latch desc_b on first
    // uop and use the latched base for subsequent uops in the same WGMMA.
    // C4: is_first_uop is now provided by op_args.tcu (single source of
    // truth — VX_tcu_uops sets it alongside fu_lock), not re-derived here.
    `UNUSED_VAR (req_step_m)
    wire is_first_uop = req_is_first_uop;
    wire [BANK_ADDR_WIDTH-1:0] effective_desc_b_row_base =
        is_first_uop ? desc_b_row_base : slot_desc_b_row_base_r;

    // Per-mode fetch addresses.
    wire [BANK_ADDR_WIDTH-1:0] fetch_addr_dense =
        effective_desc_b_row_base + BANK_ADDR_WIDTH'(dense_offset);
    wire [BANK_ADDR_WIDTH-1:0] fetch_addr_a_sparse =
        effective_desc_b_row_base + BANK_ADDR_WIDTH'(sparse_offset_a);
    wire [BANK_ADDR_WIDTH-1:0] fetch_addr_b_sparse =
        fetch_addr_a_sparse + BANK_ADDR_WIDTH'(sp_k_stride);

    wire [BANK_ADDR_WIDTH-1:0] fetch_addr_a =
        req_is_sparse ? fetch_addr_a_sparse : fetch_addr_dense;

    wire bank_row_resident_dense =
        slot_a_valid_r && !slot_is_sparse_r
        && (slot_a_addr_r == fetch_addr_dense)
        && (slot_a_sub_half_r == dense_sub_half);

    wire bank_row_resident_sparse =
        slot_a_valid_r && slot_b_valid_r && slot_is_sparse_r
        && (slot_a_addr_r == fetch_addr_a_sparse)
        && (slot_a_sparse_pos_r == sparse_pos_w)
        && (slot_b_addr_r == fetch_addr_b_sparse)
        && (slot_b_sparse_pos_r == sparse_pos_w);

    wire bank_row_resident = req_is_sparse ? bank_row_resident_sparse
                                           : bank_row_resident_dense;
    wire need_fetch        = req_valid && !bank_row_resident;
    wire alloc_en          = need_fetch && !slot_fetching_r;

    assign bbuf_ready = !req_valid || bank_row_resident;

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

    wire can_issue = in_fetch && !req_inflight_r;
    wire last_rsp  = in_fetch && tcu_lmem_if.rsp_valid;

    // Issue address: slot A's addr in FETCH_A, slot B's in FETCH_B.
    wire [BANK_ADDR_WIDTH-1:0] active_lmem_addr =
        in_fetch_b ? slot_b_addr_r : slot_a_addr_r;

    assign tcu_lmem_if.req_valid       = can_issue;
    assign tcu_lmem_if.req_data.rw     = 1'b0;
    assign tcu_lmem_if.req_data.addr   = active_lmem_addr;
    assign tcu_lmem_if.req_data.data   = '0;
    assign tcu_lmem_if.req_data.byteen = '0;
    assign tcu_lmem_if.req_data.attr   = '0;
    assign tcu_lmem_if.req_data.tag    = '0;
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
            slot_a_sparse_pos_r    <= '0;
            slot_b_sparse_pos_r    <= '0;
            slot_is_sparse_r       <= 1'b0;
        end else begin
            if (tcu_lmem_if.rsp_valid)
                req_inflight_r <= 1'b0;
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                req_inflight_r <= 1'b1;

            // Latch desc_b on first uop of every WGMMA so non-first uops can
            // re-derive fetch_addr without needing the (gated) req_desc_b bus.
            if (req_valid && is_first_uop)
                slot_desc_b_row_base_r <= desc_b_row_base;

            case (fsm_state_r)
                S_IDLE: begin
                    if (alloc_en) begin
                        fsm_state_r         <= S_FETCH_A;
                        slot_fetching_r     <= 1'b1;
                        slot_a_valid_r      <= 1'b0;
                        slot_b_valid_r      <= 1'b0;
                        slot_a_addr_r       <= fetch_addr_a;
                        slot_a_sub_half_r   <= dense_sub_half;
                        slot_a_sparse_pos_r <= sparse_pos_w;
                        slot_b_addr_r       <= fetch_addr_b_sparse;
                        slot_b_sparse_pos_r <= sparse_pos_w;
                        slot_is_sparse_r    <= req_is_sparse;
                        req_inflight_r      <= 1'b0;
                    end
                end
                S_FETCH_A: begin
                    if (last_rsp) begin
                        slot_a_valid_r <= 1'b1;
                        req_inflight_r <= 1'b0;
                        if (slot_is_sparse_r) begin
                            fsm_state_r <= S_FETCH_B;
                        end else begin
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
    //   Sparse write: copies exactly one dense block (B_BLOCK_WORDS words)
    //                 from the physical response at offset
    //                 sparse_pos × B_BLOCK_WORDS. The other words in the
    //                 physical bank-row are dropped — a different step_n
    //                 needs them and will refill on the next request. The
    //                 upper (NUM_BANKS - B_BLOCK_WORDS) storage entries are
    //                 unused in sparse mode but the dense width is retained
    //                 so the LUTRAM can serve both modes.
    // -----------------------------------------------------------------------

    logic [B_BUF_WORDS*32-1:0] storage_a_wdata, storage_b_wdata;
    logic [B_BUF_WORDS-1:0]    storage_a_wren,  storage_b_wren;

    // Per-slot extraction offset (in 32-bit-word units) within the physical
    // LMEM response, plus how many 32-bit words to copy. Computed outside
    // the comb block so all code paths assign a defined value.
    //
    // Width covers the worst case: NUM_BANKS * XLEN_RATIO 32-bit words per
    // physical LMEM response. Use +1 to leave headroom for the multiply
    // and avoid silent truncation when NUM_BANKS itself is a power of two
    // (e.g. NUM_BANKS=32 → 5-bit raw value overflows on 5'(32) literal).
    localparam OFF_W = $clog2(NUM_BANKS * XLEN_RATIO) + 1;
    wire [OFF_W-1:0] a_off_words = slot_is_sparse_r
                                 ? (OFF_W'(slot_a_sparse_pos_r) * OFF_W'(B_BLOCK_WORDS))
                                 : (OFF_W'(slot_a_sub_half_r)   * OFF_W'(NUM_BANKS));
    wire [OFF_W-1:0] b_off_words = slot_is_sparse_r
                                 ? (OFF_W'(slot_b_sparse_pos_r) * OFF_W'(B_BLOCK_WORDS))
                                 : '0;
    wire [OFF_W-1:0] write_count = slot_is_sparse_r ? OFF_W'(B_BLOCK_WORDS)
                                                    : OFF_W'(NUM_BANKS);

    // Sparse storage layout fix:
    //   B_smem block is J-major within block (n_in outer, k_word inner),
    //   so a linear copy gives storage_a[b] = block[b] = (j = b/tcK, k_word = b%tcK).
    //   But the FEDP indexes rs2[k*tcN*2 + j*2 + cand], i.e. K-major within
    //   the storage. For NT=8 (small dims) these coincide; for NT=16/32
    //   they diverge and B comes out wrong. Permute on write so that
    //   storage[b] holds the block word the FEDP will read at slot b:
    //       storage[k_pair*tcN*2 + j*2 + cand] = block[j*tcK + k_pair*2 + cand]
    //   i.e. src(b) = (b/2/tcN)*2 + (b%2) + ((b/2)%tcN)*tcK.
    logic [OFF_W-1:0] sparse_src [B_BUF_WORDS];
    always_comb begin
        for (int b = 0; b < B_BUF_WORDS; ++b) begin
            automatic int unsigned cand_b   = b & 1;
            automatic int unsigned j_b      = (b >> 1) % TCU_TC_N;
            automatic int unsigned k_pair_b = (b >> 1) / TCU_TC_N;
            sparse_src[b] = OFF_W'(j_b * TCU_TC_K + k_pair_b * 2 + cand_b);
        end
    end

    always_comb begin
        storage_a_wdata = '0;
        storage_a_wren  = '0;
        storage_b_wdata = '0;
        storage_b_wren  = '0;
        if (tcu_lmem_if.rsp_valid) begin
            for (int b = 0; b < B_BUF_WORDS; ++b) begin
                if (b < int'(write_count)) begin
                    automatic logic [OFF_W-1:0] src_off = slot_is_sparse_r
                                                       ? sparse_src[b]
                                                       : OFF_W'(b);
                    if (in_fetch_a) begin
                        storage_a_wren[b]             = 1'b1;
                        storage_a_wdata[b * 32 +: 32] =
                            tcu_lmem_if.rsp_data.data[(int'(a_off_words) + int'(src_off)) * 32 +: 32];
                    end
                    if (in_fetch_b) begin
                        storage_b_wren[b]             = 1'b1;
                        storage_b_wdata[b * 32 +: 32] =
                            tcu_lmem_if.rsp_data.data[(int'(b_off_words) + int'(src_off)) * 32 +: 32];
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
        .write (in_fetch_b && tcu_lmem_if.rsp_valid),
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
    //   Sparse: rs2[0..B_BLOCK_WORDS-1]               = storage_A[0..B-1]
    //           rs2[B_BLOCK_WORDS..2*B_BLOCK_WORDS-1] = storage_B[0..B-1]
    //           Each slot already holds exactly one dense block at the
    //           sparse_pos selected at fetch time, so the mux is a static
    //           concat. This matches tcu_core's sparse indexing
    //             rs2[k_idx*TC_N*2 + j*2 + cand].
    // -----------------------------------------------------------------------

    logic [TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0] rs2_mux;
    always_comb begin
        rs2_mux = '0;
        for (int lane = 0; lane < TCU_WG_RS2_WIDTH; ++lane) begin
            if (slot_is_sparse_r) begin
                if (lane < int'(B_BLOCK_WORDS)) begin
                    rs2_mux[lane] = `VX_CFG_XLEN'(storage_a_rdata[lane]);
                end else if (lane < int'(2 * B_BLOCK_WORDS)) begin
                    rs2_mux[lane] = `VX_CFG_XLEN'(
                        storage_b_rdata[lane - int'(B_BLOCK_WORDS)]);
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
                `TRACE(3, ("%t: %s bbuf: alloc desc_b=0x%0h sparse=%0d step_k=%0d step_n=%0d addr_a=0x%0h addr_b=0x%0h sub_half=%0d sparse_pos=%0d\n",
                    $time, INSTANCE_ID, req_desc_b, req_is_sparse, req_step_k, req_step_n,
                    fetch_addr_a, fetch_addr_b_sparse, dense_sub_half, sparse_pos_w))
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
