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
// TB-shared B buffer (1 bank-row storage per slot).
//
// Single instance per VX_tcu_unit. Holds the bank-row(s) of B that contain
// the current (step_k, step_n) block. All Q tcu_cores read from the same
// buffer (structural fan-out, not arbitrated).
//
// Two B layouts, selected at runtime by req_is_sparse:
//
//   Dense (block-major): one logical 32-bit bank-row holds B_SUB_BLOCKS
//     consecutive (k,n) blocks. The bank-row is stored verbatim into slot A;
//     tcu_core's b_off picks the block within at execute time.
//
//   Sparse (flat candidate-pair): each (step_k, step_n) block occupies two
//     contiguous physical bank-rows laid out in FEDP candidate-pair order
//     (matches vx_tensor.h b_sp_flat_idx). The two rows are stored verbatim
//     into slots A and B; a fixed read permutation (constant wiring) presents
//     them to the FEDP as rs2[k_idx*TC_N*2 + n_in*2 + cand]. No transpose
//     crossbar — store-and-read are both straight wires.
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
    localparam B_BLOCK_WORDS      = TCU_TC_K * TCU_TC_N;
    localparam B_BUF_WORDS        = NUM_BANKS;             // storage per slot
                                                           // (= 1 logical 32-bit bank-row)
    localparam LG_B_SUB_BLOCKS    = $clog2(TCU_WG_B_SUB_BLOCKS);
    // XLEN ratio: each physical LMEM bank-row carries (XLEN/32) logical
    // 32-bit bank-rows side-by-side.
    localparam XLEN_RATIO         = `VX_CFG_XLEN / 32;
    localparam LG_XLEN_RATIO      = (XLEN_RATIO > 1) ? $clog2(XLEN_RATIO) : 0;

    // Canonical-config invariant: 1 logical (32-bit-equivalent) bank-row
    // holds B_SUB_BLOCKS blocks (the smem layout is XLEN-independent).
    `STATIC_ASSERT (B_BLOCK_WORDS * TCU_WG_B_SUB_BLOCKS == NUM_BANKS,
                    ("VX_tcu_bbuf assumes one bank-row per B_SUB_BLOCKS blocks"))

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
    // K_STEPS=2 always → only req_step_k[0] selects the block; upper bits unused.
    `UNUSED_VAR (req_step_k[3:1])

    // Dense LMEM bank-row offset: one logical 32-bit bank-row holds
    // B_SUB_BLOCKS dense blocks; XLEN_RATIO of them fit in one physical
    // bank-row, so the LMEM addr advances every (B_SUB_BLOCKS * XLEN_RATIO)
    // blocks.
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

    // -----------------------------------------------------------------------
    // Descriptor base address
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
    //   slot A: dense bank-row, or sparse flat bank-row 0
    //   slot B: sparse flat bank-row 1 (sparse only)
    // -----------------------------------------------------------------------

    logic                       slot_a_valid_r;
    logic [BANK_ADDR_WIDTH-1:0] slot_a_addr_r;
    logic [BANK_ADDR_WIDTH-1:0] slot_desc_b_row_base_r;
    logic                       slot_fetching_r;
    logic [SUB_HALF_W-1:0]      slot_a_sub_half_r;   // dense XLEN>32 half select
    logic                       slot_b_valid_r;
    logic [BANK_ADDR_WIDTH-1:0] slot_b_addr_r;
    logic                       slot_is_sparse_r;

    // req_desc_b is only valid on the first uop of a WGMMA expansion; latch
    // desc_b on first uop and use the latched base for subsequent uops.
    `UNUSED_VAR (req_step_m)
    wire is_first_uop = req_is_first_uop;
    wire [BANK_ADDR_WIDTH-1:0] effective_desc_b_row_base =
        is_first_uop ? desc_b_row_base : slot_desc_b_row_base_r;

    // -----------------------------------------------------------------------
    // Fetch addresses
    // -----------------------------------------------------------------------

    wire [BANK_ADDR_WIDTH-1:0] fetch_addr_dense =
        effective_desc_b_row_base + BANK_ADDR_WIDTH'(dense_offset);

    // Flat sparse-B: each (step_k, step_n) block is TCU_WG_B_BLOCK_SIZE_SP
    // words in FEDP candidate-pair order, occupying FLAT_BANK_ROWS_PER_BLK
    // contiguous physical bank-rows. The two bbuf slots hold those rows
    // verbatim.
    // A sparse block (2*B_BLOCK_WORDS words) spans one physical bank-row when it
    // fits (NT8/NT32, FLAT_BANK_ROWS_PER_BLK==1) or two when it doesn't (NT16,
    // ==2). One-row uses slot A only; two-row uses slots A and B.
    localparam FLAT_BANK_ROWS_PER_BLK = TCU_WG_B_BLOCK_SIZE_SP / (NUM_BANKS * XLEN_RATIO);
    `STATIC_ASSERT (FLAT_BANK_ROWS_PER_BLK == 1 || FLAT_BANK_ROWS_PER_BLK == 2,
                    ("flat sparse-B supports 1 or 2 bank-rows per block"))
    localparam bit SPARSE_TWO_ROW = (FLAT_BANK_ROWS_PER_BLK == 2);
    // The flat block index (step_k*n_steps + step_n) is exactly block_index.
    wire [BANK_ADDR_WIDTH-1:0] fetch_addr_a_flat =
        effective_desc_b_row_base
      + BANK_ADDR_WIDTH'(block_index) * BANK_ADDR_WIDTH'(FLAT_BANK_ROWS_PER_BLK);
    wire [BANK_ADDR_WIDTH-1:0] fetch_addr_b_flat = fetch_addr_a_flat + BANK_ADDR_WIDTH'(1);

    wire [BANK_ADDR_WIDTH-1:0] fetch_addr_a =
        req_is_sparse ? fetch_addr_a_flat : fetch_addr_dense;

    wire bank_row_resident_dense =
        slot_a_valid_r && !slot_is_sparse_r
        && (slot_a_addr_r == fetch_addr_dense)
        && (slot_a_sub_half_r == dense_sub_half);

    wire bank_row_resident_sparse =
        slot_a_valid_r && slot_is_sparse_r
        && (slot_a_addr_r == fetch_addr_a_flat)
        && (!SPARSE_TWO_ROW || (slot_b_valid_r && (slot_b_addr_r == fetch_addr_b_flat)));

    wire bank_row_resident = req_is_sparse ? bank_row_resident_sparse
                                           : bank_row_resident_dense;
    wire need_fetch        = req_valid && !bank_row_resident;
    wire alloc_en          = need_fetch && !slot_fetching_r;

    assign bbuf_ready = !req_valid || bank_row_resident;

    // -----------------------------------------------------------------------
    // Fetch FSM
    //   S_IDLE → S_FETCH_A → S_IDLE                 (dense)
    //   S_IDLE → S_FETCH_A → S_FETCH_B → S_IDLE     (sparse)
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

    // Single outstanding request per fetch state (req_inflight_r gates re-fire).
    wire can_issue = in_fetch && !req_inflight_r;
    wire last_rsp  = in_fetch && tcu_lmem_if.rsp_valid;

    wire [BANK_ADDR_WIDTH-1:0] active_lmem_addr =
        in_fetch_b ? slot_b_addr_r : slot_a_addr_r;

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
            slot_is_sparse_r       <= 1'b0;
        end else begin
            if (tcu_lmem_if.rsp_valid)
                req_inflight_r <= 1'b0;
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                req_inflight_r <= 1'b1;

            // Latch desc_b base on first uop so non-first uops re-derive
            // fetch_addr without the gated req_desc_b bus.
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
                        // slot_b is read only in sparse mode; dense don't-care.
                        slot_b_addr_r       <= fetch_addr_b_flat;
                        slot_is_sparse_r    <= req_is_sparse;
                        req_inflight_r      <= 1'b0;
                    end
                end
                S_FETCH_A: begin
                    if (last_rsp) begin
                        slot_a_valid_r <= 1'b1;
                        req_inflight_r <= 1'b0;
                        // Two-row sparse needs a second fetch into slot B;
                        // one-row sparse and dense are complete after slot A.
                        if (slot_is_sparse_r && SPARSE_TWO_ROW) begin
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
    // Storage (LUTRAM): two slots × NUM_BANKS 32-bit words, written verbatim.
    //   Dense:  slot A holds one logical 32-bit bank-row (picked by sub_half
    //           at XLEN>32, the lower NUM_BANKS otherwise).
    //   Sparse: slot A holds flat bank-row 0, slot B holds flat bank-row 1.
    // tcu_core's b_off (dense) and the flat read permutation (sparse) select
    // the operands at execute time.
    // -----------------------------------------------------------------------

    logic [B_BUF_WORDS*32-1:0] storage_a_wdata, storage_b_wdata;
    logic [B_BUF_WORDS-1:0]    storage_a_wren,  storage_b_wren;

    // 32-bit-word offset of the source logical bank-row within the physical
    // LMEM response (NUM_BANKS * XLEN_RATIO words). Dense picks a sub-half;
    // sparse always starts at 0.
    localparam OFF_W = $clog2(NUM_BANKS * XLEN_RATIO) + 1;
    wire [OFF_W-1:0] a_off_words = slot_is_sparse_r
                                 ? '0
                                 : (OFF_W'(slot_a_sub_half_r) * OFF_W'(NUM_BANKS));

    always_comb begin
        storage_a_wdata = '0;
        storage_a_wren  = '0;
        storage_b_wdata = '0;
        storage_b_wren  = '0;
        if (tcu_lmem_if.rsp_valid) begin
            for (int b = 0; b < B_BUF_WORDS; ++b) begin
                if (in_fetch_a) begin
                    storage_a_wren[b]             = 1'b1;
                    storage_a_wdata[b * 32 +: 32] =
                        tcu_lmem_if.rsp_data.data[(int'(a_off_words) + b) * 32 +: 32];
                end
                if (in_fetch_b) begin
                    storage_b_wren[b]             = 1'b1;
                    storage_b_wdata[b * 32 +: 32] =
                        tcu_lmem_if.rsp_data.data[b * 32 +: 32];
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
    // Output mux (constant wiring).
    //   Dense:  rs2[0..NUM_BANKS-1] = storage_A. tcu_core's b_off picks within.
    //   Sparse: flat read permutation — storage holds the block N-inner
    //           (word = kw_in*tcN + n_in); the FEDP wants
    //           rs2[k_idx*tcN*2 + n_in*2 + cand] with kw_in = k_idx*2 + cand.
    //           slot_a||slot_b split at B_BLOCK_WORDS.
    // -----------------------------------------------------------------------

    logic [TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0] rs2_mux;
    always_comb begin
        rs2_mux = '0;
        for (int lane = 0; lane < TCU_WG_RS2_WIDTH; ++lane) begin
            if (slot_is_sparse_r) begin
                automatic int unsigned k_idx_l = lane / (TCU_TC_N * 2);
                automatic int unsigned rem_l   = lane % (TCU_TC_N * 2);
                automatic int unsigned n_in_l  = rem_l / 2;
                automatic int unsigned cand_l  = rem_l % 2;
                automatic int unsigned w_l     = (k_idx_l * 2 + cand_l) * TCU_TC_N + n_in_l;
                // Slot A holds the first B_BUF_WORDS words; the remainder (two-row
                // only) lives in slot B. Index masked to stay in range when the
                // block fits one row (slot B then unused and synth-pruned).
                if (w_l < int'(B_BUF_WORDS)) begin
                    rs2_mux[lane] = `VX_CFG_XLEN'(storage_a_rdata[w_l]);
                end else if (w_l < int'(2 * B_BLOCK_WORDS)) begin
                    rs2_mux[lane] = `VX_CFG_XLEN'(storage_b_rdata[(w_l - int'(B_BUF_WORDS)) & (B_BUF_WORDS-1)]);
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
                    fetch_addr_a, fetch_addr_b_flat, dense_sub_half))
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
