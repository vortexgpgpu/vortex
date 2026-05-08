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
// Per-block A buffer (block-major SMEM, k-stripe storage).
//
// Holds the M_STEPS A-blocks for the current k-stripe of one warp slot.
// Storage is M_STEPS × BLOCK_BANK_ROWS bank-rows = M_STEPS A-blocks.
//
// Refill key = {desc_a, step_k}. On change, fetch M_STEPS adjacent
// bank-rows starting at base_a + step_k * M_STEPS * BLOCK_BANK_ROWS.
//
// Output rs1_data carries one A-block per cycle, selected by step_m.
// Format-aware sub-word extraction (fp16/fp8 packing) happens
// downstream in the FEDP grid; abuf is word pass-through.
//
// SMEM layout assumed (per docs/proposals/wgmma_simx_v3_addendum.md):
//   A_smem[(k_blk * M_STEPS + m_blk) * BLOCK_WORDS + i*TC_K + k_in]
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
    input  wire [3:0]               req_step_m,
    input  wire [3:0]               req_step_n,
    input  wire [3:0]               req_step_k,
    input  wire [`XLEN-1:0]         req_desc_a,
    input  wire                     req_a_is_smem,

    // LMEM bank-parallel read port
    VX_mem_bus_if.master            tcu_lmem_if,

    // Outputs
    output wire                                   abuf_ready,
    output wire [TCU_BLOCK_CAP-1:0][`XLEN-1:0]    abuf_rs1_data
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (req_wid)

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    localparam BANK_SEL_BITS      = $clog2(NUM_BANKS);
    localparam WORD_SIZE_LOG2     = $clog2(`XLEN / 8);
    localparam A_BLOCK_WORDS      = TCU_TC_M * TCU_TC_K;
    localparam A_BLOCK_BANK_ROWS  = (A_BLOCK_WORDS + NUM_BANKS - 1) / NUM_BANKS;
    localparam BLOCK_WORDS_PADDED = A_BLOCK_BANK_ROWS * NUM_BANKS;
    localparam A_STRIPE_BANK_ROWS = TCU_WG_M_STEPS * A_BLOCK_BANK_ROWS;
    localparam A_STRIPE_WORDS     = A_STRIPE_BANK_ROWS * NUM_BANKS;
    localparam FETCH_CTR_W        = `CLOG2(A_STRIPE_BANK_ROWS + 1);
    localparam K_STEPS_W          = `CLOG2(TCU_WG_K_STEPS);

    // Canonical-config invariant: one A-block fits one bank-row
    // (TC_M*TC_K == NUM_BANKS). Non-canonical configs need A_SUB_BLOCKS
    // packing in the output mux; not supported in this revision.
    `STATIC_ASSERT (A_BLOCK_BANK_ROWS == 1, ("VX_tcu_abuf assumes one A-block per bank-row"))

    // -----------------------------------------------------------------------
    // Resident slot state
    // -----------------------------------------------------------------------

    logic                       slot_valid_r;
    logic [`XLEN-1:0]           slot_desc_a_r;
    logic [BANK_ADDR_WIDTH-1:0] slot_desc_a_row_base_r;
    logic [`UP(K_STEPS_W)-1:0]  slot_step_k_r;
    logic                       slot_fetching_r;

    wire [`UP(K_STEPS_W)-1:0]   req_step_k_trunc = `UP(K_STEPS_W)'(req_step_k);
    if (4 > K_STEPS_W) begin : g_step_k_upper_unused
        `UNUSED_VAR (req_step_k[3:`UP(K_STEPS_W)])
    end

    // The uop expander only reads rs1 (desc_a) on uop 0 of a WGMMA expansion
    // (VX_tcu_uops.sv:362, used_rs[0] = (wg_idx_ctr == '0) when a_from_smem).
    // On non-first uops, req_desc_a is garbage. Gate the desc_a comparison on
    // is_first_uop, and treat non-first uops as referring to the latched desc.
    wire is_first_uop = (req_step_m == '0) && (req_step_n == '0) && (req_step_k == '0);
    `UNUSED_VAR (req_step_n)  // only used in is_first_uop computation

    wire desc_a_match = !is_first_uop || (slot_desc_a_r == req_desc_a);
    wire stripe_resident = slot_valid_r
                        && desc_a_match
                        && (slot_step_k_r == req_step_k_trunc);

    // RS mode (a_from_smem=0): A from registers, abuf bypassed → always ready.
    wire need_smem  = req_valid && req_a_is_smem;
    wire need_fetch = need_smem && !stripe_resident;
    wire alloc_en   = need_fetch && !slot_fetching_r;

    assign abuf_ready = !req_valid || !req_a_is_smem || stripe_resident;

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

    localparam STRIPE_STRIDE_BANK_ROWS = TCU_WG_M_STEPS * A_BLOCK_BANK_ROWS;

    // Use latched desc_a base on non-first uops (req_desc_a is garbage there
    // because the uop expander gates the rs1 register read on uop 0 only —
    // VX_tcu_uops.sv:362). Without this, k-stripe-transition refills mid-WGMMA
    // would compute the wrong fetch_base.
    wire [BANK_ADDR_WIDTH-1:0]  effective_desc_a_row_base =
        is_first_uop ? desc_a_row_base : slot_desc_a_row_base_r;

    wire [BANK_ADDR_WIDTH-1:0]  stripe_base =
        effective_desc_a_row_base
      + BANK_ADDR_WIDTH'(req_step_k_trunc) * BANK_ADDR_WIDTH'(STRIPE_STRIDE_BANK_ROWS);
    // M_STEPS, A_BLOCK_BANK_ROWS are compile-time; STRIPE_STRIDE_BANK_ROWS
    // is a power of 2 in canonical configs, so synth folds the multiply
    // into a shift.

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

    wire all_requested = (req_ctr_r >= FETCH_CTR_W'(A_STRIPE_BANK_ROWS));
    wire can_issue     = in_fetch && !all_requested
                      && (!req_inflight_r || tcu_lmem_if.rsp_valid);
    wire last_rsp      = in_fetch && tcu_lmem_if.rsp_valid
                      && (rsp_ctr_r == FETCH_CTR_W'(A_STRIPE_BANK_ROWS - 1));

    // LMEM master driver
    assign tcu_lmem_if.req_valid       = can_issue;
    assign tcu_lmem_if.req_data.rw     = 1'b0;
    assign tcu_lmem_if.req_data.addr   = fetch_base_r + BANK_ADDR_WIDTH'(req_ctr_r);
    assign tcu_lmem_if.req_data.data   = '0;
    assign tcu_lmem_if.req_data.byteen = '0;
    assign tcu_lmem_if.req_data.attr   = '0;
    assign tcu_lmem_if.req_data.tag    = '0;
    assign tcu_lmem_if.rsp_ready       = 1'b1;
    `UNUSED_VAR (tcu_lmem_if.rsp_data.tag)

    always_ff @(posedge clk) begin
        if (reset) begin
            fsm_state_r            <= S_IDLE;
            req_ctr_r              <= '0;
            rsp_ctr_r              <= '0;
            req_inflight_r         <= 1'b0;
            slot_valid_r           <= 1'b0;
            slot_fetching_r        <= 1'b0;
            slot_desc_a_r          <= '0;
            slot_desc_a_row_base_r <= '0;
            slot_step_k_r          <= '0;
            fetch_base_r           <= '0;
        end else begin
            // Inflight tracker (single outstanding request at a time)
            if (tcu_lmem_if.rsp_valid)
                req_inflight_r <= 1'b0;
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                req_inflight_r <= 1'b1;

            // Latch desc_a on first uop of every WGMMA so non-first uops
            // (k-stripe-transition refills) can reuse the base without
            // needing the gated req_desc_a bus.
            if (req_valid && is_first_uop) begin
                slot_desc_a_r          <= req_desc_a;
                slot_desc_a_row_base_r <= desc_a_row_base;
            end

            case (fsm_state_r)
                S_IDLE: begin
                    if (alloc_en) begin
                        fsm_state_r     <= S_FETCH;
                        slot_fetching_r <= 1'b1;
                        slot_valid_r    <= 1'b0;
                        slot_step_k_r   <= req_step_k_trunc;
                        fetch_base_r    <= stripe_base;
                        req_ctr_r       <= '0;
                        rsp_ctr_r       <= '0;
                        req_inflight_r  <= 1'b0;
                    end
                end
                S_FETCH: begin
                    if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                        req_ctr_r <= req_ctr_r + FETCH_CTR_W'(1);
                    if (last_rsp) begin
                        fsm_state_r     <= S_IDLE;
                        slot_fetching_r <= 1'b0;
                        slot_valid_r    <= 1'b1;
                        req_inflight_r  <= 1'b0;
                    end else if (tcu_lmem_if.rsp_valid) begin
                        rsp_ctr_r <= rsp_ctr_r + FETCH_CTR_W'(1);
                    end
                end
                default: fsm_state_r <= S_IDLE;
            endcase
        end
    end

    // -----------------------------------------------------------------------
    // Storage (LUTRAM): A_STRIPE_BANK_ROWS × NUM_BANKS 32-bit words.
    // Each LMEM response writes one bank-row at offset rsp_ctr_r.
    // -----------------------------------------------------------------------

    logic [A_STRIPE_WORDS*32-1:0] storage_wdata;
    logic [A_STRIPE_WORDS-1:0]    storage_wren;

    always_comb begin
        storage_wdata = '0;
        storage_wren  = '0;
        if (in_fetch && tcu_lmem_if.rsp_valid) begin
            for (int b = 0; b < NUM_BANKS; ++b) begin
                if (int'(rsp_ctr_r) * NUM_BANKS + b < A_STRIPE_WORDS) begin
                    storage_wren[int'(rsp_ctr_r) * NUM_BANKS + b] = 1'b1;
                    storage_wdata[(int'(rsp_ctr_r) * NUM_BANKS + b) * 32 +: 32] =
                        tcu_lmem_if.rsp_data.data[b * `XLEN +: `XLEN];
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
        .write (in_fetch && tcu_lmem_if.rsp_valid),
        .wren  (storage_wren),
        .waddr (1'b0),
        .wdata (storage_wdata),
        .raddr (1'b0),
        .rdata (storage_rdata)
    );

    // -----------------------------------------------------------------------
    // Output: select A-block based on step_m, pass through as 32-bit words.
    // Storage holds blocks at [m_blk * BLOCK_WORDS_PADDED .. +BLOCK_WORDS_PADDED).
    // -----------------------------------------------------------------------

    logic [TCU_BLOCK_CAP-1:0][`XLEN-1:0] rs1_mux;
    always_comb begin
        rs1_mux = '0;
        for (int lane = 0; lane < TCU_BLOCK_CAP; ++lane) begin
            if (lane < int'(A_BLOCK_WORDS)) begin
                int src_idx;
                src_idx = int'(req_step_m) * BLOCK_WORDS_PADDED + lane;
                if (src_idx < int'(A_STRIPE_WORDS))
                    rs1_mux[lane] = `XLEN'(storage_rdata[src_idx]);
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
            if (last_rsp)
                `TRACE(3, ("%t: %s abuf: stripe READY\n", $time, INSTANCE_ID))
        end
    end
`endif

endmodule

`endif // TCU_WGMMA_ENABLE
