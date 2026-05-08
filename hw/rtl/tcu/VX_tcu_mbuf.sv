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
`ifdef TCU_SPARSE_ENABLE

//
// Per-block sparse-metadata buffer (SS sparse, block-major SMEM).
//
// Mirrors VX_tcu_abuf shape: per-warp, fetched alongside A's k-stripe.
// Storage holds META_TOTAL_MAX 32-bit words = WG_META_BANKS × META_STRIDE_<fmt>.
// Refill keyed on {desc_a, step_k} — the meta region lives at
//   meta_base = desc_a_row_base + A_BANK_ROWS_SP    (skips compressed A)
// in the per-warp SMEM section. Format-dependent stride is selected from
// req_fmt_s; storage is sized for the worst case (fp8 / int8, i_ratio=4).
//
// Output: tbuf_sp_meta = extracted meta block for the current uop's step_m,
// indexed as wg_bank = step_m * WG_HALF_K + step_k_half. Zero-extended to
// TCU_MAX_META_BLOCK_WIDTH so VX_tcu_sp_mux's vld_block input shape is
// preserved.
//

module VX_tcu_mbuf import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID     = "",
    parameter         NUM_BANKS       = 4,
    parameter         BANK_ADDR_WIDTH = 12
) (
    input  wire clk,
    input  wire reset,

`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] mbuf_stalls,
    output wire [PERF_CTR_BITS-1:0] lmem_reads,
`endif

    // Execute-side observation (req_valid pre-gated to WGMMA at wrapper)
    input  wire                     req_valid,
    input  wire                     req_is_sparse,
    input  wire                     req_a_is_smem,
    input  wire [3:0]               req_step_m,
    input  wire [3:0]               req_step_n,
    input  wire [3:0]               req_step_k,
    input  wire [3:0]               req_fmt_s,
    input  wire [`XLEN-1:0]         req_desc_a,

    // LMEM bank-parallel read port
    VX_mem_bus_if.master            tcu_lmem_if,

    // Outputs
    output wire                                       mbuf_ready,
    output wire [TCU_MAX_META_BLOCK_WIDTH-1:0]        mbuf_sp_meta
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // -----------------------------------------------------------------------
    // Constants (from the pre-refactor tbuf_fetch sparse path; sized for fp8 worst case)
    // -----------------------------------------------------------------------

    localparam BANK_SEL_BITS      = $clog2(NUM_BANKS);
    localparam WORD_SIZE_LOG2     = $clog2(`XLEN / 8);

    localparam SP_I_RATIO_32B     = 1;
    localparam SP_I_RATIO_16B     = 2;
    localparam SP_I_RATIO_8B      = 4;
    localparam META_ROW_BITS_32B  = TCU_TC_K * 2 * SP_I_RATIO_32B;
    localparam META_ROW_BITS_16B  = TCU_TC_K * 2 * SP_I_RATIO_16B;
    localparam META_ROW_BITS_8B   = TCU_TC_K * 2 * SP_I_RATIO_8B;
    localparam META_STRIDE_32B    = (TCU_TC_M * META_ROW_BITS_32B + 31) / 32;
    localparam META_STRIDE_16B    = (TCU_TC_M * META_ROW_BITS_16B + 31) / 32;
    localparam META_STRIDE_8B     = (TCU_TC_M * META_ROW_BITS_8B  + 31) / 32;
    localparam WG_HALF_K          = TCU_WG_K_STEPS / 2;
    localparam WG_META_BANKS      = TCU_WG_M_STEPS * WG_HALF_K;
    localparam META_STRIDE_MAX    = META_STRIDE_8B;
    localparam META_TOTAL_MAX     = WG_META_BANKS * META_STRIDE_MAX;
    localparam META_BANK_ROWS_MAX = (META_TOTAL_MAX + NUM_BANKS - 1) / NUM_BANKS;
    localparam FETCH_CTR_W        = `CLOG2(META_BANK_ROWS_MAX + 1);

    // A-side sparse total bank rows (used to compute meta base offset:
    // meta sits immediately after the compressed A region).
    localparam A_TOTAL            = TCU_WG_TILE_M * TCU_WG_TILE_K;
    localparam A_TOTAL_SP         = A_TOTAL / 2;
    localparam A_BANK_ROWS_SP     = (A_TOTAL_SP + NUM_BANKS - 1) / NUM_BANKS;

    localparam K_STEPS_W          = `CLOG2(TCU_WG_K_STEPS);

    // -----------------------------------------------------------------------
    // Format-dependent stride / fetch length
    // -----------------------------------------------------------------------

    logic [3:0]             init_meta_stride;
    logic [FETCH_CTR_W-1:0] init_meta_rows;
    always_comb begin
        case (tcu_fmt_width(req_fmt_s))
            32: begin
                init_meta_stride = 4'(META_STRIDE_32B);
                init_meta_rows   = FETCH_CTR_W'((WG_META_BANKS * META_STRIDE_32B + NUM_BANKS - 1) / NUM_BANKS);
            end
            16: begin
                init_meta_stride = 4'(META_STRIDE_16B);
                init_meta_rows   = FETCH_CTR_W'((WG_META_BANKS * META_STRIDE_16B + NUM_BANKS - 1) / NUM_BANKS);
            end
            default: begin
                init_meta_stride = 4'(META_STRIDE_8B);
                init_meta_rows   = FETCH_CTR_W'(META_BANK_ROWS_MAX);
            end
        endcase
    end

    // -----------------------------------------------------------------------
    // Resident slot
    // -----------------------------------------------------------------------

    logic                       slot_valid_r;
    logic [`XLEN-1:0]           slot_desc_a_r;
    logic [BANK_ADDR_WIDTH-1:0] slot_meta_base_r;
    logic [`UP(K_STEPS_W)-1:0]  slot_step_k_r;
    logic [3:0]                 slot_meta_stride_r;
    logic [FETCH_CTR_W-1:0]     slot_meta_rows_r;
    logic                       slot_fetching_r;

    wire [`UP(K_STEPS_W)-1:0]   req_step_k_trunc = `UP(K_STEPS_W)'(req_step_k);
    if (4 > K_STEPS_W) begin : g_step_k_upper_unused
        `UNUSED_VAR (req_step_k[3:`UP(K_STEPS_W)])
    end

    // The uop expander only reads rs1 (desc_a) on uop 0 of a WGMMA expansion
    // when a_from_smem (VX_tcu_uops.sv:362). Gate the desc_a comparison on
    // is_first_uop so non-first uops use the latched desc_a.
    wire is_first_uop = (req_step_m == '0) && (req_step_n == '0) && (req_step_k == '0);
    `UNUSED_VAR (req_step_n)

    wire desc_a_match = !is_first_uop || (slot_desc_a_r == req_desc_a);

    // Only fetch meta for SS sparse uops. Dense or RS-sparse paths bypass mbuf.
    wire active = req_valid && req_is_sparse && req_a_is_smem;
    wire stripe_resident = slot_valid_r
                        && desc_a_match
                        && (slot_step_k_r == req_step_k_trunc);
    wire need_fetch = active && !stripe_resident;
    wire alloc_en   = need_fetch && !slot_fetching_r;

    assign mbuf_ready = !active || stripe_resident;

    // -----------------------------------------------------------------------
    // Address compute: meta starts after the compressed A region.
    // -----------------------------------------------------------------------

    localparam DESC_ADDR_W = BANK_ADDR_WIDTH + BANK_SEL_BITS;
    wire [DESC_ADDR_W-1:0]      desc_a_word_base = DESC_ADDR_W'(req_desc_a[15:0] >> WORD_SIZE_LOG2);
    wire [BANK_ADDR_WIDTH-1:0]  desc_a_row_base  = desc_a_word_base[BANK_SEL_BITS +: BANK_ADDR_WIDTH];
    if (BANK_SEL_BITS > 0) begin : g_addr_lsb_unused
        `UNUSED_VAR (desc_a_word_base[BANK_SEL_BITS-1:0])
    end

    // On non-first uops req_desc_a is gated-off (garbage), so a k-stripe
    // refill mid-WGMMA must use the latched meta_base. Compute meta_base
    // from req_desc_a only when is_first_uop (kept for the first-uop alloc
    // path and the resident-key recompute); otherwise use the latched value.
    wire [BANK_ADDR_WIDTH-1:0]  meta_base_first = desc_a_row_base + BANK_ADDR_WIDTH'(A_BANK_ROWS_SP);
    wire [BANK_ADDR_WIDTH-1:0]  meta_base = is_first_uop ? meta_base_first : slot_meta_base_r;

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

    wire all_requested = (req_ctr_r >= slot_meta_rows_r);
    wire can_issue = in_fetch && !all_requested
                  && (!req_inflight_r || tcu_lmem_if.rsp_valid);
    wire last_rsp = in_fetch && tcu_lmem_if.rsp_valid
                 && (rsp_ctr_r == slot_meta_rows_r - FETCH_CTR_W'(1));

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
            fsm_state_r        <= S_IDLE;
            req_ctr_r          <= '0;
            rsp_ctr_r          <= '0;
            req_inflight_r     <= 1'b0;
            slot_valid_r       <= 1'b0;
            slot_fetching_r    <= 1'b0;
            slot_desc_a_r      <= '0;
            slot_meta_base_r   <= '0;
            slot_step_k_r      <= '0;
            slot_meta_stride_r <= '0;
            slot_meta_rows_r   <= '0;
            fetch_base_r       <= '0;
        end else begin
            if (tcu_lmem_if.rsp_valid)
                req_inflight_r <= 1'b0;
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                req_inflight_r <= 1'b1;

            // Latch desc_a-derived meta_base on first uop (req_desc_a is
            // valid only there).
            if (active && is_first_uop) begin
                slot_desc_a_r    <= req_desc_a;
                slot_meta_base_r <= meta_base_first;
            end

            case (fsm_state_r)
                S_IDLE: begin
                    if (alloc_en) begin
                        fsm_state_r        <= S_FETCH;
                        slot_fetching_r    <= 1'b1;
                        slot_valid_r       <= 1'b0;
                        slot_step_k_r      <= req_step_k_trunc;
                        slot_meta_stride_r <= init_meta_stride;
                        slot_meta_rows_r   <= init_meta_rows;
                        fetch_base_r       <= meta_base;
                        req_ctr_r          <= '0;
                        rsp_ctr_r          <= '0;
                        req_inflight_r     <= 1'b0;
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
    // Storage (LUTRAM): META_TOTAL_MAX 32-bit words.
    // -----------------------------------------------------------------------

    logic [META_TOTAL_MAX*32-1:0] storage_wdata;
    logic [META_TOTAL_MAX-1:0]    storage_wren;

    always_comb begin
        storage_wdata = '0;
        storage_wren  = '0;
        if (in_fetch && tcu_lmem_if.rsp_valid) begin
            for (int b = 0; b < NUM_BANKS; ++b) begin
                if (int'(rsp_ctr_r) * NUM_BANKS + b < META_TOTAL_MAX) begin
                    storage_wren[int'(rsp_ctr_r) * NUM_BANKS + b] = 1'b1;
                    storage_wdata[(int'(rsp_ctr_r) * NUM_BANKS + b) * 32 +: 32] =
                        tcu_lmem_if.rsp_data.data[b * `XLEN +: `XLEN];
                end
            end
        end
    end

    wire [META_TOTAL_MAX-1:0][31:0] storage_rdata;

    VX_dp_ram #(
        .DATAW   (META_TOTAL_MAX * 32),
        .SIZE    (1),
        .WRENW   (META_TOTAL_MAX),
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
    // Output: extract meta block for current uop's (step_m, step_k_half).
    //   wg_bank = step_m * WG_HALF_K + step_k_half_unit
    //   block_off = wg_bank * META_STRIDE_<fmt>  (in 32-bit words)
    // Zero-extend to TCU_MAX_META_BLOCK_WIDTH (sized for int4 i_ratio=8).
    // -----------------------------------------------------------------------

    localparam LG_WG_HALF_K = `CLOG2(WG_HALF_K);
    wire [LG_WG_HALF_K-1:0] step_k_half = (WG_HALF_K == 1) ? '0 : req_step_k[LG_WG_HALF_K-1:0];
    if (WG_HALF_K == 1) begin : g_step_k_half_dontuse
        `UNUSED_VAR (req_step_k)
    end

    wire [4:0] wg_bank = 5'(req_step_m) * 5'(WG_HALF_K) + 5'(step_k_half);
    wire [9:0] block_off_words = 10'(wg_bank) * 10'(slot_meta_stride_r);

    logic [META_STRIDE_MAX*32-1:0] extracted_meta;
    always_comb begin
        extracted_meta = '0;
        for (int w = 0; w < META_STRIDE_MAX; ++w) begin
            int idx;
            idx = int'(block_off_words) + w;
            if (w < int'(slot_meta_stride_r) && idx < int'(META_TOTAL_MAX))
                extracted_meta[w*32 +: 32] = storage_rdata[idx];
        end
    end

    assign mbuf_sp_meta = TCU_MAX_META_BLOCK_WIDTH'(extracted_meta);

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
            if (active && !mbuf_ready)
                stall_ctr_r <= stall_ctr_r + PERF_CTR_BITS'(1);
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                reads_ctr_r <= reads_ctr_r + PERF_CTR_BITS'(1);
        end
    end
    assign mbuf_stalls = stall_ctr_r;
    assign lmem_reads  = reads_ctr_r;
`endif

    // -----------------------------------------------------------------------
    // Debug trace
    // -----------------------------------------------------------------------

`ifdef DBG_TRACE_TCU
    always @(posedge clk) begin
        if (!reset) begin
            if (alloc_en)
                `TRACE(3, ("%t: %s mbuf: alloc desc_a=0x%0h step_k=%0d base=0x%0h rows=%0d\n",
                    $time, INSTANCE_ID, req_desc_a, req_step_k, meta_base, init_meta_rows))
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                `TRACE(3, ("%t: %s mbuf: rd_req addr=0x%0h\n",
                    $time, INSTANCE_ID, tcu_lmem_if.req_data.addr))
            if (last_rsp)
                `TRACE(3, ("%t: %s mbuf: meta READY\n", $time, INSTANCE_ID))
        end
    end
`endif

endmodule

`endif // TCU_SPARSE_ENABLE
`endif // TCU_WGMMA_ENABLE
