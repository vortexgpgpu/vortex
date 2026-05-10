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
// Tile-buffer fetch engine (slot management + FSM + LMEM capture).
//
// Manages a direct-mapped per-warp slot table (slot s serves warp s).
// Runs a three-phase LMEM fetch FSM: FETCH_A → FETCH_B → [FETCH_META].
// Exposes the contents of the hit slot as combinational read ports for
// VX_tcu_tbuf_gather.
//
// Storage optimizations vs. original monolithic module:
//   - slot_a_buf replaced by VX_dp_ram(LUTRAM=1, OUT_REG=0): one RAM row per
//     slot, width = A_TOTAL*32 bits, WRENW = A_TOTAL for per-word writes.
//     Frees A_TOTAL*TCU_TBUF_SIZE*32 FFs at no net LUT cost (the LUTRAM
//     cells replace the previously required slot-index MUX LUTs).
//   - shared_b_buf (non-sparse): FFs with no slot dimension.  All warps in a
//     multi-warp CTA share the same physical smem, so one B copy suffices.
//   - slot_b_ram (sparse): per-slot VX_dp_ram(LUTRAM=1); single-warp CTAs
//     each have their own smem/B tile so B must not be shared across slots.
//   - slot_meta_buf (sparse) replaced by VX_dp_ram(LUTRAM=1) same way.
//   - slot_wid removed: slot index == warp id by construction (direct-mapped).
//   - slot_in_progress simplified: warp stalls at execute while fetch runs;
//     stored descriptors always equal the current req, so XLEN-wide descriptor
//     comparisons are eliminated from the alloc_en critical path (~0.95 ns).
//   - One-hot FSM (fsm_encoding attribute): single-FF state decode.
//   - phase_total_rows / phase_row_base expressed as wires for non-sparse path.
//

module VX_tcu_tbuf_fetch import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID  = "",
    parameter         TCU_TBUF_SIZE   = `NUM_WARPS,
    parameter         NUM_BANKS       = 4,
    parameter         BANK_ADDR_WIDTH = 12,
    // Tile buffer sizes (in 32-bit words); passed from VX_tcu_tbuf.
    parameter         A_TOTAL        = 1,
    parameter         B_TOTAL        = 1
`ifdef TCU_SPARSE_ENABLE
   ,parameter         META_TOTAL_MAX = 1
`endif
) (
    input  wire clk,
    input  wire reset,

`ifdef PERF_ENABLE
    // Cycles where req_valid=1 but tbuf_ready=0 (true stall cycles).
    output wire [PERF_CTR_BITS-1:0] tbuf_fetch_stalls,
    // LMEM read transactions accepted (req_valid && req_ready).
    output wire [PERF_CTR_BITS-1:0] lmem_reads,
`endif

    // Execute-side observation
    input  wire                     req_valid,
    input  wire                     req_fire,   // execute consumed current µop
    input  wire [NW_WIDTH-1:0]      req_wid,
    input  wire                     req_is_sparse,
    input  wire [3:0]               req_step_m,
    input  wire [3:0]               req_step_n,
    input  wire [3:0]               req_step_k,
    input  wire [4:0]               req_fmt_s,
    input  wire [`XLEN-1:0]         req_desc_a,
    input  wire [`XLEN-1:0]         req_desc_b,

    // LMEM bank-parallel read port (1-cycle latency, pipelined)
    VX_tcu_lmem_if.master           tcu_lmem_if,

    // Hit status
    output wire                     tbuf_hit,
    output wire                     tbuf_ready,

    // Hit-slot data (combinational read of slot indexed by req_wid)
    output wire [A_TOTAL-1:0][31:0] hit_a_buf,
    output wire [B_TOTAL-1:0][31:0] hit_b_buf
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

    localparam SLOT_W          = `UP($clog2(TCU_TBUF_SIZE));
    localparam BANK_SEL_BITS   = $clog2(NUM_BANKS);
    localparam LMEM_ADDR_W     = BANK_ADDR_WIDTH + BANK_SEL_BITS;
    // Correct word-address shift for any XLEN (2 for 32-bit, 3 for 64-bit).
    localparam WORD_SIZE_LOG2  = $clog2(`XLEN / 8);

    localparam A_BANK_ROWS    = (A_TOTAL + NUM_BANKS - 1) / NUM_BANKS;
    localparam B_BANK_ROWS    = (B_TOTAL + NUM_BANKS - 1) / NUM_BANKS;

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

    localparam MAX_BANK_ROWS = A_BANK_ROWS > B_BANK_ROWS
                             ? (A_BANK_ROWS > META_ROWS_MAX ? A_BANK_ROWS : META_ROWS_MAX)
                             : (B_BANK_ROWS > META_ROWS_MAX ? B_BANK_ROWS : META_ROWS_MAX);
`else
    localparam MAX_BANK_ROWS = A_BANK_ROWS > B_BANK_ROWS ? A_BANK_ROWS : B_BANK_ROWS;
`endif

    localparam FETCH_CTR_W = `CLOG2(MAX_BANK_ROWS + 1);

`ifndef TCU_SPARSE_ENABLE
    wire is_sparse = 1'b0;
    `UNUSED_VAR (is_sparse)
    `UNUSED_VAR (req_is_sparse)
    `UNUSED_VAR (req_fmt_s)
`else
    wire is_sparse = req_is_sparse;
`endif

    // -----------------------------------------------------------------------
    // Slot table  (direct-mapped: slot s serves warp s)
    // -----------------------------------------------------------------------
    // slot_wid is omitted: slot index IS the warp id by construction.

    logic                       slot_valid      [TCU_TBUF_SIZE];
    logic [`XLEN-1:0]           slot_desc_a     [TCU_TBUF_SIZE];
    logic [`XLEN-1:0]           slot_desc_b     [TCU_TBUF_SIZE];
    logic                       slot_fetch_done [TCU_TBUF_SIZE];
    // Prevents alloc_en from re-firing every cycle after a fetch completes.
    logic                       warp_alloc_pending [TCU_TBUF_SIZE];
    logic [BANK_ADDR_WIDTH-1:0] slot_a_row_base [TCU_TBUF_SIZE];
    logic [BANK_ADDR_WIDTH-1:0] slot_b_row_base [TCU_TBUF_SIZE];

`ifndef TCU_SPARSE_ENABLE
    // Shared B buffer: in the non-sparse (multi-warp CTA) path all warps share
    // the same physical smem, so desc_b is identical across slots. One copy
    // of B suffices. Stored in a single-entry LUTRAM (SIZE=1) to avoid
    // B_TOTAL*32 FFs; async read gives same combinational latency.
    logic [`XLEN-1:0]  shared_b_desc;
    logic              shared_b_valid;
    logic              shared_b_dirty;
    logic [SLOT_W-1:0] last_b_fetch_slot_r;
`endif

`ifdef TCU_SPARSE_ENABLE
    logic                       slot_is_sparse      [TCU_TBUF_SIZE];
    logic [BANK_ADDR_WIDTH-1:0] slot_meta_row_base  [TCU_TBUF_SIZE];
    logic [3:0]                 slot_meta_stride    [TCU_TBUF_SIZE];
    logic [FETCH_CTR_W-1:0]     slot_meta_bank_rows [TCU_TBUF_SIZE];
`endif

    // -----------------------------------------------------------------------
    // Slot lookup (O(1), direct-mapped by wid)
    // -----------------------------------------------------------------------

    wire is_first_uop = (req_step_m == '0) && (req_step_n == '0) && (req_step_k == '0);

    // Unified slot index for both hit and alloc.
    wire [SLOT_W-1:0] slot_idx = SLOT_W'(req_wid);

    // Hit: slot is valid, data ready, and descriptors match on first µop.
    assign tbuf_hit = slot_valid[slot_idx]
                   && slot_fetch_done[slot_idx]
                   && (!is_first_uop
                       || (slot_desc_a[slot_idx] == req_desc_a
                        && slot_desc_b[slot_idx] == req_desc_b));

    // In-progress: fetch for this warp's slot is running.
    // Descriptor comparisons omitted: warp stalls at execute during fetch,
    // so slot descriptors always equal the current req descriptors.
    wire slot_in_progress = slot_valid[slot_idx] && !slot_fetch_done[slot_idx];

    // Allocate (or re-allocate) a slot on the first µop of each new tile.
    wire alloc_en = req_valid && is_first_uop && !slot_in_progress
                 && !warp_alloc_pending[slot_idx];

    assign tbuf_ready = tbuf_hit && !alloc_en;

    // -----------------------------------------------------------------------
    // Descriptor parsing → row bases (combinational, used only at alloc)
    // -----------------------------------------------------------------------

    wire [LMEM_ADDR_W-1:0] desc_a_word_base = LMEM_ADDR_W'(req_desc_a[15:0] >> WORD_SIZE_LOG2);
    wire [LMEM_ADDR_W-1:0] desc_b_word_base = LMEM_ADDR_W'(req_desc_b[15:0] >> WORD_SIZE_LOG2);

    wire [BANK_ADDR_WIDTH-1:0] desc_a_row_base = desc_a_word_base[BANK_SEL_BITS +: BANK_ADDR_WIDTH];
    wire [BANK_ADDR_WIDTH-1:0] desc_b_row_base = desc_b_word_base[BANK_SEL_BITS +: BANK_ADDR_WIDTH];

    if (BANK_SEL_BITS > 0) begin : g_word_base_lsbs_unused
        `UNUSED_VAR (desc_a_word_base[BANK_SEL_BITS-1:0])
        `UNUSED_VAR (desc_b_word_base[BANK_SEL_BITS-1:0])
    end

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

    typedef enum logic [1:0] {
        SEND_IDLE       = 2'd0,
        SEND_FETCH_A    = 2'd1,
        SEND_FETCH_B    = 2'd2,
        SEND_FETCH_META = 2'd3
    } send_state_e;

    (* fsm_encoding = "one_hot" *) send_state_e send_state_r;
    logic [SLOT_W-1:0] send_slot_r;

    wire in_fetch_a    = (send_state_r == SEND_FETCH_A);
    wire in_fetch_b    = (send_state_r == SEND_FETCH_B);
`ifdef TCU_SPARSE_ENABLE
    wire in_fetch_meta = (send_state_r == SEND_FETCH_META);
`endif
    wire in_fetch      = (send_state_r != SEND_IDLE);

    // -----------------------------------------------------------------------
    // Fetch counters (req: issued requests; rsp: received responses)
    // -----------------------------------------------------------------------

    logic [FETCH_CTR_W-1:0] req_ctr_r;
    logic [FETCH_CTR_W-1:0] rsp_ctr_r;
    logic                   req_inflight_r;

    // Per-phase termination threshold and row base as wires (non-sparse).
    // Sparse path retains always_comb for the three-way case.
`ifndef TCU_SPARSE_ENABLE
    wire [FETCH_CTR_W-1:0]     phase_total_rows =
        in_fetch_a ? FETCH_CTR_W'(A_BANK_ROWS) : FETCH_CTR_W'(B_BANK_ROWS);

    wire [BANK_ADDR_WIDTH-1:0] phase_row_base =
        in_fetch_a ? slot_a_row_base[send_slot_r] : slot_b_row_base[send_slot_r];
`else
    logic [FETCH_CTR_W-1:0]     phase_total_rows;
    logic [BANK_ADDR_WIDTH-1:0] phase_row_base;
    always_comb begin
        case (send_state_r)
            SEND_FETCH_A: begin
                phase_total_rows = FETCH_CTR_W'(slot_is_sparse[send_slot_r]
                                   ? A_BANK_ROWS_SP : A_BANK_ROWS);
                phase_row_base   = slot_a_row_base[send_slot_r];
            end
            SEND_FETCH_B: begin
                phase_total_rows = FETCH_CTR_W'(B_BANK_ROWS);
                phase_row_base   = slot_b_row_base[send_slot_r];
            end
            SEND_FETCH_META: begin
                phase_total_rows = slot_meta_bank_rows[send_slot_r];
                phase_row_base   = slot_meta_row_base[send_slot_r];
            end
            default: begin
                phase_total_rows = '0;
                phase_row_base   = '0;
            end
        endcase
    end
`endif

    wire all_requested = (req_ctr_r >= phase_total_rows);

    // -----------------------------------------------------------------------
    // LMEM request handshake
    // -----------------------------------------------------------------------

    wire can_issue = in_fetch && !all_requested
                  && (!req_inflight_r || tcu_lmem_if.rsp_valid);

    assign tcu_lmem_if.req_valid = can_issue;
    assign tcu_lmem_if.req_addr  = phase_row_base + BANK_ADDR_WIDTH'(req_ctr_r);

    // -----------------------------------------------------------------------
    // Phase-done detection
    // -----------------------------------------------------------------------

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
    wire fetch_done_now = (last_rsp_b && !slot_is_sparse[send_slot_r]) || last_rsp_meta;
`else
    wire b_cached       = shared_b_valid && !shared_b_dirty
                       && (slot_desc_b[send_slot_r] == shared_b_desc);
    wire fetch_done_now = last_rsp_b || (last_rsp_a && b_cached);
`endif

    // -----------------------------------------------------------------------
    // Sender FSM
    // -----------------------------------------------------------------------

    always_ff @(posedge clk) begin
        if (reset) begin
            send_state_r   <= SEND_IDLE;
            send_slot_r    <= '0;
            req_ctr_r      <= '0;
            rsp_ctr_r      <= '0;
            req_inflight_r <= 1'b0;
        end else begin
            if (tcu_lmem_if.rsp_valid)
                req_inflight_r <= 1'b0;
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                req_inflight_r <= 1'b1;

            case (send_state_r)
            // -----------------------------------------------------------------
            SEND_IDLE: begin
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
            // -----------------------------------------------------------------
            SEND_FETCH_A: begin
                if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                    req_ctr_r <= req_ctr_r + FETCH_CTR_W'(1);
                if (last_rsp_a) begin
                    req_ctr_r      <= '0;
                    rsp_ctr_r      <= '0;
                    req_inflight_r <= 1'b0;
                `ifdef TCU_SPARSE_ENABLE
                    send_state_r   <= SEND_FETCH_B;
                `else
                    send_state_r   <= b_cached ? SEND_IDLE : SEND_FETCH_B;
                `endif
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
                `ifdef TCU_SPARSE_ENABLE
                    send_state_r <= slot_is_sparse[send_slot_r] ? SEND_FETCH_META : SEND_IDLE;
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
            default: send_state_r <= SEND_IDLE;
            endcase
        end
    end

    // -----------------------------------------------------------------------
    // Slot state updates
    // -----------------------------------------------------------------------

    always_ff @(posedge clk) begin
        if (reset) begin
            for (int s = 0; s < TCU_TBUF_SIZE; ++s) begin
                slot_valid[s]         <= 1'b0;
                slot_fetch_done[s]    <= 1'b0;
                warp_alloc_pending[s] <= 1'b0;
            end
        `ifndef TCU_SPARSE_ENABLE
            shared_b_valid        <= 1'b0;
            shared_b_dirty        <= 1'b1;
            shared_b_desc         <= '0;
            last_b_fetch_slot_r   <= '0;
        `endif
        end else begin
        `ifndef TCU_SPARSE_ENABLE
            if (alloc_en && (slot_idx == last_b_fetch_slot_r)
                         && shared_b_valid && (req_desc_b == shared_b_desc))
                shared_b_dirty <= 1'b1;
            if (last_rsp_b) begin
                shared_b_valid      <= 1'b1;
                shared_b_dirty      <= 1'b0;
                shared_b_desc       <= slot_desc_b[send_slot_r];
                last_b_fetch_slot_r <= send_slot_r;
            end
        `endif
            if (fetch_done_now)
                slot_fetch_done[send_slot_r] <= 1'b1;

            if (alloc_en) begin
                slot_valid      [slot_idx] <= 1'b1;
                slot_fetch_done [slot_idx] <= 1'b0;
                slot_desc_a     [slot_idx] <= req_desc_a;
                slot_desc_b     [slot_idx] <= req_desc_b;
                slot_a_row_base [slot_idx] <= desc_a_row_base;
                slot_b_row_base [slot_idx] <= desc_b_row_base;
            `ifdef TCU_SPARSE_ENABLE
                slot_is_sparse      [slot_idx] <= is_sparse;
                slot_meta_row_base  [slot_idx] <= desc_meta_row_base;
                slot_meta_stride    [slot_idx] <= init_meta_stride;
                slot_meta_bank_rows [slot_idx] <= init_meta_rows;
            `endif
            end

            if (alloc_en)
                warp_alloc_pending[slot_idx] <= 1'b1;
            if (req_fire && is_first_uop)
                warp_alloc_pending[slot_idx] <= 1'b0;
        end
    end

    // -----------------------------------------------------------------------
    // slot_a_buf — VX_dp_ram (LUTRAM, async read)
    //
    // One RAM row per slot; width = A_TOTAL*32 bits; WRENW = A_TOTAL enables
    // single-word writes.  Each response cycle, NUM_BANKS consecutive words
    // are written via a sparse wren mask decoded from rsp_ctr_r + bank index.
    //
    // Area: A_TOTAL*TCU_TBUF_SIZE*32 FFs replaced by A_TOTAL*32 LUTRAM cells.
    // Timing: async read is combinational, same latency as the prior FF MUX.
    // -----------------------------------------------------------------------

    logic [A_TOTAL*32-1:0] a_wdata;
    logic [A_TOTAL-1:0]    a_wren;

    always_comb begin
        a_wdata = '0;
        a_wren  = '0;
        if (tcu_lmem_if.rsp_valid && in_fetch_a) begin
            for (int b = 0; b < NUM_BANKS; ++b) begin
                if (int'(rsp_ctr_r) * NUM_BANKS + b < A_TOTAL) begin
                    a_wren[int'(rsp_ctr_r) * NUM_BANKS + b]               = 1'b1;
                    a_wdata[(int'(rsp_ctr_r) * NUM_BANKS + b) * 32 +: 32] =
                        tcu_lmem_if.rsp_data[b * `XLEN +: `XLEN];
                end
            end
        end
    end

    VX_dp_ram #(
        .DATAW   (A_TOTAL * 32),
        .SIZE    (TCU_TBUF_SIZE),
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
        .waddr (send_slot_r),
        .wdata (a_wdata),
        .raddr (slot_idx),
        .rdata (hit_a_buf)
    );

    // -----------------------------------------------------------------------
    // B data capture
    //   Non-sparse: shared_b_buf FFs.  All slots share the same desc_b (all
    //     warps in a multi-warp CTA share the same physical smem segment),
    //     so one global copy of B is sufficient.
    //   Sparse: per-slot slot_b_ram.  Each single-warp CTA has its own smem
    //     segment with a different desc_b, so B must be stored per-slot to
    //     prevent later fetches from overwriting earlier slots' B data.
    // -----------------------------------------------------------------------

`ifndef TCU_SPARSE_ENABLE
    // shared_b_ram — single-entry LUTRAM (SIZE=1, async read).
    // Replaces B_TOTAL*32 FFs with LUTRAM cells at no timing cost.
    logic [B_TOTAL*32-1:0] b_wdata;
    logic [B_TOTAL-1:0]    b_wren;

    always_comb begin
        /* verilator lint_off WIDTHCONCAT */
        b_wdata = '0;
        b_wren  = '0;
        /* verilator lint_on WIDTHCONCAT */
        if (tcu_lmem_if.rsp_valid && in_fetch_b) begin
            for (int b = 0; b < NUM_BANKS; ++b) begin
                if (int'(rsp_ctr_r) * NUM_BANKS + b < B_TOTAL) begin
                    b_wren[int'(rsp_ctr_r) * NUM_BANKS + b]               = 1'b1;
                    b_wdata[(int'(rsp_ctr_r) * NUM_BANKS + b) * 32 +: 32] =
                        tcu_lmem_if.rsp_data[b * `XLEN +: `XLEN];
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
`else // TCU_SPARSE_ENABLE
    logic [B_TOTAL*32-1:0] b_wdata;
    logic [B_TOTAL-1:0]    b_wren;

    always_comb begin
        /* verilator lint_off WIDTHCONCAT */
        b_wdata = '0;
        b_wren  = '0;
        /* verilator lint_on WIDTHCONCAT */
        if (tcu_lmem_if.rsp_valid && in_fetch_b) begin
            for (int b = 0; b < NUM_BANKS; ++b) begin
                if (int'(rsp_ctr_r) * NUM_BANKS + b < B_TOTAL) begin
                    b_wren[int'(rsp_ctr_r) * NUM_BANKS + b]               = 1'b1;
                    b_wdata[(int'(rsp_ctr_r) * NUM_BANKS + b) * 32 +: 32] =
                        tcu_lmem_if.rsp_data[b * `XLEN +: `XLEN];
                end
            end
        end
    end

    VX_dp_ram #(
        .DATAW   (B_TOTAL * 32),
        .SIZE    (TCU_TBUF_SIZE),
        .WRENW   (B_TOTAL),
        .LUTRAM  (1),
        .OUT_REG (0),
        .RDW_MODE("W")
    ) slot_b_ram (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (tcu_lmem_if.rsp_valid && in_fetch_b),
        .wren  (b_wren),
        .waddr (send_slot_r),
        .wdata (b_wdata),
        .raddr (slot_idx),
        .rdata (hit_b_buf)
    );
`endif

    // -----------------------------------------------------------------------
    // slot_meta_buf — VX_dp_ram (sparse only, same pattern as slot_a_ram)
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
                        tcu_lmem_if.rsp_data[b * `XLEN +: `XLEN];
                end
            end
        end
    end

    VX_dp_ram #(
        .DATAW   (META_TOTAL_MAX * 32),
        .SIZE    (TCU_TBUF_SIZE),
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
        .waddr (send_slot_r),
        .wdata (meta_wdata),
        .raddr (slot_idx),
        .rdata (hit_meta_buf)
    );

    assign hit_is_sparse   = slot_is_sparse[slot_idx];
    assign hit_meta_stride = slot_meta_stride[slot_idx];
`endif

    // -----------------------------------------------------------------------
    // Performance counters
    // -----------------------------------------------------------------------

`ifdef PERF_ENABLE
    reg [PERF_CTR_BITS-1:0] fetch_stall_ctr_r;
    reg [PERF_CTR_BITS-1:0] lmem_reads_ctr_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            fetch_stall_ctr_r <= '0;
            lmem_reads_ctr_r  <= '0;
        end else begin
            if (req_valid && !tbuf_ready)
                fetch_stall_ctr_r <= fetch_stall_ctr_r + PERF_CTR_BITS'(1);
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                lmem_reads_ctr_r <= lmem_reads_ctr_r + PERF_CTR_BITS'(1);
        end
    end
    assign tbuf_fetch_stalls = fetch_stall_ctr_r;
    assign lmem_reads        = lmem_reads_ctr_r;
`endif

    // -----------------------------------------------------------------------
    // Debug trace
    // -----------------------------------------------------------------------

`ifdef DBG_TRACE_TCU
    always @(posedge clk) begin
        if (!reset) begin
            if (alloc_en)
                `TRACE(3, ("%t: %s tbuf-fetch: alloc slot=%0d wid=%0d desc_a=0x%0h desc_b=0x%0h sparse=%0b\n",
                    $time, INSTANCE_ID, slot_idx, req_wid, req_desc_a, req_desc_b, is_sparse))
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                `TRACE(3, ("%t: %s tbuf-fetch: rd_req slot=%0d addr=0x%0h phase=%0d req_ctr=%0d\n",
                    $time, INSTANCE_ID, send_slot_r, tcu_lmem_if.req_addr, send_state_r, req_ctr_r))
            if (tcu_lmem_if.rsp_valid)
                `TRACE(3, ("%t: %s tbuf-fetch: rd_rsp slot=%0d data[0]=0x%0h phase=%0d rsp_ctr=%0d\n",
                    $time, INSTANCE_ID, send_slot_r, tcu_lmem_if.rsp_data[0 +: `XLEN], send_state_r, rsp_ctr_r))
            if (fetch_done_now)
                `TRACE(3, ("%t: %s tbuf-fetch: slot=%0d READY\n", $time, INSTANCE_ID, send_slot_r))
        end
    end
`endif

endmodule

`endif // TCU_WGMMA_ENABLE
