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
    input  wire                     req_fire,   // execute consumed current uop
    input  wire                     req_is_sparse,
    input  wire [3:0]               req_step_m,
    input  wire [3:0]               req_step_n,
    input  wire [3:0]               req_step_k,
    input  wire [3:0]               req_fmt_s,
    input  wire [`XLEN-1:0]         req_desc_a,
    input  wire [`XLEN-1:0]         req_desc_b,

    // LMEM bank-parallel read port (1-cycle latency, pipelined)
    VX_mem_bus_if.master            tcu_lmem_if,

    // Hit status
    output wire                     tbuf_hit,
    output wire                     tbuf_ready,

    // Slot data (combinational read)
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
    // Single slot state
    // -----------------------------------------------------------------------

    logic                       slot_valid;
    logic [`XLEN-1:0]           slot_desc_a;
    logic [`XLEN-1:0]           slot_desc_b;
    logic                       slot_fetch_done;
    logic                       alloc_pending;
    logic [BANK_ADDR_WIDTH-1:0] slot_a_row_base;
    logic [BANK_ADDR_WIDTH-1:0] slot_b_row_base;

`ifdef TCU_SPARSE_ENABLE
    logic                       slot_is_sparse;
    logic [BANK_ADDR_WIDTH-1:0] slot_meta_row_base;
    logic [3:0]                 slot_meta_stride;
    logic [FETCH_CTR_W-1:0]     slot_meta_bank_rows;
`endif

    // -----------------------------------------------------------------------
    // Slot lookup
    // -----------------------------------------------------------------------

    wire is_first_uop = (req_step_m == '0) && (req_step_n == '0) && (req_step_k == '0);

    // Descriptor match: always validate against current slot contents.
    wire desc_match = (slot_desc_a == req_desc_a) && (slot_desc_b == req_desc_b);

    // Hit: slot is valid, data ready, and descriptors match.
    assign tbuf_hit = slot_valid && slot_fetch_done && desc_match;

    // In-progress: fetch for slot is running.
    wire slot_in_progress = slot_valid && !slot_fetch_done;

    // Allocate (re-fetch) on:
    //   - first uop of every tile (LMEM contents may have changed via DMA
    //     even if the descriptor address is the same), OR
    //   - any uop with a descriptor mismatch (handles warp interleaving:
    //     a different warp may have evicted this warp's tile data).
    wire alloc_en = req_valid && !slot_in_progress && !alloc_pending
                 && (is_first_uop || !slot_valid || !slot_fetch_done || !desc_match);

    assign tbuf_ready = tbuf_hit && !alloc_en;

    // -----------------------------------------------------------------------
    // Descriptor parsing -> row bases (combinational, used only at alloc)
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

    // Per-phase termination threshold and row base.
`ifndef TCU_SPARSE_ENABLE
    wire [FETCH_CTR_W-1:0]     phase_total_rows =
        in_fetch_a ? FETCH_CTR_W'(A_BANK_ROWS) : FETCH_CTR_W'(B_BANK_ROWS);

    wire [BANK_ADDR_WIDTH-1:0] phase_row_base =
        in_fetch_a ? slot_a_row_base : slot_b_row_base;
`else
    logic [FETCH_CTR_W-1:0]     phase_total_rows;
    logic [BANK_ADDR_WIDTH-1:0] phase_row_base;
    always_comb begin
        case (send_state_r)
            SEND_FETCH_A: begin
                phase_total_rows = FETCH_CTR_W'(slot_is_sparse
                                   ? A_BANK_ROWS_SP : A_BANK_ROWS);
                phase_row_base   = slot_a_row_base;
            end
            SEND_FETCH_B: begin
                phase_total_rows = FETCH_CTR_W'(B_BANK_ROWS);
                phase_row_base   = slot_b_row_base;
            end
            SEND_FETCH_META: begin
                phase_total_rows = slot_meta_bank_rows;
                phase_row_base   = slot_meta_row_base;
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

    assign tcu_lmem_if.req_valid       = can_issue;
    assign tcu_lmem_if.req_data.rw     = 1'b0;
    assign tcu_lmem_if.req_data.addr   = phase_row_base + BANK_ADDR_WIDTH'(req_ctr_r);
    assign tcu_lmem_if.req_data.data   = '0;
    assign tcu_lmem_if.req_data.byteen = '0;
    assign tcu_lmem_if.req_data.flags  = '0;
    assign tcu_lmem_if.req_data.tag    = '0;
    assign tcu_lmem_if.rsp_ready       = 1'b1;

    // -----------------------------------------------------------------------
    // Phase-done detection
    // -----------------------------------------------------------------------

`ifdef TCU_SPARSE_ENABLE
    wire last_rsp_a = in_fetch_a && tcu_lmem_if.rsp_valid
                   && (rsp_ctr_r == (slot_is_sparse
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
                      && (rsp_ctr_r == slot_meta_bank_rows - FETCH_CTR_W'(1));
    wire fetch_done_now = (last_rsp_b && !slot_is_sparse) || last_rsp_meta;
`else
    wire fetch_done_now = last_rsp_b;
`endif

    // -----------------------------------------------------------------------
    // Sender FSM
    // -----------------------------------------------------------------------

    always_ff @(posedge clk) begin
        if (reset) begin
            send_state_r   <= SEND_IDLE;
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
                if (slot_valid && !slot_fetch_done) begin
                    req_ctr_r      <= '0;
                    rsp_ctr_r      <= '0;
                    req_inflight_r <= 1'b0;
                    send_state_r   <= SEND_FETCH_A;
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
                    send_state_r   <= SEND_FETCH_B;
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
                    send_state_r <= slot_is_sparse ? SEND_FETCH_META : SEND_IDLE;
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
            slot_valid      <= 1'b0;
            slot_fetch_done <= 1'b0;
            alloc_pending   <= 1'b0;
        end else begin
            if (fetch_done_now)
                slot_fetch_done <= 1'b1;

            if (alloc_en) begin
                slot_valid      <= 1'b1;
                slot_fetch_done <= 1'b0;
                slot_desc_a     <= req_desc_a;
                slot_desc_b     <= req_desc_b;
                slot_a_row_base <= desc_a_row_base;
                slot_b_row_base <= desc_b_row_base;
            `ifdef TCU_SPARSE_ENABLE
                slot_is_sparse      <= is_sparse;
                slot_meta_row_base  <= desc_meta_row_base;
                slot_meta_stride    <= init_meta_stride;
                slot_meta_bank_rows <= init_meta_rows;
            `endif
            end

            if (alloc_en)
                alloc_pending <= 1'b1;
            if (req_fire)
                alloc_pending <= 1'b0;
        end
    end

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
        .SIZE    (1),
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
        .waddr (1'b0),
        .wdata (a_wdata),
        .raddr (1'b0),
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
    ) slot_b_ram (
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
        .SIZE    (1),
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
        .waddr (1'b0),
        .wdata (meta_wdata),
        .raddr (1'b0),
        .rdata (hit_meta_buf)
    );

    assign hit_is_sparse   = slot_is_sparse;
    assign hit_meta_stride = slot_meta_stride;
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
                `TRACE(3, ("%t: %s tbuf-fetch: alloc desc_a=0x%0h desc_b=0x%0h sparse=%0b\n",
                    $time, INSTANCE_ID, req_desc_a, req_desc_b, is_sparse))
            if (tcu_lmem_if.req_valid && tcu_lmem_if.req_ready)
                `TRACE(3, ("%t: %s tbuf-fetch: rd_req addr=0x%0h phase=%0d req_ctr=%0d\n",
                    $time, INSTANCE_ID, tcu_lmem_if.req_data.addr, send_state_r, req_ctr_r))
            if (tcu_lmem_if.rsp_valid)
                `TRACE(3, ("%t: %s tbuf-fetch: rd_rsp data[0]=0x%0h phase=%0d rsp_ctr=%0d\n",
                    $time, INSTANCE_ID, tcu_lmem_if.rsp_data.data[0 +: `XLEN], send_state_r, rsp_ctr_r))
            if (fetch_done_now)
                `TRACE(3, ("%t: %s tbuf-fetch: READY\n", $time, INSTANCE_ID))
        end
    end
`endif

endmodule

`endif // TCU_WGMMA_ENABLE
