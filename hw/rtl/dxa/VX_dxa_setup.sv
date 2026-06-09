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

// DXA Setup: owns transfer lifecycle, issue decode, metadata latching,
// and DSP-based parameter precomputation. The setup engine runs in
// parallel with an active transfer's drain: a new request is accepted
// the cycle the DSPs are idle, the result lands in `staged_*`, and is
// promoted to the live `r_*` registers the cycle the previous transfer
// completes. Setup latency overlaps with drain, eliminating the
// pipeline bubble for streamed transfers.
//
// Rank-dependent setup latency (cycles from launch_accept to staged):
//   rank 1-2: 4, rank 3: 6, rank 4: 8, rank 5: 10.
// DSP operands are registered for timing closure; this overlaps drain
// and is throughput-neutral.

`include "VX_define.vh"

module VX_dxa_setup import VX_gpu_pkg::*, VX_dxa_pkg::*; (
    input  wire                        clk,
    input  wire                        reset,

    // Launch interface (from dispatch).
    input  wire                        req_valid,
    output wire                        req_ready,
    input  wire dxa_req_data_t         req_data,
    input  wire dxa_desc_t             desc_data,

    // Transfer done (from smem_wr).
    input  wire                        transfer_done,

    // State outputs.
    output wire                        transfer_active,
    output wire                        pipeline_start,

    // Precomputed parameters (stable during ACTIVE).
    output wire dxa_setup_params_t     setup_params,

    // Latched metadata (stable during ACTIVE).
    output wire [NC_WIDTH-1:0]         active_core_id,
    output wire [UUID_WIDTH-1:0]       active_uuid,
    output wire [NW_WIDTH-1:0]         active_wid,
    output wire [BAR_ADDR_W-1:0]       active_bar_addr,
    output wire                        active_notify_smem_done,

    // Multicast (always available; active when cta_mask has >1 bit set).
    output wire                        active_is_multicast,
    output wire [`VX_CFG_NUM_WARPS-1:0]       active_cta_mask,
    output wire [31:0]                 active_smem_stride
);
    localparam MUL_LATENCY = 2;

    // ════════════════════════════════════════════════════════════════════
    // Transfer state: TS_IDLE / TS_ACTIVE.
    // (TS_SETUP is gone — setup is tracked separately by setup_state_r.)
    // ════════════════════════════════════════════════════════════════════
    localparam TS_IDLE   = 1'b0;
    localparam TS_ACTIVE = 1'b1;

    reg state_r;
    assign transfer_active = (state_r == TS_ACTIVE);

    // ════════════════════════════════════════════════════════════════════
    // Setup engine state: SS_IDLE / SS_RUNNING / SS_STAGED.
    // ════════════════════════════════════════════════════════════════════
    localparam SS_IDLE    = 2'd0;
    localparam SS_RUNNING = 2'd1;
    localparam SS_STAGED  = 2'd2;

    reg [1:0] setup_state_r;

    // Accept a new request whenever the setup engine has capacity (i.e.,
    // not currently computing nor holding a staged result).
    assign req_ready = (setup_state_r == SS_IDLE);

    wire launch_accept = req_valid && req_ready;

    // ════════════════════════════════════════════════════════════════════
    // Issue decode (combinatorial)
    // ════════════════════════════════════════════════════════════════════

    wire [`VX_DXA_DESC_META_ELEMSZ_BITS-1:0] esize_enc =
        desc_data.meta[`VX_DXA_DESC_META_ELEMSZ_LSB +: `VX_DXA_DESC_META_ELEMSZ_BITS];
    wire [`VX_DXA_DESC_META_DIM_BITS-1:0] rank_raw =
        desc_data.meta[`VX_DXA_DESC_META_DIM_LSB +: `VX_DXA_DESC_META_DIM_BITS];
    // LAYOUT bit (meta[5]): 0 = row-major SMEM (default), 1 = K-major
    // SMEM (NVIDIA-TMA transposing mode). Restricted to rank ≤ 2.
    wire dec_dest_kmajor =
        desc_data.meta[`VX_DXA_DESC_META_LAYOUT_LSB +: `VX_DXA_DESC_META_LAYOUT_BITS];

    // Assume correct input: rank in [1,5], tiles/sizes nonzero for active dims.
    wire [31:0] dec_rank       = 32'(rank_raw);
    wire [31:0] dec_elem_bytes = 32'(1) << esize_enc;

    wire [31:0] dec_tile0 = 32'(desc_data.tile01[15:0]);
    wire [31:0] dec_tile1 = (dec_rank >= 2) ? 32'(desc_data.tile01[31:16]) : 32'd1;
    wire [31:0] dec_tile2 = (dec_rank >= 3) ? 32'(desc_data.tile23[15:0])  : 32'd1;
    wire [31:0] dec_tile3 = (dec_rank >= 4) ? 32'(desc_data.tile23[31:16]) : 32'd1;
    wire [31:0] dec_tile4 = (dec_rank >= 5) ? desc_data.tile4              : 32'd1;

    wire [31:0] dec_stride0 = (dec_rank >= 2) ? desc_data.stride0 : 32'd0;
    wire [31:0] dec_stride1 = (dec_rank >= 3) ? desc_data.stride1 : 32'd0;
    wire [31:0] dec_stride2 = (dec_rank >= 4) ? desc_data.stride2 : 32'd0;
    wire [31:0] dec_stride3 = (dec_rank >= 5) ? desc_data.stride3 : 32'd0;

    wire [31:0] dec_size1 = (dec_rank >= 2) ? desc_data.size1 : 32'd1;
    wire [31:0] dec_size2 = (dec_rank >= 3) ? desc_data.size2 : 32'd1;
    wire [31:0] dec_size3 = (dec_rank >= 4) ? desc_data.size3 : 32'd1;
    wire [31:0] dec_size4 = (dec_rank >= 5) ? desc_data.size4 : 32'd1;

    `RUNTIME_ASSERT(!launch_accept || (dec_rank >= 1 && dec_rank <= 5), ("invalid DXA rank: %0d", dec_rank))
    `RUNTIME_ASSERT(!launch_accept || !dec_dest_kmajor || (dec_rank <= 2),
        ("DXA K-major (LAYOUT=1) requires rank ≤ 2, got rank=%0d", dec_rank))

    // K-major per-lane SMEM stride = tile1 * elem_bytes. Since elem_bytes is
    // always a power of 2 (encoded as esize_enc), this collapses to a shift.
    // tile1 ≤ 32 and elem_bytes ≤ 8, so 16 bits is sufficient.
    wire [31:0] dec_per_lane_stride_bytes_full = dec_tile1 << esize_enc;
    wire [15:0] dec_per_lane_stride_bytes = dec_per_lane_stride_bytes_full[15:0];
    `UNUSED_VAR (dec_per_lane_stride_bytes_full[31:16])

    // ════════════════════════════════════════════════════════════════════
    // Bar address decode (from req_data.meta)
    // ════════════════════════════════════════════════════════════════════

    wire [BAR_ADDR_W-1:0] launch_bar_addr;
    if (`VX_CFG_NUM_WARPS > 1) begin : g_bar_w
        assign launch_bar_addr = {req_data.meta[4 +: NW_BITS], req_data.meta[(4 + BAR_ID_SHIFT) +: NB_BITS]};
    end else begin : g_bar_wo
        assign launch_bar_addr = req_data.meta[(4 + BAR_ID_SHIFT) +: NB_BITS];
    end

    // ════════════════════════════════════════════════════════════════════
    // Active result registers (used live by downstream).
    // ════════════════════════════════════════════════════════════════════

    reg [NC_WIDTH-1:0]                     r_core_id;
    reg [UUID_WIDTH-1:0]                   r_uuid;
    reg [NW_WIDTH-1:0]                     r_wid;
    reg [BAR_ADDR_W-1:0]                   r_bar_addr;
    reg                                    r_notify_smem_done;
    reg                                    r_is_multicast;
    reg [`VX_CFG_NUM_WARPS-1:0]            r_cta_mask;
    reg [31:0]                             r_smem_stride;
    reg [`VX_CFG_MEM_ADDR_WIDTH-1:0]       r_initial_gmem_base;
    reg [DXA_SMEM_ADDR_W-1:0]              r_initial_smem_base;
    reg [31:0]                             r_row_len_bytes;
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0]     r_delta;
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0]     r_dim_tiles;
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0]     r_oob_limit;
    reg [31:0]                             r_cfill;
    reg                                    r_dest_kmajor;
    reg [15:0]                             r_per_lane_stride_bytes;
    reg [3:0]                              r_elem_bytes;

    // ════════════════════════════════════════════════════════════════════
    // Staged result registers (filled by the setup engine for the next
    // transfer while the current one is still draining).
    // ════════════════════════════════════════════════════════════════════

    reg [NC_WIDTH-1:0]                     s_core_id;
    reg [UUID_WIDTH-1:0]                   s_uuid;
    reg [NW_WIDTH-1:0]                     s_wid;
    reg [BAR_ADDR_W-1:0]                   s_bar_addr;
    reg                                    s_notify_smem_done;
    reg                                    s_is_multicast;
    reg [`VX_CFG_NUM_WARPS-1:0]            s_cta_mask;
    reg [31:0]                             s_smem_stride;
    reg [`VX_CFG_MEM_ADDR_WIDTH-1:0]       s_initial_gmem_base;
    reg [DXA_SMEM_ADDR_W-1:0]              s_initial_smem_base;
    reg [31:0]                             s_row_len_bytes;
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0]     s_delta;
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0]     s_dim_tiles;
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0]     s_oob_limit;
    reg [31:0]                             s_cfill;
    reg                                    s_dest_kmajor;
    reg [15:0]                             s_per_lane_stride_bytes;
    reg [3:0]                              s_elem_bytes;

    assign active_core_id          = r_core_id;
    assign active_uuid             = r_uuid;
    assign active_wid              = r_wid;
    assign active_bar_addr         = r_bar_addr;
    assign active_notify_smem_done = r_notify_smem_done;
    assign active_is_multicast     = r_is_multicast;
    assign active_cta_mask         = r_cta_mask;
    assign active_smem_stride      = r_smem_stride;

    // ════════════════════════════════════════════════════════════════════
    // Setup pipeline: 3 parallel DSP multipliers, multi-phase.
    // ════════════════════════════════════════════════════════════════════
    //
    // Phase 0 (ctr_r=0, always):  mul0 = tile0×elem_bytes (→ row_len_bytes)
    //                             mul1 = coord0×elem_bytes (→ dim0 offset)
    //                             mul2 = coord1×stride0   (→ dim1 offset)
    // Phase 1 (ctr_r=2, rank≥3):  mul1 = coord2×stride1   (→ dim2 offset)
    // Phase 2 (ctr_r=4, rank≥4):  mul1 = coord3×stride2   (→ dim3 offset)
    // Phase 3 (ctr_r=6, rank≥5):  mul1 = coord4×stride3   (→ dim4 offset)
    //
    // Captures (posedge): ctr_r ∈ {2, 4, 6, 8}.

    reg [3:0] ctr_r;

    // Operand latches — set at launch_accept, consumed across phases.
    // 'lat_rank' is used to gate which phases run.
    reg [31:0] lat_rank;
    reg [31:0] lat_tile0, lat_elem_bytes;
    reg [31:0] lat_coord0, lat_coord1, lat_stride0;
    reg [31:0] lat_coord2, lat_coord3, lat_coord4;
    reg [31:0] lat_stride1, lat_stride2, lat_stride3;

    // Multiplier instances.
    wire [31:0] mul0_result, mul1_result, mul2_result;
    reg  [31:0] mul0_a, mul0_b, mul1_a, mul1_b, mul2_a, mul2_b;
    // Registered operands: gives each DSP an input-register stage so the
    // operand-mux + (tile-1) subtract is not in the same combinational cone
    // as the multiply. Adds +1 cycle of setup latency (phase captures shift
    // by one below). Setup overlaps drain, so the extra cycle is immaterial.
    reg  [31:0] mul0_a_r, mul0_b_r, mul1_a_r, mul1_b_r, mul2_a_r, mul2_b_r;
    always @(posedge clk) begin
        mul0_a_r <= mul0_a; mul0_b_r <= mul0_b;
        mul1_a_r <= mul1_a; mul1_b_r <= mul1_b;
        mul2_a_r <= mul2_a; mul2_b_r <= mul2_b;
    end

    VX_multiplier #(
        .A_WIDTH (32),
        .B_WIDTH (32),
        .R_WIDTH (32),
        .LATENCY (MUL_LATENCY)
    ) mul0 (
        .clk    (clk),
        .enable (1'b1),
        .dataa  (mul0_a_r),
        .datab  (mul0_b_r),
        .result (mul0_result)
    );

    VX_multiplier #(
        .A_WIDTH (32),
        .B_WIDTH (32),
        .R_WIDTH (32),
        .LATENCY (MUL_LATENCY)
    ) mul1 (
        .clk    (clk),
        .enable (1'b1),
        .dataa  (mul1_a_r),
        .datab  (mul1_b_r),
        .result (mul1_result)
    );

    VX_multiplier #(
        .A_WIDTH (32),
        .B_WIDTH (32),
        .R_WIDTH (32),
        .LATENCY (MUL_LATENCY)
    ) mul2 (
        .clk    (clk),
        .enable (1'b1),
        .dataa  (mul2_a_r),
        .datab  (mul2_b_r),
        .result (mul2_result)
    );

    // Latched dim tiles (for wrap-delta computation).
    reg [31:0] lat_tile1, lat_tile2, lat_tile3;

    // Multiplier input mux (driven by ctr_r and lat_rank).
    // Phases 1-3 use the otherwise-idle mul0 to precompute wrap deltas:
    //   delta_wrap[d] = stride[d] - (tile[d-1] - 1) * stride[d-1].
    always @(*) begin
        mul0_a = '0; mul0_b = '0;
        mul1_a = '0; mul1_b = '0;
        mul2_a = '0; mul2_b = '0;

        case (ctr_r)
        4'd0: begin
            // Phase 0: always runs.
            mul0_a = lat_tile0;  mul0_b = lat_elem_bytes;
            mul1_a = lat_coord0; mul1_b = lat_elem_bytes;
            mul2_a = lat_coord1; mul2_b = lat_stride0;
        end
        4'd2: begin
            // Phase 1: rank ≥ 3.
            if (lat_rank >= 3) begin
                mul0_a = lat_tile1 - 32'd1; mul0_b = lat_stride0;  // (tile1-1)*stride0
                mul1_a = lat_coord2;        mul1_b = lat_stride1;
            end
        end
        4'd4: begin
            // Phase 2: rank ≥ 4.
            if (lat_rank >= 4) begin
                mul0_a = lat_tile2 - 32'd1; mul0_b = lat_stride1;  // (tile2-1)*stride1
                mul1_a = lat_coord3;        mul1_b = lat_stride2;
            end
        end
        4'd6: begin
            // Phase 3: rank ≥ 5.
            if (lat_rank >= 5) begin
                mul0_a = lat_tile3 - 32'd1; mul0_b = lat_stride2;  // (tile3-1)*stride2
                mul1_a = lat_coord4;        mul1_b = lat_stride3;
            end
        end
        default: ;
        endcase
    end

    // Cycle at which the LAST capture fires (= setup_state goes STAGED).
    //   rank 1-2: cycle 2 (Phase 0 capture only).
    //   rank 3:   cycle 4.  rank 4: cycle 6.  rank 5: cycle 8.
    wire [3:0] done_at_ctr =
        (lat_rank <= 2) ? 4'd3 :
        (lat_rank == 3) ? 4'd5 :
        (lat_rank == 4) ? 4'd7 :
                          4'd9;

    // ════════════════════════════════════════════════════════════════════
    // Promote: copy staged → active and pulse pipeline_start.
    // Fires when a staged result exists AND the active slot is free
    // (either idle, or completing this cycle via transfer_done).
    // ════════════════════════════════════════════════════════════════════

    wire promote_now = (setup_state_r == SS_STAGED)
                    && (state_r == TS_IDLE || transfer_done);

    reg pipeline_start_r;
    assign pipeline_start = pipeline_start_r;

    // ── OOB limit helpers (subtraction, no DSP needed) ──
    function automatic [31:0] oob_sub(input [31:0] sz, input [31:0] coord);
        oob_sub = (sz > coord) ? (sz - coord) : 32'd0;
    endfunction

    // ════════════════════════════════════════════════════════════════════
    // Sequential update
    // ════════════════════════════════════════════════════════════════════
    always @(posedge clk) begin
        if (reset) begin
            state_r            <= TS_IDLE;
            setup_state_r      <= SS_IDLE;
            ctr_r              <= '0;
            pipeline_start_r   <= 1'b0;
            r_core_id          <= '0;
            r_uuid             <= '0;
            r_wid              <= '0;
            r_bar_addr         <= '0;
            r_notify_smem_done <= 1'b0;
            // dest_kmajor / elem_bytes / per_lane_stride feed mux selectors
            // in addr_gen / smem_wr. Reset them so X doesn't propagate via
            // ag_dest_kmajor before the first transfer is staged.
            r_dest_kmajor      <= 1'b0;
            r_per_lane_stride_bytes <= '0;
            r_elem_bytes       <= '0;
            s_dest_kmajor      <= 1'b0;
            s_per_lane_stride_bytes <= '0;
            s_elem_bytes       <= '0;
        end else begin
            // Default: pipeline_start is a 1-cycle pulse.
            pipeline_start_r <= 1'b0;

            // ── Transfer state ───────────────────────────────────────────
            if (promote_now) begin
                // Move staged → active and fire pipeline_start.
                r_core_id           <= s_core_id;
                r_uuid              <= s_uuid;
                r_wid               <= s_wid;
                r_bar_addr          <= s_bar_addr;
                r_notify_smem_done  <= s_notify_smem_done;
                r_is_multicast      <= s_is_multicast;
                r_cta_mask          <= s_cta_mask;
                r_smem_stride       <= s_smem_stride;
                r_initial_gmem_base <= s_initial_gmem_base;
                r_initial_smem_base <= s_initial_smem_base;
                r_row_len_bytes     <= s_row_len_bytes;
                r_delta             <= s_delta;
                r_dim_tiles         <= s_dim_tiles;
                r_oob_limit         <= s_oob_limit;
                r_cfill             <= s_cfill;
                r_dest_kmajor       <= s_dest_kmajor;
                r_per_lane_stride_bytes <= s_per_lane_stride_bytes;
                r_elem_bytes        <= s_elem_bytes;
                state_r             <= TS_ACTIVE;
                pipeline_start_r    <= 1'b1;
                setup_state_r       <= SS_IDLE;
            end else if (state_r == TS_ACTIVE && transfer_done) begin
                // Active drain complete, no staged result waiting.
                state_r <= TS_IDLE;
            end

            // ── Setup engine ─────────────────────────────────────────────
            case (setup_state_r)
            SS_IDLE: begin
                if (launch_accept) begin
                    // Latch metadata directly into staged_*.
                    s_core_id           <= req_data.core_id;
                    s_uuid              <= req_data.uuid;
                    s_wid               <= req_data.wid;
                    s_bar_addr          <= launch_bar_addr;
                `ifdef VX_CFG_EXT_DXA_ENABLE
                    s_notify_smem_done  <= 1'b1;
                `else
                    s_notify_smem_done  <= 1'b0;
                `endif
                    s_is_multicast      <= ($countones(req_data.cta_mask) > 1);
                    s_cta_mask          <= req_data.cta_mask;
                    s_smem_stride       <= desc_data.smem_stride;
                    s_initial_smem_base <= req_data.smem_addr;
                    s_cfill             <= desc_data.cfill;
                    s_dest_kmajor       <= dec_dest_kmajor;
                    s_per_lane_stride_bytes <= dec_per_lane_stride_bytes;
                    s_elem_bytes        <= 4'(dec_elem_bytes);
                    // Rolling-cursor deltas: delta[0] is the inner-dim step.
                    // Higher deltas are precomputed below (Phase 1-3 captures).
                    s_delta[0]          <= dec_stride0;
                    s_delta[1]          <= '0;
                    s_delta[2]          <= '0;
                    s_delta[3]          <= '0;
                    s_dim_tiles[0]      <= dec_tile1;
                    s_dim_tiles[1]      <= dec_tile2;
                    s_dim_tiles[2]      <= dec_tile3;
                    s_dim_tiles[3]      <= dec_tile4;
                    // OOB limits.
                    s_oob_limit[0]      <= oob_sub(dec_size1, (dec_rank >= 2) ? req_data.coords[1][31:0] : 32'd0);
                    s_oob_limit[1]      <= oob_sub(dec_size2, (dec_rank >= 3) ? req_data.coords[2][31:0] : 32'd0);
                    s_oob_limit[2]      <= oob_sub(dec_size3, (dec_rank >= 4) ? req_data.coords[3][31:0] : 32'd0);
                    s_oob_limit[3]      <= oob_sub(dec_size4, (dec_rank >= 5) ? req_data.coords[4][31:0] : 32'd0);
                    // Operand latches (DSP inputs across phases).
                    lat_rank            <= dec_rank;
                    lat_tile0           <= dec_tile0;
                    lat_tile1           <= dec_tile1;
                    lat_tile2           <= dec_tile2;
                    lat_tile3           <= dec_tile3;
                    lat_elem_bytes      <= dec_elem_bytes;
                    lat_coord0          <= req_data.coords[0][31:0];
                    lat_coord1          <= (dec_rank >= 2) ? req_data.coords[1][31:0] : 32'd0;
                    lat_coord2          <= (dec_rank >= 3) ? req_data.coords[2][31:0] : 32'd0;
                    lat_coord3          <= (dec_rank >= 4) ? req_data.coords[3][31:0] : 32'd0;
                    lat_coord4          <= (dec_rank >= 5) ? req_data.coords[4][31:0] : 32'd0;
                    lat_stride0         <= dec_stride0;
                    lat_stride1         <= dec_stride1;
                    lat_stride2         <= dec_stride2;
                    lat_stride3         <= dec_stride3;
                    // Seed staged base address with desc.base_addr.
                    s_initial_gmem_base <= desc_data.base_addr;
                    ctr_r               <= '0;
                    setup_state_r       <= SS_RUNNING;
                end
            end
            SS_RUNNING: begin
                ctr_r <= ctr_r + 4'd1;

                // ── Phase 0 capture at ctr=3 (operand reg adds +1 cycle) ──
                if (ctr_r == 4'd3) begin
                    s_row_len_bytes     <= mul0_result;
                    s_initial_gmem_base <= s_initial_gmem_base
                                         + `VX_CFG_MEM_ADDR_WIDTH'(mul1_result)
                                         + `VX_CFG_MEM_ADDR_WIDTH'(mul2_result);
                end

                // ── Phase 1 capture at ctr=5 (rank≥3) ──
                if (ctr_r == 4'd5 && lat_rank >= 3) begin
                    s_initial_gmem_base <= s_initial_gmem_base
                                         + `VX_CFG_MEM_ADDR_WIDTH'(mul1_result);
                    // delta[1] = stride1 - (tile1-1)*stride0
                    s_delta[1] <= lat_stride1 - mul0_result;
                end

                // ── Phase 2 capture at ctr=7 (rank≥4) ──
                if (ctr_r == 4'd7 && lat_rank >= 4) begin
                    s_initial_gmem_base <= s_initial_gmem_base
                                         + `VX_CFG_MEM_ADDR_WIDTH'(mul1_result);
                    // delta[2] = stride2 - (tile2-1)*stride1
                    s_delta[2] <= lat_stride2 - mul0_result;
                end

                // ── Phase 3 capture at ctr=9 (rank=5) ──
                if (ctr_r == 4'd9 && lat_rank >= 5) begin
                    s_initial_gmem_base <= s_initial_gmem_base
                                         + `VX_CFG_MEM_ADDR_WIDTH'(mul1_result);
                    // delta[3] = stride3 - (tile3-1)*stride2
                    s_delta[3] <= lat_stride3 - mul0_result;
                end

                // Transition to STAGED at the last capture cycle.
                if (ctr_r == done_at_ctr) begin
                    setup_state_r <= SS_STAGED;
                    ctr_r         <= '0;
                end
            end
            SS_STAGED: begin
                // Wait for promote_now to consume the staged result.
                // Handled in the promote block above.
            end
            default: setup_state_r <= SS_IDLE;
            endcase
        end
    end

    // ════════════════════════════════════════════════════════════════════
    // Output assignments
    // ════════════════════════════════════════════════════════════════════
    assign setup_params.initial_gmem_base  = r_initial_gmem_base;
    assign setup_params.initial_smem_base  = r_initial_smem_base;
    assign setup_params.row_len_bytes      = r_row_len_bytes;
    assign setup_params.delta              = r_delta;
    assign setup_params.dim_tiles          = r_dim_tiles;
    assign setup_params.oob_limit          = r_oob_limit;
    assign setup_params.cfill              = r_cfill;
    assign setup_params.dest_kmajor        = r_dest_kmajor;
    assign setup_params.per_lane_stride_bytes = r_per_lane_stride_bytes;
    assign setup_params.elem_bytes         = r_elem_bytes;

    `UNUSED_VAR (desc_data.size0)
    `UNUSED_VAR (req_data.meta)

endmodule
