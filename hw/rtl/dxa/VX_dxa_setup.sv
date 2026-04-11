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

// DXA Setup: owns transfer lifecycle (IDLE/SETUP/ACTIVE), issue decode,
// metadata latching, and DSP-based parameter precomputation.
// Supports 1D-5D tiles with rank-dependent setup latency:
//   rank 1-2: 6 cycles, rank 3: 8, rank 4: 10, rank 5: 12.
// Reuses 3 DSP multipliers across phases for area efficiency.

`include "VX_define.vh"

module VX_dxa_setup import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter SMEM_BYTES = DXA_LMEM_WORD_SIZE
) (
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
    output wire [`NUM_WARPS-1:0]       active_cta_mask,
    output wire [31:0]                 active_smem_stride
);
    localparam MUL_LATENCY = 2;
    localparam SMEM_OFF_BITS = `CLOG2(SMEM_BYTES);

    // ════════════════════════════════════════════════════════════════════
    // Transfer state machine: IDLE → SETUP → ACTIVE → IDLE
    // ════════════════════════════════════════════════════════════════════
    localparam TS_IDLE   = 2'd0;
    localparam TS_SETUP  = 2'd1;
    localparam TS_ACTIVE = 2'd2;

    reg [1:0] state_r;

    assign req_ready       = (state_r == TS_IDLE);
    assign transfer_active = (state_r == TS_ACTIVE);

    wire launch_accept = req_valid && req_ready;

    // ════════════════════════════════════════════════════════════════════
    // Issue decode (combinatorial)
    // ════════════════════════════════════════════════════════════════════

    wire [`VX_DXA_DESC_META_ELEMSZ_BITS-1:0] esize_enc =
        desc_data.meta[`VX_DXA_DESC_META_ELEMSZ_LSB +: `VX_DXA_DESC_META_ELEMSZ_BITS];
    wire [`VX_DXA_DESC_META_DIM_BITS-1:0] rank_raw =
        desc_data.meta[`VX_DXA_DESC_META_DIM_LSB +: `VX_DXA_DESC_META_DIM_BITS];

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

    // ════════════════════════════════════════════════════════════════════
    // Bar address decode (from req_data.meta)
    // ════════════════════════════════════════════════════════════════════

    wire [BAR_ADDR_W-1:0] launch_bar_addr;
    if (`NUM_WARPS > 1) begin : g_bar_w
        assign launch_bar_addr = {req_data.meta[4 +: NW_BITS], req_data.meta[(4 + BAR_ID_SHIFT) +: NB_BITS]};
    end else begin : g_bar_wo
        assign launch_bar_addr = req_data.meta[(4 + BAR_ID_SHIFT) +: NB_BITS];
    end

    // ════════════════════════════════════════════════════════════════════
    // Latched metadata registers
    // ════════════════════════════════════════════════════════════════════

    reg [NC_WIDTH-1:0]     r_core_id;
    reg [UUID_WIDTH-1:0]   r_uuid;
    reg [NW_WIDTH-1:0]     r_wid;
    reg [BAR_ADDR_W-1:0]   r_bar_addr;
    reg                    r_notify_smem_done;
    reg                    r_is_multicast;
    reg [`NUM_WARPS-1:0]   r_cta_mask;
    reg [31:0]             r_smem_stride;

    assign active_core_id         = r_core_id;
    assign active_uuid            = r_uuid;
    assign active_wid             = r_wid;
    assign active_bar_addr        = r_bar_addr;
    assign active_notify_smem_done = r_notify_smem_done;
    assign active_is_multicast    = r_is_multicast;
    assign active_cta_mask        = r_cta_mask;
    assign active_smem_stride     = r_smem_stride;

    // ════════════════════════════════════════════════════════════════════
    // Setup pipeline: 3 parallel DSP multipliers, multi-phase
    // ════════════════════════════════════════════════════════════════════
    //
    // Phase 0 (all ranks):
    //   mul0: tile0 × elem_bytes → row_len_bytes
    //   mul1: coord0 × elem_bytes → off0
    //   mul2: coord1 × stride0   → off1
    //
    // Phase 1 (rank≥3):
    //   mul0: tile1 × tile2 → partial_rows
    //   mul1: coord2 × stride1 → off2
    //
    // Phase 2 (rank≥4):
    //   mul0: partial_rows × tile3 → partial_rows2
    //   mul1: coord3 × stride2 → off3
    //
    // Phase 3 (rank=5):
    //   mul0: partial_rows2 × tile4 → total_rows
    //   mul1: coord4 × stride3 → off4
    //
    // Final phase:
    //   mul0: total_rows × row_len_bytes → total_bytes

    reg [3:0] ctr_r;

    // Latched multiply operands (captured on launch_accept).
    reg [31:0] lat_tile0, lat_elem_bytes;
    reg [31:0] lat_coord0, lat_coord1, lat_stride0;
    reg [31:0] lat_tile1, lat_tile2, lat_tile3, lat_tile4;
    reg [31:0] lat_coord2, lat_coord3, lat_coord4;
    reg [31:0] lat_stride1, lat_stride2, lat_stride3;
    reg [`MEM_ADDR_WIDTH-1:0] lat_gbase;

    // Registered output parameters.
    reg [`MEM_ADDR_WIDTH-1:0]              r_initial_gmem_base;
    reg [`XLEN-1:0]                        r_initial_smem_base;
    reg [31:0]                             r_row_len_bytes;
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0]     r_strides;
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0]     r_dim_tiles;
    reg [DXA_MAX_OUTER_DIMS-1:0][31:0]     r_oob_limit;
    reg [31:0]                             r_total_rows;
    reg [31:0]                             r_total_smem_writes;
    reg [31:0]                             r_total_bytes;
    reg [31:0]                             r_cfill;
    reg [31:0]                             r_elem_bytes;
    reg [31:0]                             r_rank;

    // ── Multiplier instances ──

    wire [31:0] mul0_result, mul1_result, mul2_result;
    reg  [31:0] mul0_a, mul0_b, mul1_a, mul1_b, mul2_a, mul2_b;

    VX_multiplier #(
        .A_WIDTH (32),
        .B_WIDTH (32),
        .R_WIDTH (32),
        .LATENCY (MUL_LATENCY)
    ) mul0 (
        .clk    (clk),
        .enable (1'b1),
        .dataa  (mul0_a),
        .datab  (mul0_b),
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
        .dataa  (mul1_a),
        .datab  (mul1_b),
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
        .dataa  (mul2_a),
        .datab  (mul2_b),
        .result (mul2_result)
    );

    // ── Multiplier input mux (combinatorial, driven by ctr_r and r_rank) ──

    always @(*) begin
        // Defaults
        mul0_a = '0; mul0_b = '0;
        mul1_a = '0; mul1_b = '0;
        mul2_a = '0; mul2_b = '0;

        case (ctr_r)
        4'd0: begin
            // Phase 0: tile0×elem_bytes, coord0×elem_bytes, coord1×stride0
            mul0_a = lat_tile0;  mul0_b = lat_elem_bytes;
            mul1_a = lat_coord0; mul1_b = lat_elem_bytes;
            mul2_a = lat_coord1; mul2_b = lat_stride0;
        end
        4'd2: begin
            if (r_rank <= 2) begin
                // Final phase: total_rows(=tile1) × row_len_bytes
                mul0_a = r_total_rows; mul0_b = mul0_result;
            end else begin
                // Phase 1: tile1×tile2, coord2×stride1
                mul0_a = lat_tile1; mul0_b = lat_tile2;
                mul1_a = lat_coord2; mul1_b = lat_stride1;
            end
        end
        4'd4: begin
            if (r_rank == 3) begin
                // Final phase: partial_rows × row_len_bytes
                mul0_a = mul0_result; mul0_b = r_row_len_bytes;
            end else if (r_rank >= 4) begin
                // Phase 2: partial_rows×tile3, coord3×stride2
                mul0_a = mul0_result; mul0_b = lat_tile3;
                mul1_a = lat_coord3;  mul1_b = lat_stride2;
            end
        end
        4'd6: begin
            if (r_rank == 4) begin
                // Final phase: partial_rows2 × row_len_bytes
                mul0_a = mul0_result; mul0_b = r_row_len_bytes;
            end else if (r_rank >= 5) begin
                // Phase 3: partial_rows2×tile4, coord4×stride3
                mul0_a = mul0_result; mul0_b = lat_tile4;
                mul1_a = lat_coord4;  mul1_b = lat_stride3;
            end
        end
        4'd8: begin
            // Final phase for rank 5: total_rows × row_len_bytes
            mul0_a = mul0_result; mul0_b = r_row_len_bytes;
        end
        default: ;
        endcase
    end

    // ── Done counter: rank-dependent ──
    // rank 1-2: done at ctr=5; rank R≥3: done at ctr = 2*R+1
    wire [3:0] done_ctr = (r_rank <= 2) ? 4'd5 : 4'(2 * r_rank[2:0] + 1);

    // ════════════════════════════════════════════════════════════════════
    // State machine + pipeline counter
    // ════════════════════════════════════════════════════════════════════

    reg setup_done_r;

    always @(posedge clk) begin
        if (reset) begin
            setup_done_r <= 1'b0;
        end else begin
            setup_done_r <= (state_r == TS_SETUP) && (ctr_r == done_ctr - 4'd1);
        end
    end

    assign pipeline_start = setup_done_r;

    // ── OOB limit helpers (subtraction, no DSP needed) ──
    function automatic [31:0] oob_sub(input [31:0] sz, input [31:0] coord);
        oob_sub = (sz > coord) ? (sz - coord) : 32'd0;
    endfunction

    always @(posedge clk) begin
        if (reset) begin
            state_r            <= TS_IDLE;
            ctr_r              <= '0;
            r_core_id          <= '0;
            r_uuid             <= '0;
            r_wid              <= '0;
            r_bar_addr         <= '0;
            r_notify_smem_done <= 1'b0;
        end else begin
            case (state_r)
            TS_IDLE: begin
                if (launch_accept) begin
                    state_r <= TS_SETUP;
                    ctr_r   <= '0;
                    // Latch metadata.
                    r_core_id <= req_data.core_id;
                    r_uuid    <= req_data.uuid;
                    r_wid     <= req_data.wid;
                    r_bar_addr <= launch_bar_addr;
                `ifdef EXT_DXA_ENABLE
                    r_notify_smem_done <= 1'b1;
                `else
                    r_notify_smem_done <= 1'b0;
                `endif
                    // Multicast: active when cta_mask has >1 bit set.
                    r_is_multicast <= ($countones(req_data.cta_mask) > 1);
                    r_cta_mask     <= req_data.cta_mask;
                    r_smem_stride  <= desc_data.smem_stride;
                    // Latch multiply operands for all dims.
                    lat_tile0      <= dec_tile0;
                    lat_elem_bytes <= dec_elem_bytes;
                    lat_coord0     <= req_data.coords[0][31:0];
                    lat_coord1     <= (dec_rank >= 2) ? req_data.coords[1][31:0] : 32'd0;
                    lat_stride0    <= dec_stride0;
                    lat_tile1      <= dec_tile1;
                    lat_tile2      <= dec_tile2;
                    lat_tile3      <= dec_tile3;
                    lat_tile4      <= dec_tile4;
                    lat_coord2     <= (dec_rank >= 3) ? req_data.coords[2][31:0] : 32'd0;
                    lat_coord3     <= (dec_rank >= 4) ? req_data.coords[3][31:0] : 32'd0;
                    lat_coord4     <= (dec_rank >= 5) ? req_data.coords[4][31:0] : 32'd0;
                    lat_stride1    <= dec_stride1;
                    lat_stride2    <= dec_stride2;
                    lat_stride3    <= dec_stride3;
                    lat_gbase      <= desc_data.base_addr;
                    // Direct latches (no multiply needed).
                    r_initial_smem_base <= req_data.smem_addr;
                    r_cfill        <= desc_data.cfill;
                    r_elem_bytes   <= dec_elem_bytes;
                    r_rank         <= dec_rank;
                    // Strides (pass-through to addr_gen).
                    r_strides[0]   <= dec_stride0;
                    r_strides[1]   <= dec_stride1;
                    r_strides[2]   <= dec_stride2;
                    r_strides[3]   <= dec_stride3;
                    // Dim tile limits.
                    r_dim_tiles[0] <= dec_tile1;
                    r_dim_tiles[1] <= dec_tile2;
                    r_dim_tiles[2] <= dec_tile3;
                    r_dim_tiles[3] <= dec_tile4;
                    // For rank 1-2: total_rows is just tile1 (1 for rank 1).
                    r_total_rows   <= dec_tile1;
                    // OOB limits per outer dim (subtraction only).
                    r_oob_limit[0] <= oob_sub(dec_size1, (dec_rank >= 2) ? req_data.coords[1][31:0] : 32'd0);
                    r_oob_limit[1] <= oob_sub(dec_size2, (dec_rank >= 3) ? req_data.coords[2][31:0] : 32'd0);
                    r_oob_limit[2] <= oob_sub(dec_size3, (dec_rank >= 4) ? req_data.coords[3][31:0] : 32'd0);
                    r_oob_limit[3] <= oob_sub(dec_size4, (dec_rank >= 5) ? req_data.coords[4][31:0] : 32'd0);
                end
            end
            TS_SETUP: begin
                ctr_r <= ctr_r + 4'd1;

                // ── Phase 0 capture at ctr=2 ──
                if (ctr_r == 4'd2) begin
                    r_row_len_bytes     <= mul0_result;
                    r_initial_gmem_base <= lat_gbase
                                         + `MEM_ADDR_WIDTH'(mul1_result)
                                         + `MEM_ADDR_WIDTH'(mul2_result);
                end

                // ── Phase 1 capture at ctr=4 (rank≥3) ──
                if (ctr_r == 4'd4 && r_rank >= 3) begin
                    r_total_rows <= mul0_result; // tile1×tile2 (final for rank 3)
                    r_initial_gmem_base <= r_initial_gmem_base
                                         + `MEM_ADDR_WIDTH'(mul1_result);
                end

                // ── Phase 2 capture at ctr=6 (rank≥4) ──
                if (ctr_r == 4'd6 && r_rank >= 4) begin
                    r_total_rows <= mul0_result; // partial_rows×tile3
                    r_initial_gmem_base <= r_initial_gmem_base
                                         + `MEM_ADDR_WIDTH'(mul1_result);
                end

                // ── Phase 3 capture at ctr=8 (rank=5) ──
                if (ctr_r == 4'd8 && r_rank >= 5) begin
                    r_total_rows <= mul0_result; // total_rows final
                    r_initial_gmem_base <= r_initial_gmem_base
                                         + `MEM_ADDR_WIDTH'(mul1_result);
                end

                // ── Total-bytes capture (2 cycles after final phase load) ──
                // rank 1-2: final loaded at ctr=2, captured at ctr=4
                // rank 3: final loaded at ctr=4, captured at ctr=6
                // rank 4: final loaded at ctr=6, captured at ctr=8
                // rank 5: final loaded at ctr=8, captured at ctr=10
                if (ctr_r == done_ctr - 4'd1) begin
                    r_total_bytes       <= mul0_result;
                    r_total_smem_writes <= (mul0_result
                        + 32'(r_initial_smem_base[SMEM_OFF_BITS-1:0])
                        + SMEM_BYTES - 1) >> SMEM_OFF_BITS;
                end

                // Transition to ACTIVE.
                if (ctr_r == done_ctr) begin
                    state_r <= TS_ACTIVE;
                    ctr_r   <= '0;
                end
            end
            TS_ACTIVE: begin
                if (transfer_done) begin
                    state_r <= TS_IDLE;
                end
            end
            default: state_r <= TS_IDLE;
            endcase
        end
    end

    // ════════════════════════════════════════════════════════════════════
    // Output assignments
    // ════════════════════════════════════════════════════════════════════
    assign setup_params.initial_gmem_base  = r_initial_gmem_base;
    assign setup_params.initial_smem_base  = r_initial_smem_base;
    assign setup_params.row_len_bytes      = r_row_len_bytes;
    assign setup_params.strides            = r_strides;
    assign setup_params.dim_tiles          = r_dim_tiles;
    assign setup_params.oob_limit          = r_oob_limit;
    assign setup_params.total_rows         = r_total_rows;
    assign setup_params.total_smem_writes  = r_total_smem_writes;
    assign setup_params.total_bytes        = r_total_bytes;
    assign setup_params.cfill              = r_cfill;
    assign setup_params.elem_bytes         = r_elem_bytes;
    assign setup_params.rank               = r_rank;

    `UNUSED_VAR (desc_data.size0)
    `UNUSED_VAR (req_data.meta)

endmodule
