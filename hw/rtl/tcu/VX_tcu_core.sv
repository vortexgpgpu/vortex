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

module VX_tcu_core import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    input wire          clk,
    input wire          reset,

`ifdef TCU_WGMMA_ENABLE
    input wire [TCU_BLOCK_CAP-1:0][`XLEN-1:0] tbuf_rs1_data,
    input wire [TCU_WG_RS2_WIDTH-1:0][`XLEN-1:0] tbuf_rs2_data,
`ifdef TCU_SPARSE_ENABLE
    input wire [TCU_MAX_META_BLOCK_WIDTH-1:0] tbuf_sp_meta,
`endif
    input wire          tbuf_ready,
`endif

    // Inputs
    VX_execute_if.slave execute_if,

    // Outputs
    VX_result_if.master result_if
);
    `UNUSED_SPARAM (INSTANCE_ID);

`ifdef TCU_TYPE_DSP
    localparam FCVT_LATENCY = 1;
    localparam FMUL_LATENCY = 8;
    localparam FADD_LATENCY = 11;
    localparam FACC_LATENCY = $clog2(2 * TCU_TC_K + 1) * FADD_LATENCY;
    localparam FEDP_LATENCY = FCVT_LATENCY + FMUL_LATENCY + FACC_LATENCY;
`elsif TCU_TYPE_BHF
    localparam FMUL_LATENCY = 2;
    localparam FADD_LATENCY = 2;
    localparam FRND_LATENCY = 1;
    localparam FACC_LATENCY  = $clog2(2 * TCU_TC_K + 1) * (FADD_LATENCY + FRND_LATENCY);
    localparam FEDP_LATENCY = (FMUL_LATENCY + FRND_LATENCY) + 1 + FACC_LATENCY;
`elsif TCU_TYPE_DPI
    localparam FMUL_LATENCY = 2;
    localparam FACC_LATENCY = 2;
    localparam FEDP_LATENCY = FMUL_LATENCY + FACC_LATENCY;
`else // TCU_TYPE_TFR
    localparam FMUL_LATENCY = 1;
    localparam FALN_LATENCY = 1;
    localparam FACC_LATENCY = 1;
    localparam FRND_LATENCY = 1;
    localparam FEDP_LATENCY = FMUL_LATENCY + FALN_LATENCY + FACC_LATENCY + FRND_LATENCY;
`endif

    localparam PIPE_LATENCY = FEDP_LATENCY + 1;
    localparam MDATA_QUEUE_DEPTH = 1 << $clog2(PIPE_LATENCY);

    localparam LG_A_BS    = $clog2(TCU_A_BLOCK_SIZE);
    localparam LG_B_BS    = $clog2(TCU_B_BLOCK_SIZE);
    localparam OFF_W      = $clog2(TCU_BLOCK_CAP);

`ifdef TCU_SPARSE_ENABLE
    localparam LG_B_BS_SP = $clog2(TCU_B_BLOCK_SIZE_SP);
    wire is_sparse = execute_if.data.op_args.tcu.is_sparse;
    wire is_meta_store = (execute_if.data.op_type == INST_TCU_META_STORE);
`endif

    // -----------------------------------------------------------------------
    // WGMMA / WMMA abstraction layer
    // -----------------------------------------------------------------------
    // All WGMMA-vs-WMMA runtime differences are resolved here behind a
    // common interface.  Downstream code uses only these wires and never
    // references tbuf_* or is_wgmma directly.

    wire [TCU_BLOCK_CAP-1:0][`XLEN-1:0] rs1_data;
`ifdef TCU_WGMMA_ENABLE
    wire [TCU_WG_RS2_WIDTH-1:0][`XLEN-1:0] rs2_data;
`else
    wire [TCU_BLOCK_CAP-1:0][`XLEN-1:0] rs2_data;
`endif
    wire exe_ready_extra;              // additional ready gating (tbuf_ready)

`ifdef TCU_ACC_ENABLE
    wire [3:0] last_k_steps;          // final step_k value for is_last_k

    // WMMA K-step limit (needed before the mux region for last_k_steps)
  `ifdef TCU_SPARSE_ENABLE
    wire [3:0] k_steps_val = is_sparse ? 4'((TCU_K_STEPS / 2) - 1) : 4'(TCU_K_STEPS - 1);
  `else
    wire [3:0] k_steps_val = 4'(TCU_K_STEPS - 1);
  `endif
`endif

`ifdef TCU_WGMMA_ENABLE
    wire is_wgmma = (execute_if.data.op_type == INST_TCU_WGMMA);
    wire wg_a_smem = execute_if.data.op_args.tcu.a_from_smem;

    // A/B operand mux: tile buffer (smem) or register file
    assign rs1_data = (is_wgmma && wg_a_smem) ? tbuf_rs1_data : execute_if.data.rs1_data;
    /* verilator lint_off WIDTHEXPAND */
    assign rs2_data = is_wgmma ? tbuf_rs2_data
                               : TCU_WG_RS2_WIDTH'(execute_if.data.rs2_data);
    /* verilator lint_on WIDTHEXPAND */

  `ifdef TCU_SPARSE_ENABLE
    // Sparse metadata mux: tile-buffer vs register-file metadata
    wire [TCU_MAX_META_BLOCK_WIDTH-1:0] vld_meta_block = is_wgmma ? tbuf_sp_meta : wmma_sp_meta;
  `endif

  `ifdef TCU_ACC_ENABLE
    // K-step limit: WGMMA and WMMA have different tile-K sizes
    `ifdef TCU_SPARSE_ENABLE
    wire [3:0] wg_k_steps_val = execute_if.data.op_args.tcu.is_sparse
        ? 4'((TCU_WG_K_STEPS / 2) - 1) : 4'(TCU_WG_K_STEPS - 1);
    `else
    wire [3:0] wg_k_steps_val = 4'(TCU_WG_K_STEPS - 1);
    `endif
    assign last_k_steps   = is_wgmma ? wg_k_steps_val : k_steps_val;
  `endif
    assign exe_ready_extra = ~is_wgmma || tbuf_ready;
`else
    assign rs1_data = execute_if.data.rs1_data;
    assign rs2_data = execute_if.data.rs2_data;
  `ifdef TCU_SPARSE_ENABLE
    wire [TCU_MAX_META_BLOCK_WIDTH-1:0] vld_meta_block = wmma_sp_meta;
  `endif
  `ifdef TCU_ACC_ENABLE
    assign last_k_steps   = k_steps_val;
  `endif
    assign exe_ready_extra = 1'b1;
`endif

    wire [3:0] step_m = execute_if.data.op_args.tcu.step_m;
    wire [3:0] step_n = execute_if.data.op_args.tcu.step_n;
    wire [3:0] step_k = execute_if.data.op_args.tcu.step_k;

    wire [3:0] fmt_s = execute_if.data.op_args.tcu.fmt_s;
    wire [3:0] fmt_d = execute_if.data.op_args.tcu.fmt_d;

    wire execute_fire = execute_if.valid && execute_if.ready;

    // -----------------------------------------------------------------------
    // Sparse metadata: VX_tcu_meta (for WMMA_SP) + optional tile-buffer mux
    // -----------------------------------------------------------------------

`ifdef TCU_SPARSE_ENABLE
    wire meta_wr_en = execute_fire && is_meta_store;
`endif

    // Modify header for non-writeback uops:
    //   meta_store: force rd=0
    //   non-last-k: force wb=0 (intermediate accumulator result, not RF write)
    tcu_header_t mdata_queue_in;
    always_comb begin
        mdata_queue_in = execute_if.data.header;
    `ifdef TCU_SPARSE_ENABLE
        if (is_meta_store) begin
            mdata_queue_in.rd = '0;
        end
    `endif
    `ifdef TCU_ACC_ENABLE
        if (!is_last_k) begin
            mdata_queue_in.wb = 1'b0;
        end
    `endif
    end

    `UNUSED_VAR ({step_m, step_n, step_k, fmt_s, fmt_d, execute_if.data});

`ifdef TCU_ACC_ENABLE
    // -----------------------------------------------------------------------
    // Internal K accumulation control (Nvidia-style)
    // -----------------------------------------------------------------------
    // k=0 (first_k): c_val from RF (rs3_data), no writeback
    // 0<k<K-1: c_val from internal accumulator, no writeback
    // k=K-1 (last_k): c_val from internal accumulator, writeback to RF
    // Accumulator is a multi-tile LUTRAM indexed by {wid, step_m, step_n}.

    wire is_first_k = (step_k == '0);
    wire is_last_k  = (step_k == last_k_steps);

    // -----------------------------------------------------------------------
    // Multi-tile accumulator (LUTRAM, async read, read-first)
    // -----------------------------------------------------------------------
    // One LUTRAM per FEDP element (tcM × tcN instances).
    // Width = 32 bits.
    // TCU_WLOCK_ENABLE=on:  shared accumulator, depth = MAX_TILES
    // TCU_WLOCK_ENABLE=off: per-warp accumulator, depth = NUM_WARPS * MAX_TILES

  `ifdef TCU_WGMMA_ENABLE
    localparam ACCUM_MAX_N = (TCU_N_STEPS > TCU_WG_N_STEPS) ? TCU_N_STEPS : TCU_WG_N_STEPS;
    localparam ACCUM_MAX_M = (TCU_M_STEPS > TCU_WG_M_STEPS) ? TCU_M_STEPS : TCU_WG_M_STEPS;
  `else
    localparam ACCUM_MAX_N = TCU_N_STEPS;
    localparam ACCUM_MAX_M = TCU_M_STEPS;
  `endif
    localparam ACCUM_MAX_TILES = ACCUM_MAX_M * ACCUM_MAX_N;
    localparam ACCUM_TILE_W   = $clog2(ACCUM_MAX_TILES);
  `ifdef TCU_WLOCK_ENABLE
    localparam ACCUM_DEPTH    = ACCUM_MAX_TILES;
  `else
    localparam ACCUM_DEPTH    = `NUM_WARPS * ACCUM_MAX_TILES;
  `endif
    localparam ACCUM_ADDRW    = $clog2(ACCUM_DEPTH);

    wire [NW_WIDTH-1:0] exe_wid = execute_if.data.header.wid;

    // Tile index: step_m * ACCUM_MAX_N + step_n
    wire [ACCUM_TILE_W-1:0] exe_tile_idx = ACCUM_TILE_W'(step_m) * ACCUM_TILE_W'(ACCUM_MAX_N)
                                          + ACCUM_TILE_W'(step_n);
    // Read address for accumulator lookup
  `ifdef TCU_WLOCK_ENABLE
    wire [ACCUM_ADDRW-1:0] accum_raddr = ACCUM_ADDRW'(exe_tile_idx);
  `else
    wire [ACCUM_ADDRW-1:0] accum_raddr = {exe_wid, exe_tile_idx};
  `endif

    // Track is_last_k, warp ID, and tile index through the FEDP pipeline
    reg [PIPE_LATENCY-1:0] last_k_pipe;
    reg [PIPE_LATENCY-1:0][NW_WIDTH-1:0] wid_pipe;
    reg [PIPE_LATENCY-1:0][ACCUM_TILE_W-1:0] tile_pipe;

    // Per-warp inflight counter: tracks non-last-k uops in the FEDP pipeline.
    // k>0 uops stall until ALL previous k-round results are latched.
    localparam INFLIGHT_W = $clog2(ACCUM_MAX_TILES + 1);
    reg [`NUM_WARPS-1:0][INFLIGHT_W-1:0] inflight_k;
    wire k_stall = !is_first_k && (inflight_k[exe_wid] != '0);

  `ifdef SIMULATION
    `ifdef TCU_WLOCK_ENABLE
    // Detect illegal warp interleaving: a different warp must not enter
    // the TCU while another warp has non-last_k uops still in the FEDP pipeline.
    wire [`NUM_WARPS-1:0] inflight_mask;
    for (genvar w = 0; w < `NUM_WARPS; ++w) begin : g_inflight_mask
        assign inflight_mask[w] = (inflight_k[w] != '0);
    end
    wire [`NUM_WARPS-1:0] other_inflight = inflight_mask & ~(`NUM_WARPS'(1) << exe_wid);

    `RUNTIME_ASSERT(~execute_fire || (other_inflight == '0),
        ("%s warp interleave during K-accumulation! wid=%0d entering while inflight_mask=%b (#%0d)",
            INSTANCE_ID, exe_wid, inflight_mask, execute_if.data.header.uuid))
    `endif
  `endif
`else
    wire k_stall = 1'b0;
`endif

    // -----------------------------------------------------------------------
    // Pipeline control
    // -----------------------------------------------------------------------

    wire mdata_queue_full;

    wire result_fire = result_if.valid && result_if.ready;
    wire fedp_enable, fedp_done;

`ifdef TCU_ACC_ENABLE
    wire fedp_done_raw;           // any FEDP result (including intermediate k)

    // Write-back address from pipeline exit
  `ifdef TCU_WLOCK_ENABLE
    wire [ACCUM_ADDRW-1:0] accum_waddr = ACCUM_ADDRW'(tile_pipe[0]);
  `else
    wire [ACCUM_ADDRW-1:0] accum_waddr = {wid_pipe[0], tile_pipe[0]};
  `endif
    wire accum_wr = fedp_done_raw && !last_k_pipe[0] && fedp_enable;
`endif

    reg [PIPE_LATENCY-1:0] fedp_delay_pipe;
    always @(posedge clk) begin
        if (reset) begin
            fedp_delay_pipe <= '0;
        `ifdef TCU_ACC_ENABLE
            last_k_pipe     <= '0;
            for (int w = 0; w < `NUM_WARPS; ++w)
                inflight_k[w] <= '0;
        `endif
        end else begin
            if (fedp_enable) begin
                fedp_delay_pipe <= fedp_delay_pipe >> 1;
            `ifdef TCU_ACC_ENABLE
                last_k_pipe     <= last_k_pipe >> 1;
                for (int s = PIPE_LATENCY-1; s > 0; --s) begin
                    wid_pipe[s-1]  <= wid_pipe[s];
                    tile_pipe[s-1] <= tile_pipe[s];
                end
            `endif
            end
            if (execute_fire) begin
                fedp_delay_pipe[PIPE_LATENCY-1] <= 1;
            `ifdef TCU_ACC_ENABLE
                last_k_pipe[PIPE_LATENCY-1]     <= is_last_k;
                wid_pipe[PIPE_LATENCY-1]         <= exe_wid;
                tile_pipe[PIPE_LATENCY-1]        <= exe_tile_idx;
            `endif
            end
        `ifdef TCU_ACC_ENABLE
            // Per-warp inflight tracking for k-stall:
            //   increment when a non-last-k uop enters the pipeline
            //   decrement when a non-last-k result exits and is latched
            for (int w = 0; w < `NUM_WARPS; ++w) begin
                if ((execute_fire && !is_last_k && NW_WIDTH'(w) == exe_wid)
                 && (accum_wr && NW_WIDTH'(w) == wid_pipe[0])) begin
                    // simultaneous inc+dec: no change
                end else if (execute_fire && !is_last_k && NW_WIDTH'(w) == exe_wid) begin
                    inflight_k[w] <= inflight_k[w] + INFLIGHT_W'(1);
                end else if (accum_wr && NW_WIDTH'(w) == wid_pipe[0]) begin
                    inflight_k[w] <= inflight_k[w] - INFLIGHT_W'(1);
                end
            end
        `endif
        end
    end
`ifdef TCU_ACC_ENABLE
    assign fedp_done_raw    = fedp_delay_pipe[0];
    assign fedp_done        = fedp_done_raw; // all uops produce downstream result (wb=0 skips RF write)
`else
    assign fedp_done        = fedp_delay_pipe[0];
`endif

    assign result_if.valid  = fedp_done;
    assign fedp_enable      = ~fedp_done || result_if.ready;
    assign execute_if.ready = ~mdata_queue_full && fedp_enable && !k_stall && exe_ready_extra;

    // All uops push to the metadata queue; non-last-k have wb=0 in their
    // header so the writeback stage skips the RF write.
    wire mdata_push = execute_fire;

    VX_fifo_queue #(
        .DATAW ($bits(tcu_header_t)),
        .DEPTH (MDATA_QUEUE_DEPTH),
        .OUT_REG (1)
    ) mdata_queue (
        .clk    (clk),
        .reset  (reset),
        .push   (mdata_push),
        .pop    (result_fire),
        .data_in(mdata_queue_in),
        .data_out(result_if.data.header),
        `UNUSED_PIN(empty),
        `UNUSED_PIN(alm_empty),
        .full   (mdata_queue_full),
        `UNUSED_PIN(alm_full),
        `UNUSED_PIN(size)
    );

    // -----------------------------------------------------------------------
    // Operand offset computation
    // -----------------------------------------------------------------------

    wire [OFF_W-1:0] a_off = (OFF_W'(step_m) & OFF_W'(TCU_A_SUB_BLOCKS-1)) << LG_A_BS;
`ifdef TCU_SPARSE_ENABLE
    wire [OFF_W-1:0] b_off = is_sparse
        ? (OFF_W'(step_n) & OFF_W'(TCU_B_SUB_BLOCKS_SP-1)) << LG_B_BS_SP
        : (OFF_W'(step_n) & OFF_W'(TCU_B_SUB_BLOCKS-1)) << LG_B_BS;
`else
    wire [OFF_W-1:0] b_off = (OFF_W'(step_n) & OFF_W'(TCU_B_SUB_BLOCKS-1)) << LG_B_BS;
`endif

    // -----------------------------------------------------------------------
    // Unified sparse metadata
    // -----------------------------------------------------------------------
    // WMMA_SP:  from VX_tcu_meta (per-warp register-file metadata store)
    // WGMMA_SP: from VX_tcu_tbuf (pre-extracted from SMEM metadata)
    // Both produce TCU_TC_M per-row slices of TCU_MAX_META_ROW_WIDTH bits,
    // indexed by (step_m, step_k).

`ifdef TCU_SPARSE_ENABLE
    wire [TCU_MAX_META_BLOCK_WIDTH-1:0] wmma_sp_meta;
    VX_tcu_meta #(
        .INSTANCE_ID (INSTANCE_ID)
    ) tcu_meta (
        .clk    (clk),
        .reset  (reset),
        .wr_en  (meta_wr_en),
        .wid    (execute_if.data.header.wid),
        .wr_idx (fmt_d),
        .wr_data(rs1_data),
        .step_m (step_m),
        .step_k (step_k),
        .vld_block(wmma_sp_meta)
    );

    // vld_meta_block is muxed in the operand mux region above
`endif

    // -----------------------------------------------------------------------
    // FEDP grid: TCU_TC_M × TCU_TC_N compute elements
    // -----------------------------------------------------------------------

    wire [TCU_TC_M-1:0][TCU_TC_N-1:0][31:0] d_val;

    for (genvar i = 0; i < TCU_TC_M; ++i) begin : g_i
        for (genvar j = 0; j < TCU_TC_N; ++j) begin : g_j
        `ifdef TCU_SPARSE_ENABLE
            wire [TCU_TC_K-1:0][31:0] a_row, b_col, b_col_dense, b_col_sparse, b_col_1, b_col_2;
        `else
            wire [TCU_TC_K-1:0][31:0] a_row, b_col;
        `endif
            for (genvar k_idx = 0; k_idx < TCU_TC_K; ++k_idx) begin : g_slice_assign
                assign a_row[k_idx] = 32'(rs1_data[a_off + i * TCU_TC_K + k_idx]);
            `ifdef TCU_SPARSE_ENABLE
                assign b_col_dense[k_idx] = 32'(rs2_data[b_off + j * TCU_TC_K + k_idx]);
                localparam J_SP = SYM_SPARSE ? (j % (TCU_TC_N / 2)) : j;
                assign b_col_1[k_idx] = 32'(rs2_data[b_off + J_SP * TCU_TC_K * 2 + k_idx * 2]);
                assign b_col_2[k_idx] = 32'(rs2_data[b_off + J_SP * TCU_TC_K * 2 + k_idx * 2 + 1]);
            `else
                assign b_col[k_idx] = 32'(rs2_data[b_off + j * TCU_TC_K + k_idx]);
            `endif
            end

        `ifdef TCU_ACC_ENABLE
            // Per-element accumulator LUTRAM (async read, read-first)
            wire [31:0] accum_rdata;

            VX_dp_ram #(
                .DATAW     (32),
                .SIZE      (ACCUM_DEPTH),
                .LUTRAM    (1),
                .OUT_REG   (0),
                .RDW_MODE  ("R"),
                .RADDR_REG (1)
            ) accum_ram (
                .clk   (clk),
                .reset (reset),
                .read  (1'b1),
                .write (accum_wr),
                .wren  (1'b1),
                .waddr (accum_waddr),
                .wdata (d_val[i][j]),
                .raddr (accum_raddr),
                .rdata (accum_rdata)
            );

            // k=0: C from register file; k>0: C from internal accumulator LUTRAM
            wire [31:0] c_val = is_first_k
                ? 32'(execute_if.data.rs3_data[i * TCU_TC_N + j])
                : accum_rdata;
        `else
            // No internal accumulator: C always from register file
            wire [31:0] c_val = 32'(execute_if.data.rs3_data[i * TCU_TC_N + j]);
        `endif

        `ifdef TCU_SPARSE_ENABLE
            VX_tcu_sp_mux #(
                .INSTANCE_ID (INSTANCE_ID),
                .ROW_IDX     (i)
            ) tcu_sp_mux (
                .fmt_s     (fmt_s),
                .b_col_in1 (b_col_1),
                .b_col_in2 (b_col_2),
                .vld_mask  (vld_meta_block),
                .b_col_out (b_col_sparse)
            );
            assign b_col = is_sparse ? b_col_sparse : b_col_dense;
        `endif

            wire [3:0] fmt_s_r, fmt_d_r;
            wire [TCU_TC_K-1:0][31:0] a_row_r, b_col_r;
            wire [31:0] c_val_r;

            `BUFFER_EX (
                {c_val_r, fmt_s_r, fmt_d_r, b_col_r, a_row_r},
                {c_val,   fmt_s,   fmt_d,   b_col,   a_row},
                fedp_enable,
                0, // resetw
                1  // depth
            );

        `ifdef TCU_TYPE_DPI
            VX_tcu_fedp_dpi #(
                .INSTANCE_ID (INSTANCE_ID),
                .LATENCY (FEDP_LATENCY),
                .N (TCU_TC_K)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .enable(fedp_enable),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row(a_row_r),
                .b_col(b_col_r),
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `elsif TCU_TYPE_BHF
            VX_tcu_fedp_bhf #(
                .INSTANCE_ID (INSTANCE_ID),
                .LATENCY (FEDP_LATENCY),
                .N (TCU_TC_K)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .enable(fedp_enable),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row(a_row_r),
                .b_col(b_col_r),
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `elsif TCU_TYPE_TFR
            VX_tcu_fedp_tfr #(
                .INSTANCE_ID (INSTANCE_ID),
                .LATENCY (FEDP_LATENCY),
                .N (TCU_TC_K)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .vld_mask('1),
                .enable(fedp_enable),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row (a_row_r),
                .b_col (b_col_r),
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `elsif TCU_TYPE_DSP
            VX_tcu_fedp_dsp #(
                .INSTANCE_ID (INSTANCE_ID),
                .LATENCY (FEDP_LATENCY),
                .N (TCU_TC_K)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .enable(fedp_enable),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row(a_row_r),
                .b_col(b_col_r),
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `endif

            // NaN-box the fp32 result for XLEN=64: upper 32 bits must be all-1s per RVF spec.
            if (`XLEN > 32) begin : g_result_nanbox
                assign result_if.data.data[i * TCU_TC_N + j] = {32'hffffffff, d_val[i][j]};
            end else begin : g_result_passthrough
                assign result_if.data.data[i * TCU_TC_N + j] = d_val[i][j];
            end

        `ifdef DBG_TRACE_TCU
            always @(posedge clk) begin
                if (execute_if.valid && execute_if.ready) begin
                    `TRACE(3, ("%t: %s FEDP-enq: wid=%0d, i=%0d, j=%0d, m=%0d, n=%0d, a_row=", $time, INSTANCE_ID, execute_if.data.header.wid, i, j, step_m, step_n))
                    `TRACE_ARRAY1D(2, "0x%0h", a_row, TCU_TC_K)
                    `TRACE(3, (", b_col="));
                    `TRACE_ARRAY1D(2, "0x%0h", b_col, TCU_TC_K)
                    `TRACE(3, (", c_val=0x%0h (#%0d)\n", c_val, execute_if.data.header.uuid));
                end
                if (result_if.valid && result_if.ready) begin
                    `TRACE(3, ("%t: %s FEDP-deq: wid=%0d, i=%0d, j=%0d, d_val=0x%0h (#%0d)\n", $time, INSTANCE_ID, result_if.data.header.wid, i, j, d_val[i][j], result_if.data.header.uuid));
                end
            end
        `endif // DBG_TRACE_TCU
        end
    end

endmodule
