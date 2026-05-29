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
    input wire [TCU_BLOCK_CAP-1:0][`XLEN-1:0] tbuf_c_data,  // C tile (lmem-accumulator mode)
`ifdef TCU_SPARSE_ENABLE
    input wire [TCU_MAX_META_BLOCK_WIDTH-1:0] tbuf_sp_meta,
`endif
    input wire          tbuf_ready,   // full tile ready (used for sparse + perf)
    input wire          b_ready,      // B tile fully fetched
    input wire [TCU_WG_M_STEPS-1:0] a_row_ready, // per-m-step A readiness
    input wire          c_ready,      // C tile fetched from lmem (lmem-accumulator mode)

    // C LUTRAM write-back: FEDP result → C LUTRAM in VX_tcu_tbuf
    output wire                           c_wb_valid,
    output wire [TCU_WG_C_TOTAL-1:0]     c_wb_wren,
    output wire [TCU_WG_C_TOTAL-1:0][31:0] c_wb_data,
    // Pulse when the last (m,n,k=K-1) FEDP output lands → triggers STORE_D
    output wire                           c_all_done,
`endif

`ifdef PERF_ENABLE
`ifdef TCU_WGMMA_ENABLE
    output wire [PERF_CTR_BITS-1:0] wgmma_stalls_mdata,
    output wire [PERF_CTR_BITS-1:0] wgmma_stalls_pipe,
    output wire [PERF_CTR_BITS-1:0] compute_cycles,
`endif
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
    localparam FMUL_LATENCY = (`LATENCY_TCU - 3);
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
    wire exe_ready_extra; // additional ready gating (tbuf_ready)

`ifdef TCU_WGMMA_ENABLE
    wire is_wgmma       = (execute_if.data.op_type == INST_TCU_WGMMA);
    wire is_prefetch_b  = (execute_if.data.op_type == INST_TCU_WGMMA_PREFETCH_B);
    wire wg_a_smem    = execute_if.data.op_args.tcu.a_from_smem;
    wire cd_from_lmem = execute_if.data.op_args.tcu.cd_from_lmem;

    // A/B operand mux: tile buffer (smem) or register file
    assign rs1_data = (is_wgmma && wg_a_smem) ? tbuf_rs1_data : execute_if.data.rs1_data;
    /* verilator lint_off WIDTHEXPAND */
    assign rs2_data = is_wgmma ? tbuf_rs2_data : TCU_WG_RS2_WIDTH'(execute_if.data.rs2_data);
    /* verilator lint_on WIDTHEXPAND */

  `ifdef TCU_SPARSE_ENABLE
    // Sparse metadata mux: tile-buffer (SS mode) vs VX_tcu_meta SRAM (WMMA / RS mode)
    wire [TCU_MAX_META_BLOCK_WIDTH-1:0] vld_meta_block = (is_wgmma && wg_a_smem) ? tbuf_sp_meta : wmma_sp_meta;
  `endif

    // Dense SS: fire each uop once B is ready AND this uop's A row has arrived.
    // Sparse SS: wait for full tile (includes metadata) via tbuf_ready.
    // RS (A from regs): only need B ready.
    // lmem-accum: additionally stall if k>0 and C LUTRAM has an in-flight write.
    localparam M_IDX_W = $clog2(TCU_WG_M_STEPS);
    localparam N_IDX_W = $clog2(TCU_WG_N_STEPS);
    // c_dirty[m][n] = FEDP in-flight for this C LUTRAM slot (lmem-accum only)
    reg [TCU_WG_M_STEPS-1:0][TCU_WG_N_STEPS-1:0] c_dirty;
    wire c_lutram_stall = is_wgmma && cd_from_lmem
                       && (step_k != '0)
                       && c_dirty[M_IDX_W'(step_m)][N_IDX_W'(step_n)];
  `ifdef TCU_SPARSE_ENABLE
    assign exe_ready_extra = (~is_wgmma
        || (wg_a_smem
            ? (is_sparse ? tbuf_ready : (b_ready && a_row_ready[M_IDX_W'(step_m)]))
            : b_ready))
        && !c_lutram_stall;
  `else
    `UNUSED_VAR (tbuf_ready)
    assign exe_ready_extra = (~is_wgmma
        || (wg_a_smem ? (b_ready && a_row_ready[M_IDX_W'(step_m)]) : b_ready))
        && !c_lutram_stall;
  `endif
`else
    assign rs1_data = execute_if.data.rs1_data;
    assign rs2_data = execute_if.data.rs2_data;
  `ifdef TCU_SPARSE_ENABLE
    wire [TCU_MAX_META_BLOCK_WIDTH-1:0] vld_meta_block = wmma_sp_meta;
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

    // Modify header for meta_store/prefetch_b: force rd=0 (no register writeback)
    tcu_header_t mdata_queue_in;
    always_comb begin
        mdata_queue_in = execute_if.data.header;
    `ifdef TCU_SPARSE_ENABLE
        if (is_meta_store || is_prefetch_b) begin
            mdata_queue_in.rd = '0;
        end
    `else
        if (is_prefetch_b) begin
            mdata_queue_in.rd = '0;
        end
    `endif
    end

    `UNUSED_VAR ({step_m, step_n, step_k, fmt_s, fmt_d, execute_if.data});
    `UNUSED_VAR (c_ready);

    // -----------------------------------------------------------------------
    // Pipeline control
    // -----------------------------------------------------------------------

    wire mdata_queue_full;

    wire result_fire = result_if.valid && result_if.ready;
    wire fedp_enable, fedp_done;

    reg [PIPE_LATENCY-1:0] fedp_delay_pipe;

`ifdef TCU_WGMMA_ENABLE
    // Per-stage metadata for in-flight uops: needed to reconstruct (m,n,k) at
    // FEDP output time for C LUTRAM write-back and c_dirty bookkeeping.
    reg [PIPE_LATENCY-1:0][3:0] sm_pipe, sn_pipe, sk_pipe;
    reg [PIPE_LATENCY-1:0]       cl_pipe; // cd_from_lmem per stage

    always @(posedge clk) begin
        if (reset) begin
            fedp_delay_pipe <= '0;
            sm_pipe <= '0; sn_pipe <= '0; sk_pipe <= '0; cl_pipe <= '0;
        end else begin
            if (fedp_enable) begin
                fedp_delay_pipe             <= fedp_delay_pipe >> 1;
                sm_pipe[PIPE_LATENCY-2:0]   <= sm_pipe[PIPE_LATENCY-1:1];
                sn_pipe[PIPE_LATENCY-2:0]   <= sn_pipe[PIPE_LATENCY-1:1];
                sk_pipe[PIPE_LATENCY-2:0]   <= sk_pipe[PIPE_LATENCY-1:1];
                cl_pipe[PIPE_LATENCY-2:0]   <= cl_pipe[PIPE_LATENCY-1:1];
            end
            if (execute_fire) begin
                fedp_delay_pipe[PIPE_LATENCY-1] <= 1;
                sm_pipe[PIPE_LATENCY-1]         <= step_m;
                sn_pipe[PIPE_LATENCY-1]         <= step_n;
                sk_pipe[PIPE_LATENCY-1]         <= step_k;
                cl_pipe[PIPE_LATENCY-1]         <= (is_wgmma && cd_from_lmem);
            end
        end
    end

    wire lmem_fedp_done = fedp_done && cl_pipe[0];

    // c_dirty: set when execute_fire for lmem mode, cleared when FEDP output
    // for that (m,n) position writes to C LUTRAM.
    always_ff @(posedge clk) begin
        if (reset) begin
            c_dirty <= '0;
        end else begin
            // Clear on FEDP done (write to C LUTRAM happens this cycle)
            if (lmem_fedp_done)
                c_dirty[sm_pipe[0]][sn_pipe[0]] <= 1'b0;
            // Set when a uop fires for lmem mode (FEDP now in-flight)
            if (execute_fire && is_wgmma && cd_from_lmem)
                c_dirty[M_IDX_W'(step_m)][N_IDX_W'(step_n)] <= 1'b1;
        end
    end

    // c_all_done: fires when the last (m=M-1, n=N-1, k=K-1) FEDP result lands
    assign c_all_done = lmem_fedp_done
        && (sm_pipe[0] == 4'(TCU_WG_M_STEPS - 1))
        && (sn_pipe[0] == 4'(TCU_WG_N_STEPS - 1))
        && (sk_pipe[0] == 4'(TCU_WG_K_STEPS - 1));

`else
    always @(posedge clk) begin
        if (reset) begin
            fedp_delay_pipe <= '0;
        end else begin
            if (fedp_enable)
                fedp_delay_pipe <= fedp_delay_pipe >> 1;
            if (execute_fire)
                fedp_delay_pipe[PIPE_LATENCY-1] <= 1;
        end
    end
`endif

    assign fedp_done = fedp_delay_pipe[0];

`ifdef TCU_WGMMA_ENABLE
    // For lmem mode, suppress result_if and skip mdata_queue; pipeline advances
    // on its own without waiting for result_if.ready.
    assign result_if.valid  = fedp_done && !lmem_fedp_done;
    assign fedp_enable      = ~fedp_done || result_if.ready || lmem_fedp_done;
    wire mdata_block = mdata_queue_full && !(is_wgmma && cd_from_lmem);
    assign execute_if.ready = ~mdata_block && fedp_enable && exe_ready_extra;
    wire mdata_push = execute_fire && !(is_wgmma && cd_from_lmem);
`else
    assign result_if.valid  = fedp_done;
    assign fedp_enable      = ~fedp_done || result_if.ready;
    assign execute_if.ready = ~mdata_queue_full && fedp_enable && exe_ready_extra;
    wire mdata_push = execute_fire;
`endif

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

        `ifdef TCU_WGMMA_ENABLE
            wire [31:0] c_val = (is_wgmma && cd_from_lmem)
                ? 32'(tbuf_c_data[i * TCU_TC_N + j])
                : 32'(execute_if.data.rs3_data[i * TCU_TC_N + j]);
        `else
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

            VX_pipe_register #(
                .DATAW (32 + 4 + 4 + TCU_TC_K * 32 + TCU_TC_K * 32)
            ) pipe_fedp (
                .clk      (clk),
                .reset    (reset),
                .enable   (fedp_enable),
                .data_in  ({c_val,   fmt_s,   fmt_d,   b_col,   a_row}),
                .data_out ({c_val_r, fmt_s_r, fmt_d_r, b_col_r, a_row_r})
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

        `ifdef TCU_WGMMA_ENABLE
            // C LUTRAM write-back: d_val → C tile at (sm_pipe[0], sn_pipe[0])
            // lutram_idx = (step_m * TC_M + i) * TILE_N + step_n * TC_N + j
            localparam TILE_N_LOCAL = TCU_WG_TILE_N;
            wire [$clog2(TCU_WG_C_TOTAL)-1:0] c_wb_idx =
                $clog2(TCU_WG_C_TOTAL)'((sm_pipe[0] * TCU_TC_M + i) * TILE_N_LOCAL
                                        + sn_pipe[0] * TCU_TC_N + j);
            assign c_wb_wren[c_wb_idx] = lmem_fedp_done;
            assign c_wb_data[c_wb_idx] = d_val[i][j];
        `endif

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

`ifdef TCU_WGMMA_ENABLE
    assign c_wb_valid = lmem_fedp_done;
`endif

`ifdef PERF_ENABLE
`ifdef TCU_WGMMA_ENABLE
    reg [PERF_CTR_BITS-1:0] wgmma_stalls_mdata_ctr_r;
    reg [PERF_CTR_BITS-1:0] wgmma_stalls_pipe_ctr_r;
    reg [PERF_CTR_BITS-1:0] compute_cycles_ctr_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            wgmma_stalls_mdata_ctr_r <= '0;
            wgmma_stalls_pipe_ctr_r  <= '0;
            compute_cycles_ctr_r     <= '0;
        end else begin
            if (execute_if.valid && is_wgmma && mdata_queue_full)
                wgmma_stalls_mdata_ctr_r <= wgmma_stalls_mdata_ctr_r + PERF_CTR_BITS'(1);
            if (execute_if.valid && is_wgmma && !fedp_enable)
                wgmma_stalls_pipe_ctr_r  <= wgmma_stalls_pipe_ctr_r  + PERF_CTR_BITS'(1);
            if (fedp_delay_pipe != '0)
                compute_cycles_ctr_r <= compute_cycles_ctr_r + PERF_CTR_BITS'(1);
        end
    end
    assign wgmma_stalls_mdata = wgmma_stalls_mdata_ctr_r;
    assign wgmma_stalls_pipe  = wgmma_stalls_pipe_ctr_r;
    assign compute_cycles     = compute_cycles_ctr_r;
`endif
`endif

endmodule
