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

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    input wire [TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0] tbuf_rs1_data,
    input wire [TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0] tbuf_rs2_data,
    input wire          tbuf_ready,
`endif

    // External metadata write port from the shared VX_tcu_agu.
`ifdef VX_CFG_TCU_META_ENABLE
    input wire                     ext_meta_wr_en,
    input wire [NW_WIDTH-1:0]      ext_meta_wr_wid,
    input wire [4:0]               ext_meta_wr_idx,
    input wire [TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0] ext_meta_wr_data,
`endif

    // Inputs
    VX_execute_if.slave execute_if,

    // Outputs
    VX_result_if.master result_if
);
    `UNUSED_SPARAM (INSTANCE_ID);

`ifdef VX_CFG_TCU_TYPE_DSP
    localparam FCVT_LATENCY = 1;
    localparam FMUL_LATENCY = 8;
    localparam FADD_LATENCY = 11;
    localparam FACC_LATENCY = $clog2(2 * TCU_TC_K + 1) * FADD_LATENCY;
    localparam FEDP_LATENCY = FCVT_LATENCY + FMUL_LATENCY + FACC_LATENCY;
`elsif VX_CFG_TCU_TYPE_BHF
    localparam FMUL_LATENCY = 2;
    localparam FADD_LATENCY = 2;
    localparam FRND_LATENCY = 1;
    localparam FACC_LATENCY  = $clog2(2 * TCU_TC_K + 1) * (FADD_LATENCY + FRND_LATENCY);
    localparam FEDP_LATENCY = (FMUL_LATENCY + FRND_LATENCY) + 1 + FACC_LATENCY;
`elsif VX_CFG_TCU_TYPE_FPNEW
    localparam FMUL_LATENCY = 6;
    localparam FMUX_LATENCY = 1;
    localparam FADD_LATENCY = 7;
    localparam FACC_LATENCY = $clog2(2 * TCU_TC_K) * FADD_LATENCY;
    localparam FEDP_LATENCY = FMUL_LATENCY + FMUX_LATENCY + FACC_LATENCY + FADD_LATENCY;
`elsif VX_CFG_TCU_TYPE_DPI
    localparam FMUL_LATENCY = 2;
    localparam FACC_LATENCY = 2;
    localparam FEDP_LATENCY = FMUL_LATENCY + FACC_LATENCY;
`elsif VX_CFG_TCU_TYPE_TET
    localparam FMUL_LATENCY = 2;
    localparam FALN_LATENCY = 2;
    localparam FACC_LATENCY = 2;
    localparam FRND_LATENCY = 2;
    localparam FEDP_LATENCY = FMUL_LATENCY + FALN_LATENCY + FACC_LATENCY + FRND_LATENCY;
`else // VX_CFG_TCU_TYPE_TFR
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

`ifdef VX_CFG_TCU_SPARSE_ENABLE
    localparam LG_B_BS_SP = $clog2(TCU_B_BLOCK_SIZE_SP);
    wire is_sparse = (execute_if.data.op_type == INST_TCU_WMMA_SP)
              `ifdef VX_CFG_TCU_WGMMA_ENABLE
                 || (execute_if.data.op_type == INST_TCU_WGMMA_SP)
              `endif
                 ;
`endif

`ifdef VX_CFG_TCU_MX_ENABLE
    wire is_wmma = (execute_if.data.op_type == INST_TCU_WMMA)
              `ifdef VX_CFG_TCU_SPARSE_ENABLE
                 || (execute_if.data.op_type == INST_TCU_WMMA_SP)
              `endif
                 ;
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    wire mx_is_sparse = is_sparse;
`else
    wire mx_is_sparse = 1'b0;
`endif
    localparam FEDP_SF = TCU_MX_MAX_SF;
`else
    localparam FEDP_SF = 1;
`endif

    // -----------------------------------------------------------------------
    // WGMMA / WMMA abstraction layer
    // -----------------------------------------------------------------------
    // All WGMMA-vs-WMMA runtime differences are resolved here behind a
    // common interface.  Downstream code uses only these wires and never
    // references tbuf_* or is_wgmma directly.

    wire [TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0] rs1_data;
`ifdef VX_CFG_TCU_WGMMA_ENABLE
    wire [TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0] rs2_data;
`else
    wire [TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0] rs2_data;
`endif
    wire exe_ready_extra; // additional ready gating (tbuf_ready)

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    wire is_wgmma = (execute_if.data.op_type == INST_TCU_WGMMA)
              `ifdef VX_CFG_TCU_SPARSE_ENABLE
                 || (execute_if.data.op_type == INST_TCU_WGMMA_SP)
              `endif
                 ;
    wire wg_a_smem = execute_if.data.op_args.tcu.a_from_smem;

    // A/B operand mux: tile buffer (smem) or register file. The
    // RF-side rs2_data is NUM_THREADS lanes wide; the WGMMA bbuf can be
    // wider (TCU_WG_RS2_WIDTH lanes). Pad/truncate to the wgmma width on
    // the false branch so both arms match TCU_WG_RS2_WIDTH * XLEN bits.
    localparam WG_RS2_BITS = TCU_WG_RS2_WIDTH * `VX_CFG_XLEN;
    wire [WG_RS2_BITS-1:0] rs2_data_rf = WG_RS2_BITS'(execute_if.data.rs2_data);
    assign rs1_data = (is_wgmma && wg_a_smem) ? tbuf_rs1_data : execute_if.data.rs1_data;
    assign rs2_data = is_wgmma ? tbuf_rs2_data : rs2_data_rf;

  `ifdef VX_CFG_TCU_SPARSE_ENABLE
    // Sparse metadata lives in VX_tcu_sp_meta SRAM, preloaded via TCU_LD.
    wire [TCU_MAX_META_BLOCK_WIDTH-1:0] vld_meta_block = wmma_sp_meta;
  `endif

    assign exe_ready_extra = ~is_wgmma || tbuf_ready;
`else
    assign rs1_data = execute_if.data.rs1_data;
    assign rs2_data = execute_if.data.rs2_data;
  `ifdef VX_CFG_TCU_SPARSE_ENABLE
    wire [TCU_MAX_META_BLOCK_WIDTH-1:0] vld_meta_block = wmma_sp_meta;
  `endif
    assign exe_ready_extra = 1'b1;
`endif

    wire [3:0] step_m = execute_if.data.op_args.tcu.step_m;
    wire [3:0] step_n = execute_if.data.op_args.tcu.step_n;
    wire [3:0] step_k = execute_if.data.op_args.tcu.step_k;

    wire [4:0] fmt_s = execute_if.data.op_args.tcu.fmt_s;
    wire [4:0] fmt_d = execute_if.data.op_args.tcu.fmt_d;

    wire execute_fire = execute_if.valid && execute_if.ready;

    // -----------------------------------------------------------------------
    // Sparse metadata: VX_tcu_sp_meta (for WMMA_SP) + optional tile-buffer mux
    // -----------------------------------------------------------------------

    tcu_header_t mdata_queue_in;
    always_comb begin
        mdata_queue_in = execute_if.data.header;
    end

    `UNUSED_VAR ({step_m, step_n, step_k, fmt_s, fmt_d, execute_if.data});

`ifdef VX_TCU_LD_TRACE
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    // META_RD trace: logs vld_meta_block at FEDP consume time.
    // Format: META_RD,wid,step_m,step_k,wg_bank,word_lo32
    wire trc_is_sp = (execute_if.data.op_type == INST_OP_BITS'(INST_TCU_WMMA_SP))
        `ifdef VX_CFG_TCU_WGMMA_ENABLE
                  || (execute_if.data.op_type == INST_OP_BITS'(INST_TCU_WGMMA_SP))
        `endif
                  ;
    wire [3:0] trc_wg_bank = ((TCU_K_STEPS > 2) ? (step_m << 1) : step_m) | step_k;
    always @(posedge clk) begin
        if (execute_fire && trc_is_sp) begin
            $write("META_RD,%0d,%0d,%0d,%0d,0x%08h\n",
                execute_if.data.header.wid, step_m, step_k, trc_wg_bank,
                vld_meta_block[31:0]);
        end
    end
`endif
`endif

    // -----------------------------------------------------------------------
    // Pipeline control
    // -----------------------------------------------------------------------

    wire mdata_queue_full;

    wire result_fire = result_if.valid && result_if.ready;
    wire fedp_enable, fedp_done;

    reg [PIPE_LATENCY-1:0] fedp_delay_pipe;
    always @(posedge clk) begin
        if (reset) begin
            fedp_delay_pipe <= '0;
        end else begin
            if (fedp_enable) begin
                fedp_delay_pipe <= fedp_delay_pipe >> 1;
            end
            if (execute_fire) begin
                fedp_delay_pipe[PIPE_LATENCY-1] <= 1;
            end
        end
    end

    assign fedp_done        = fedp_delay_pipe[0];
    assign result_if.valid  = fedp_done;
    assign fedp_enable      = ~fedp_done || result_if.ready;
    assign execute_if.ready = ~mdata_queue_full && fedp_enable && exe_ready_extra;

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
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    wire [OFF_W-1:0] b_off = is_sparse
        ? (OFF_W'(step_n) & OFF_W'(TCU_B_SUB_BLOCKS_SP-1)) << LG_B_BS_SP
        : (OFF_W'(step_n) & OFF_W'(TCU_B_SUB_BLOCKS-1)) << LG_B_BS;
`else
    wire [OFF_W-1:0] b_off = (OFF_W'(step_n) & OFF_W'(TCU_B_SUB_BLOCKS-1)) << LG_B_BS;
`endif

    // -----------------------------------------------------------------------
    // Unified sparse metadata
    // -----------------------------------------------------------------------

`ifdef VX_CFG_TCU_SPARSE_ENABLE
    wire sparse_meta_wr_en = ext_meta_wr_en && !ext_meta_wr_idx[4];
    wire [TCU_MAX_META_BLOCK_WIDTH-1:0] wmma_sp_meta;
    VX_tcu_sp_meta #(
        .INSTANCE_ID (INSTANCE_ID)
    ) tcu_meta (
        .clk    (clk),
        .reset  (reset),
        .wr_en  (sparse_meta_wr_en),
        .wr_wid (ext_meta_wr_wid),
        .wr_idx (ext_meta_wr_idx[3:0]),
        .wr_data(ext_meta_wr_data),
        // Read wid follows the consuming warp's identity (the
        // WMMA_SP/WGMMA_SP currently in execute). Decoupling read wid
        // from write wid prevents the FEDP from seeing another warp's
        // metadata when the AGU's owner_header_r holds a stale wid.
        .rd_wid (execute_if.data.header.wid),
        .step_m (step_m),
        .step_k (step_k),
        .vld_block(wmma_sp_meta)
    );
`endif

`ifdef VX_CFG_TCU_MX_ENABLE
`ifndef VX_CFG_TCU_SPARSE_ENABLE
    `UNUSED_VAR (ext_meta_wr_idx[3:1])
`endif
    wire [TCU_BLOCK_CAP-1:0][31:0] mx_meta_a;
    wire [TCU_BLOCK_CAP-1:0][31:0] mx_meta_b;
    VX_tcu_mx_meta mx_meta (
        .clk     (clk),
        .reset   (reset),
        .wr_en   (ext_meta_wr_en && ext_meta_wr_idx[4]),
        .wr_wid  (ext_meta_wr_wid),
        .wr_axis (ext_meta_wr_idx[0]),
        .wr_data (ext_meta_wr_data),
        .rd_wid  (execute_if.data.header.wid),
        .meta_a  (mx_meta_a),
        .meta_b  (mx_meta_b)
    );

    localparam MX_IDX_W = $clog2(TCU_TILE_M > TCU_TILE_N ? TCU_TILE_M : TCU_TILE_N);
    localparam MX_K_IDX_W = `LOG2UP(TCU_TILE_K * TCU_MAX_ELT_RATIO);
    localparam MX_SCALE_IDX_W = $clog2(TCU_BLOCK_CAP * 4);

    function automatic [7:0] mx_scale_at(
        input logic [TCU_BLOCK_CAP-1:0][31:0] meta,
        input logic [4:0] fmt,
        input logic [MX_IDX_W-1:0] mn_idx,
        input logic [MX_K_IDX_W-1:0] k_base_idx
    );
        logic [MX_SCALE_IDX_W-1:0] scale_k;
        logic [MX_SCALE_IDX_W-1:0] scale_idx;
        logic [`LOG2UP(TCU_BLOCK_CAP)-1:0] word_idx;
        logic [1:0] byte_idx;
        begin
            scale_k = MX_SCALE_IDX_W'(k_base_idx / mx_scale_block_size(fmt));
            scale_idx = MX_SCALE_IDX_W'(mn_idx) * MX_SCALE_IDX_W'(mx_scale_blocks_k(fmt))
                      + MX_SCALE_IDX_W'(scale_k);
            word_idx = `LOG2UP(TCU_BLOCK_CAP)'(scale_idx >> 2);
            byte_idx = scale_idx[1:0];
            mx_scale_at = meta[word_idx][byte_idx * 8 +: 8];
        end
    endfunction

    wire [TCU_TC_M-1:0][FEDP_SF-1:0][7:0] mx_sf_a;
    wire [TCU_TC_N-1:0][FEDP_SF-1:0][7:0] mx_sf_b;
    wire [3:0] mx_elems_per_word = 4'(32 / tcu_fmt_width(fmt_s));
    wire [MX_K_IDX_W:0] mx_fedp_elems = (MX_K_IDX_W+1)'(
        (MX_K_IDX_W+1)'(TCU_TC_K) * (MX_K_IDX_W+1)'(mx_elems_per_word)
        * (MX_K_IDX_W+1)'(mx_is_sparse ? 2 : 1));
    wire [MX_K_IDX_W-1:0] mx_k_base_idx = MX_K_IDX_W'(step_k * mx_fedp_elems);

    for (genvar i = 0; i < TCU_TC_M; ++i) begin : g_mx_sf_a_i
        wire [MX_IDX_W-1:0] mx_a_idx = MX_IDX_W'(step_m) * MX_IDX_W'(TCU_TC_M) + MX_IDX_W'(i);
        for (genvar s = 0; s < FEDP_SF; ++s) begin : g_s
            wire [MX_K_IDX_W-1:0] mx_k_idx = mx_k_base_idx + MX_K_IDX_W'((s * mx_fedp_elems) / FEDP_SF);
            assign mx_sf_a[i][s] = is_wmma ? mx_scale_at(mx_meta_a, fmt_s, mx_a_idx, mx_k_idx) : '0;
        end
    end

    for (genvar j = 0; j < TCU_TC_N; ++j) begin : g_mx_sf_b_j
        wire [MX_IDX_W-1:0] mx_b_idx = MX_IDX_W'(step_n) * MX_IDX_W'(TCU_TC_N) + MX_IDX_W'(j);
        for (genvar s = 0; s < FEDP_SF; ++s) begin : g_s
            wire [MX_K_IDX_W-1:0] mx_k_idx = mx_k_base_idx + MX_K_IDX_W'((s * mx_fedp_elems) / FEDP_SF);
            assign mx_sf_b[j][s] = is_wmma ? mx_scale_at(mx_meta_b, fmt_s, mx_b_idx, mx_k_idx) : '0;
        end
    end
`endif

    // -----------------------------------------------------------------------
    // FEDP grid: TCU_TC_M × TCU_TC_N compute elements
    // -----------------------------------------------------------------------

    wire [TCU_TC_M-1:0][TCU_TC_N-1:0][31:0] d_val;

    for (genvar i = 0; i < TCU_TC_M; ++i) begin : g_i
        for (genvar j = 0; j < TCU_TC_N; ++j) begin : g_j
        `ifdef VX_CFG_TCU_SPARSE_ENABLE
            wire [TCU_TC_K-1:0][31:0] a_row, b_col, b_col_dense, b_col_sparse, b_col_1, b_col_2;
        `else
            wire [TCU_TC_K-1:0][31:0] a_row, b_col;
        `endif
        `ifdef VX_CFG_TCU_MX_ENABLE
            wire [FEDP_SF-1:0][7:0] sf_a = mx_sf_a[i];
            wire [FEDP_SF-1:0][7:0] sf_b = mx_sf_b[j];
        `endif
            for (genvar k_idx = 0; k_idx < TCU_TC_K; ++k_idx) begin : g_slice_assign
                assign a_row[k_idx] = 32'(rs1_data[a_off + i * TCU_TC_K + k_idx]);
            `ifdef VX_CFG_TCU_SPARSE_ENABLE
                assign b_col_dense[k_idx] = 32'(rs2_data[b_off + j * TCU_TC_K + k_idx]);
                localparam J_SP = SYM_SPARSE ? (j % (TCU_TC_N / 2)) : j;
                // rs2_data sparse-pair layout differs by op:
                //   WGMMA_SP: source is tbuf (shared mem), K-major →
                //     idx = k_idx*(TC_N*2) + J_SP*2 + lane
                //   WMMA_SP : source is the register file, J-major →
                //     idx = J_SP*(TC_K*2) + k_idx*2 + lane
                // The two layouts are incompatible; separate formulas are required.
            `ifdef VX_CFG_TCU_WGMMA_ENABLE
                wire [31:0] b_col_1_wg = 32'(rs2_data[b_off + k_idx * TCU_TC_N * 2 + J_SP * 2]);
                wire [31:0] b_col_2_wg = 32'(rs2_data[b_off + k_idx * TCU_TC_N * 2 + J_SP * 2 + 1]);
                wire [31:0] b_col_1_wm = 32'(rs2_data[b_off + J_SP * TCU_TC_K * 2 + k_idx * 2]);
                wire [31:0] b_col_2_wm = 32'(rs2_data[b_off + J_SP * TCU_TC_K * 2 + k_idx * 2 + 1]);
                assign b_col_1[k_idx] = is_wgmma ? b_col_1_wg : b_col_1_wm;
                assign b_col_2[k_idx] = is_wgmma ? b_col_2_wg : b_col_2_wm;
            `else
                assign b_col_1[k_idx] = 32'(rs2_data[b_off + J_SP * TCU_TC_K * 2 + k_idx * 2]);
                assign b_col_2[k_idx] = 32'(rs2_data[b_off + J_SP * TCU_TC_K * 2 + k_idx * 2 + 1]);
            `endif
            `else
                assign b_col[k_idx] = 32'(rs2_data[b_off + j * TCU_TC_K + k_idx]);
            `endif
            end

            wire [31:0] c_val = 32'(execute_if.data.rs3_data[i * TCU_TC_N + j]);

        `ifdef VX_CFG_TCU_SPARSE_ENABLE
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

        `ifdef VX_TCU_LD_TRACE
            // GATHER trace: GATHER,wid,step_m,step_n,i,k,bword0,bword1,lo,hi,gathered
            // One line per (i, j, k_idx); emitted only for sparse ops.
            always @(posedge clk) begin
                if (execute_fire && is_sparse) begin
                    for (int kk = 0; kk < TCU_TC_K; ++kk) begin
                        $write("GATHER,%0d,%0d,%0d,%0d,%0d,0x%08h,0x%08h,?,?,0x%08h\n",
                            execute_if.data.header.wid, step_m, step_n,
                            i, j*TCU_TC_K + kk,
                            b_col_1[kk], b_col_2[kk], b_col_sparse[kk]);
                    end
                end
            end
        `endif
        `endif

        // Dual-side sparse lane mask
        `ifdef VX_CFG_TCU_TYPE_TFR
            `define VX_TCU_TFR_LANE_MASK_ENABLE
        `endif
        `ifdef VX_CFG_TCU_TYPE_TET
            `define VX_TCU_TFR_LANE_MASK_ENABLE
        `endif

        `ifdef VX_TCU_TFR_LANE_MASK_ENABLE
            wire [TCU_MAX_INPUTS-1:0] vld_mask_r;
        `ifdef VX_CFG_TCU_DSM_ENABLE
            wire [TCU_MAX_INPUTS-1:0] vld_mask;
            VX_tcu_dsm #(
                .N (TCU_TC_K)
            ) dual_sparse_mask (
                .fmt_s    (fmt_s),
                .a_row    (a_row),
                .b_col    (b_col),
                .vld_mask (vld_mask)
            );
            VX_pipe_register #(
                .DATAW (TCU_MAX_INPUTS)
            ) pipe_vld_mask (
                .clk      (clk),
                .reset    (reset),
                .enable   (fedp_enable),
                .data_in  (vld_mask),
                .data_out (vld_mask_r)
            );
        `else
            assign vld_mask_r = '1;
        `endif
        `endif
        `ifdef VX_TCU_TFR_LANE_MASK_ENABLE
            `undef VX_TCU_TFR_LANE_MASK_ENABLE
        `endif

            wire [4:0] fmt_s_r, fmt_d_r;
            wire [TCU_TC_K-1:0][31:0] a_row_r, b_col_r;
        `ifdef VX_CFG_TCU_MX_ENABLE
            wire [FEDP_SF-1:0][7:0] sf_a_r, sf_b_r;
        `endif
            wire [31:0] c_val_r;

        `ifdef VX_CFG_TCU_MX_ENABLE
            VX_pipe_register #(
                .DATAW (32 + 5 + 5 + TCU_TC_K * 32 + TCU_TC_K * 32 + 2 * FEDP_SF * 8)
            ) pipe_fedp (
                .clk      (clk),
                .reset    (reset),
                .enable   (fedp_enable),
                .data_in  ({c_val,   sf_b,   sf_a,   fmt_s,   fmt_d,   b_col,   a_row}),
                .data_out ({c_val_r, sf_b_r, sf_a_r, fmt_s_r, fmt_d_r, b_col_r, a_row_r})
            );
        `else
            VX_pipe_register #(
                .DATAW (32 + 5 + 5 + TCU_TC_K * 32 + TCU_TC_K * 32)
            ) pipe_fedp (
                .clk      (clk),
                .reset    (reset),
                .enable   (fedp_enable),
                .data_in  ({c_val,   fmt_s,   fmt_d,   b_col,   a_row}),
                .data_out ({c_val_r, fmt_s_r, fmt_d_r, b_col_r, a_row_r})
            );
        `endif

        `ifdef VX_CFG_TCU_TYPE_DPI
            VX_tcu_fedp_dpi #(
                .INSTANCE_ID (INSTANCE_ID),
                .LATENCY (FEDP_LATENCY),
                .N (TCU_TC_K),
                .SF (FEDP_SF)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .enable(fedp_enable),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row(a_row_r),
                .b_col(b_col_r),
            `ifdef VX_CFG_TCU_MX_ENABLE
                .sf_a  (sf_a_r),
                .sf_b  (sf_b_r),
            `endif
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `elsif VX_CFG_TCU_TYPE_BHF
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
        `elsif VX_CFG_TCU_TYPE_FPNEW
            VX_tcu_fedp_fpnew #(
                .INSTANCE_ID (INSTANCE_ID),
                .LATENCY (FEDP_LATENCY),
                .N (TCU_TC_K)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .enable(fedp_enable),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row (a_row_r),
                .b_col (b_col_r),
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `elsif VX_CFG_TCU_TYPE_TFR
            VX_tcu_fedp_tfr #(
                .INSTANCE_ID (INSTANCE_ID),
                .LATENCY (FEDP_LATENCY),
                .N (TCU_TC_K),
                .SF (FEDP_SF)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .vld_mask(vld_mask_r),
                .enable(fedp_enable),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row (a_row_r),
                .b_col (b_col_r),
            `ifdef VX_CFG_TCU_MX_ENABLE
                .sf_a  (sf_a_r),
                .sf_b  (sf_b_r),
            `endif
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `elsif VX_CFG_TCU_TYPE_TET
            VX_tcu_fedp_tet #(
                .INSTANCE_ID (INSTANCE_ID),
                .LATENCY (FEDP_LATENCY),
                .N (TCU_TC_K),
                .SF (FEDP_SF)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .vld_mask(vld_mask_r),
                .enable(fedp_enable),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row (a_row_r),
                .b_col (b_col_r),
            `ifdef VX_CFG_TCU_MX_ENABLE
                .sf_a  (sf_a_r),
                .sf_b  (sf_b_r),
            `endif
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `elsif VX_CFG_TCU_TYPE_DSP
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
            if (`VX_CFG_XLEN > 32) begin : g_result_nanbox
                assign result_if.data.data[i * TCU_TC_N + j] = {32'hffffffff, d_val[i][j]};
            end else begin : g_result_passthrough
                assign result_if.data.data[i * TCU_TC_N + j] = d_val[i][j];
            end

        `ifdef DBG_TRACE_TCU
            always @(posedge clk) begin
                if (execute_if.valid && execute_if.ready) begin
                    `TRACE(3, ("%t: %s FEDP-enq: wid=%0d, cta_id=%0d, i=%0d, j=%0d, m=%0d, n=%0d, a_row=", $time, INSTANCE_ID, execute_if.data.header.wid, execute_if.data.header.cta_id, i, j, step_m, step_n))
                    `TRACE_ARRAY1D(2, "0x%0h", a_row, TCU_TC_K)
                    `TRACE(3, (", b_col="));
                    `TRACE_ARRAY1D(2, "0x%0h", b_col, TCU_TC_K)
                    `TRACE(3, (", c_val=0x%0h (#%0d)\n", c_val, execute_if.data.header.uuid));
                end
                if (result_if.valid && result_if.ready) begin
                    `TRACE(3, ("%t: %s FEDP-deq: wid=%0d, cta_id=%0d, i=%0d, j=%0d, d_val=0x%0h (#%0d)\n", $time, INSTANCE_ID, result_if.data.header.wid, result_if.data.header.cta_id, i, j, d_val[i][j], result_if.data.header.uuid));
                end
            end
        `endif // DBG_TRACE_TCU
        end
    end

endmodule
