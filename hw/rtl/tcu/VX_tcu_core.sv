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

    wire is_sparse = (execute_if.data.op_type == INST_TCU_WMMA_SP);
    wire is_meta_store = (execute_if.data.op_type == INST_TCU_META_STORE);
`endif

    wire [3:0] step_m = execute_if.data.op_args.tcu.step_m;
    wire [3:0] step_n = execute_if.data.op_args.tcu.step_n;
    wire [3:0] step_k = execute_if.data.op_args.tcu.step_k;

    wire [3:0] fmt_s = execute_if.data.op_args.tcu.fmt_s;
    wire [3:0] fmt_d = execute_if.data.op_args.tcu.fmt_d;

`ifdef TCU_SPARSE_ENABLE
    wire [`LOG2UP(`NUM_WARPS)-1:0] wid = execute_if.data.header.wid;

    // meta_store: extract per-row write data from rs1_data lanes
    localparam PER_WARP_DEPTH = TCU_META_PER_WARP_DEPTH;
    localparam COLS_PER_LOAD  = TCU_META_COLS_PER_LOAD;
    localparam LG_CPL = $clog2((COLS_PER_LOAD > 1) ? COLS_PER_LOAD : 2);
    localparam LG_PD  = $clog2(PER_WARP_DEPTH);
    wire meta_wr_en = execute_fire && is_meta_store;
    wire [PER_WARP_DEPTH-1:0][31:0] meta_wr_data;
    wire [$clog2(TCU_BLOCK_CAP)-1:0] meta_thread_offset;
    if (COLS_PER_LOAD > 1) begin : g_meta_off
        assign meta_thread_offset = {fmt_d[LG_CPL-1:0], {LG_PD{1'b0}}};
    end else begin : g_meta_off
        assign meta_thread_offset = '0;
    end
    for (genvar r = 0; r < PER_WARP_DEPTH; ++r) begin : g_meta_wr
        assign meta_wr_data[r] = 32'(execute_if.data.rs1_data[meta_thread_offset + r]);
    end

    // meta_store: force rd=0 in mdata_queue header (x0 write is harmless)
    tcu_header_t mdata_queue_in;
    always_comb begin
        mdata_queue_in = execute_if.data.header;
        if (is_meta_store) begin
            mdata_queue_in.rd = '0;
        end
    end
`else
    tcu_header_t mdata_queue_in;
    always_comb begin
        mdata_queue_in = execute_if.data.header;
    end
`endif

    `UNUSED_VAR ({step_m, step_n, step_k, fmt_s, fmt_d, execute_if.data});

    wire mdata_queue_full;

    wire execute_fire = execute_if.valid && execute_if.ready;
    wire result_fire = result_if.valid && result_if.ready;
    wire fedp_enable, fedp_done;

    // FEDP delay handling
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
    assign fedp_done = fedp_delay_pipe[0];

    assign result_if.valid  = fedp_done;
    assign fedp_enable      = ~result_if.valid || result_if.ready;
    assign execute_if.ready = ~mdata_queue_full && fedp_enable;

    VX_fifo_queue #(
        .DATAW ($bits(tcu_header_t)),
        .DEPTH (MDATA_QUEUE_DEPTH),
        .OUT_REG (1)
    ) mdata_queue (
        .clk    (clk),
        .reset  (reset),
        .push   (execute_fire),
        .pop    (result_fire),
        .data_in(mdata_queue_in),
        .data_out(result_if.data.header),
        `UNUSED_PIN(empty),
        `UNUSED_PIN(alm_empty),
        .full   (mdata_queue_full),
        `UNUSED_PIN(alm_full),
        `UNUSED_PIN(size)
    );

    wire [OFF_W-1:0] a_off = (OFF_W'(step_m) & OFF_W'(TCU_A_SUB_BLOCKS-1)) << LG_A_BS;
`ifdef TCU_SPARSE_ENABLE
    wire [OFF_W-1:0] b_off = is_sparse
        ? (OFF_W'(step_n) & OFF_W'(TCU_B_SUB_BLOCKS_SP-1)) << LG_B_BS_SP
        : (OFF_W'(step_n) & OFF_W'(TCU_B_SUB_BLOCKS-1))    << LG_B_BS;
`else
    wire [OFF_W-1:0] b_off = (OFF_W'(step_n) & OFF_W'(TCU_B_SUB_BLOCKS-1)) << LG_B_BS;
`endif

    wire [TCU_TC_M-1:0][TCU_TC_N-1:0][31:0] d_val;

`ifdef TCU_SPARSE_ENABLE
    // 2:4 sparsity metadata (sized for worst-case: int4, I_RATIO=8)
    wire [TCU_MAX_META_BLOCK_WIDTH-1:0] vld_meta_block;

    VX_tcu_meta #(
        .INSTANCE_ID     (INSTANCE_ID),
        .META_BLOCK_WIDTH(TCU_MAX_META_BLOCK_WIDTH),
        .PER_WARP_DEPTH  (PER_WARP_DEPTH)
    ) tcu_meta (
        .clk           (clk),
        .reset         (reset),
        .raddr_wid     (wid),
        .step_m        (step_m),
        .step_k        (step_k),
        .vld_meta_block(vld_meta_block),
        .wr_en         (meta_wr_en),
        .wr_wid        (wid),
        .wr_col_idx    (fmt_d),
        .wr_data       (meta_wr_data)
    );
`endif

    for (genvar i = 0; i < TCU_TC_M; ++i) begin : g_i
        for (genvar j = 0; j < TCU_TC_N; ++j) begin : g_j
        `ifdef TCU_SPARSE_ENABLE
            wire [TCU_TC_K-1:0][31:0] a_row, b_col, b_col_dense, b_col_sparse, b_col_1, b_col_2;
        `else
            wire [TCU_TC_K-1:0][31:0] a_row, b_col;
        `endif
            for (genvar k_idx = 0; k_idx < TCU_TC_K; ++k_idx) begin : g_slice_assign
                assign a_row[k_idx]   = 32'(execute_if.data.rs1_data[a_off + i * TCU_TC_K + k_idx]);
            `ifdef TCU_SPARSE_ENABLE
                assign b_col_dense[k_idx] = 32'(execute_if.data.rs2_data[b_off + j * TCU_TC_K + k_idx]);
                // NT=16 sparse: j_sp = j % 2 (wraps j=2,3 back to lanes 0..15)
                // NT=8/32: j_sp = j (no wrapping needed)
                localparam J_SP = NT16_SPARSE ? (j % (TCU_TC_N / 2)) : j;
                assign b_col_1[k_idx] = 32'(execute_if.data.rs2_data[b_off + J_SP * TCU_TC_K * 2 + k_idx * 2]);
                assign b_col_2[k_idx] = 32'(execute_if.data.rs2_data[b_off + J_SP * TCU_TC_K * 2 + k_idx * 2 + 1]);
            `else
                assign b_col[k_idx] = 32'(execute_if.data.rs2_data[b_off + j * TCU_TC_K + k_idx]);
            `endif
            end
            wire [31:0] c_val = 32'(execute_if.data.rs3_data[i * TCU_TC_N + j]);
        `ifdef TCU_SPARSE_ENABLE
            /* verilator lint_off UNUSEDSIGNAL */
            wire [TCU_MAX_INPUTS-1:0] vld_mask = '1; // TODO: should connect to input source
            /* verilator lint_on UNUSEDSIGNAL */

            VX_tcu_sel #(
                .INSTANCE_ID (INSTANCE_ID),
                .ROW_IDX     (i)
            ) tcu_sel (
                .fmt_s          (fmt_s),
                .b_col_1        (b_col_1),
                .b_col_2        (b_col_2),
                .vld_meta_block (vld_meta_block),
                .b_col          (b_col_sparse)
            );

            // Select dense or sparse B column
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

        assign result_if.data.data[i * TCU_TC_N + j] = `XLEN'($signed(d_val[i][j]));

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
