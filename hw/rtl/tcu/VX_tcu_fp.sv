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

module VX_tcu_fp import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
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

    localparam MDATA_WIDTH = UUID_WIDTH + NW_WIDTH + PC_BITS + NUM_REGS_BITS;

`ifdef TCU_DSP
    localparam FMUL_LATENCY = 8;
    localparam FADD_LATENCY = 11;
    localparam FRND_LATENCY = 2;
`else
    localparam FMUL_LATENCY = 2;
    localparam FADD_LATENCY = 1;
    localparam FRND_LATENCY = 1;
`endif
    localparam ACC_LATENCY  = $clog2(2 * TCU_TC_K) * FADD_LATENCY + FADD_LATENCY;
    localparam FEDP_LATENCY = FMUL_LATENCY + ACC_LATENCY + FRND_LATENCY;

    localparam PIPE_LATENCY = FEDP_LATENCY + 1;
    localparam MDATA_QUEUE_DEPTH = 1 << $clog2(PIPE_LATENCY);

    localparam LG_A_BS = $clog2(TCU_A_BLOCK_SIZE);
    localparam LG_B_BS = $clog2(TCU_B_BLOCK_SIZE);
    localparam OFF_W   = $clog2(TCU_BLOCK_CAP);

    wire [3:0] step_m = execute_if.data.op_args.tcu.step_m;
    wire [3:0] step_n = execute_if.data.op_args.tcu.step_n;
    wire [3:0] step_k = execute_if.data.op_args.tcu.step_k;
    wire [3:0] fmt_s = execute_if.data.op_args.tcu.fmt_s;
    wire [3:0] fmt_d = execute_if.data.op_args.tcu.fmt_d;

    `UNUSED_VAR ({fmt_s, fmt_d, step_m, step_n, step_k});


    wire [MDATA_WIDTH-1:0] mdata_queue_din, mdata_queue_dout;
    wire mdata_queue_full;

    assign mdata_queue_din = {
        execute_if.data.uuid,
        execute_if.data.wid,
        execute_if.data.PC,
        execute_if.data.rd
    };

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
        .DATAW (MDATA_WIDTH),
        .DEPTH (MDATA_QUEUE_DEPTH),
        .OUT_REG (1)
    ) mdata_queue (
        .clk    (clk),
        .reset  (reset),
        .push   (execute_fire),
        .pop    (result_fire),
        .data_in(mdata_queue_din),
        .data_out(mdata_queue_dout),
        `UNUSED_PIN(empty),
        `UNUSED_PIN(alm_empty),
        .full   (mdata_queue_full),
        `UNUSED_PIN(alm_full),
        `UNUSED_PIN(size)
    );

    wire [OFF_W-1:0] a_off = (OFF_W'(step_m) & OFF_W'(TCU_A_SUB_BLOCKS-1)) << LG_A_BS;
    wire [OFF_W-1:0] b_off = (OFF_W'(step_n) & OFF_W'(TCU_B_SUB_BLOCKS-1)) << LG_B_BS;

    // local_m / thread base for sparse A + metadata (match sim tensor_unit.cpp)
    wire [1:0] sp_deg_fp = execute_if.data.op_args.tcu.sparsity_degree;
    wire [7:0] step_m_fp = 8'(execute_if.data.op_args.tcu.step_m);
    wire [7:0] mspg_fp = (sp_deg_fp == 2'd2) ? 8'(TCU_MSPG_24) : 8'(TCU_MSPG_14);
    wire [7:0] local_m_fp = step_m_fp % mspg_fp;

    wire [TCU_TC_M-1:0][TCU_TC_N-1:0][`XLEN-1:0] d_val;

    for (genvar i = 0; i < TCU_TC_M; ++i) begin : g_i
        for (genvar j = 0; j < TCU_TC_N; ++j) begin : g_j

            wire [TCU_TC_K-1:0][`XLEN-1:0] a_row = execute_if.data.rs1_data[a_off + i * TCU_TC_K +: TCU_TC_K];
            wire [TCU_TC_K-1:0][`XLEN-1:0] b_col = execute_if.data.rs2_data[b_off + j * TCU_TC_K +: TCU_TC_K];
            wire [`XLEN-1:0] c_val = execute_if.data.rs3_data[i * TCU_TC_N + j];
            wire [31:0] step_k_zext = 32'(step_k);
            wire [31:0] base_th = 32'(local_m_fp) * 32'(TCU_TC_M) + 32'(i);
            wire [`LOG2UP(`NUM_TCU_LANES)-1:0] meta_lane_idx =
                (sp_deg_fp == 2'd2)
                    ? `LOG2UP(`NUM_TCU_LANES)'(base_th * 32'd2 + step_k_zext)
                    : `LOG2UP(`NUM_TCU_LANES)'(base_th);
            wire [`XLEN-1:0] metadata_word = execute_if.data.rs4_data[meta_lane_idx];
            wire [OFF_W-1:0] a_flat_14 = OFF_W'(a_off) + OFF_W'(base_th);
            wire [OFF_W-1:0] a_flat_24 = OFF_W'(a_off) + (OFF_W'(base_th) << 1) + OFF_W'(step_k_zext);

            // Use metadata-masked A row for sparse WMMA; otherwise use original A row.
            wire is_sparse_wmma = (execute_if.data.op_args.tcu.sparsity_degree != 2'b00);
            wire [TCU_TC_K-1:0][`XLEN-1:0] a_row_sel;

            reg [TCU_TC_K-1:0][`XLEN-1:0] a_row_masked;
            integer pos;
            integer val_rank;
            reg [15:0] src_lo, src_hi, src_val;
            reg [`XLEN-1:0] src_packed;
            reg [3:0] meta4;
            reg [15:0] comp_val;

            always_comb begin
                // Dense/default passthrough.
                a_row_masked = a_row;
                src_packed = '0;
                src_lo = '0;
                src_hi = '0;
                src_val = '0;
                meta4 = '0;
                val_rank = 0;
                comp_val = '0;

                if (is_sparse_wmma) begin
                    // Sparse fp16: 2:4 unpacks two values from one packed reg per k-block;
                    // 1:4 uses one reg per (local_m,i), step_k selects nibble + element lane.
                    a_row_masked = '0;

                    if (sp_deg_fp == 2'd2) begin
                        src_packed = execute_if.data.rs1_data[a_flat_24];
                        src_lo = src_packed[15:0];
                        src_hi = src_packed[31:16];
                        meta4 = metadata_word[3:0];
                        val_rank = 0;
                        for (pos = 0; pos < 4; pos = pos + 1) begin
                            if (meta4[pos]) begin
                                src_val = (val_rank == 0) ? src_lo : src_hi;
                                a_row_masked[pos >> 1][((pos % 2) * 16) +: 16] = src_val;
                                val_rank = val_rank + 1;
                            end
                        end
                    end else begin
                        // 1:4 (sparsity_degree == 1): tensor_unit.cpp
                        src_packed = execute_if.data.rs1_data[a_flat_14];
                        meta4 = 4'((metadata_word >> (step_k_zext * 32'd4)) & 32'hF);
                        comp_val = 16'(src_packed >> (step_k_zext * 32'd16));
                        if (meta4[0]) begin
                            a_row_masked[0][0 +: 16] = comp_val;
                        end else if (meta4[1]) begin
                            a_row_masked[0][16 +: 16] = comp_val;
                        end else if (meta4[2]) begin
                            a_row_masked[1][0 +: 16] = comp_val;
                        end else if (meta4[3]) begin
                            a_row_masked[1][16 +: 16] = comp_val;
                        end
                    end
                end
            end

            assign a_row_sel = is_sparse_wmma ? a_row_masked : a_row;
            wire [2:0] fmt_s_r, fmt_d_r;
            wire [TCU_TC_K-1:0][`XLEN-1:0] a_row_r, b_col_r;
            wire [`XLEN-1:0] c_val_r;

            `BUFFER_EX (
                {a_row_r, b_col_r, c_val_r, fmt_s_r,    fmt_d_r},
                {a_row_sel, b_col, c_val,   fmt_s[2:0], fmt_d[2:0]},
                fedp_enable,
                0, // resetw
                1  // depth
            );

        `ifdef TCU_DPI
            VX_tcu_fedp_dpi #(
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
        `elsif TCU_BHF
            VX_tcu_fedp_bhf #(
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
        `elsif TCU_DSP
            VX_tcu_fedp_dsp #(
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
        `endif

        `ifdef DBG_TRACE_TCU
            always @(posedge clk) begin
                if (execute_if.valid && execute_if.ready) begin
                    `TRACE(3, ("%t: %s FEDP-enq: wid=%0d, i=%0d, j=%0d, m=%0d, n=%0d, a_row=", $time, INSTANCE_ID, execute_if.data.wid, i, j, step_m, step_n))
                    `TRACE_ARRAY1D(2, "0x%0h", a_row_r, TCU_TC_K)
                    `TRACE(3, (", b_col="));
                    `TRACE_ARRAY1D(2, "0x%0h", b_col_r, TCU_TC_K)
                    `TRACE(3, (", c_val=0x%0h, tmask=0x%0h (#%0d)\n", c_val, execute_if.data.tmask, execute_if.data.uuid));
                    // Extra debug for sparse WMMA
                    if (is_sparse_wmma) begin
                        `TRACE(3, ("%t: %s FEDP-enq-sparse: wid=%0d, i=%0d, j=%0d, metadata=0x%04b, a_row_masked=", $time, INSTANCE_ID, execute_if.data.wid, i, j, metadata_word))
                        `TRACE_ARRAY1D(2, "0x%0h", a_row_masked, TCU_TC_K)
                        `TRACE(3, ("\n"));
                    end
                end
                if (result_if.valid && result_if.ready) begin
                    `TRACE(3, ("%t: %s FEDP-deq: wid=%0d, i=%0d, j=%0d, d_val=0x%0h (#%0d)\n", $time, INSTANCE_ID, result_if.data.wid, i, j, d_val[i][j], result_if.data.uuid));
                end
            end
        `endif // DBG_TRACE_TCU
        end
    end

    assign result_if.data.wb  = 1;
    assign result_if.data.tmask = {`NUM_THREADS{1'b1}};
    assign result_if.data.data  = d_val;
    assign result_if.data.pid = 0;
    assign result_if.data.sop = 1;
    assign result_if.data.eop = 1;

    assign {
        result_if.data.uuid,
        result_if.data.wid,
        result_if.data.PC,
        result_if.data.rd
    } = mdata_queue_dout;

endmodule
