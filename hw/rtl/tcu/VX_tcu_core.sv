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
`else // TCU_TYPE_DRL
    localparam FMUL_LATENCY = 1;
    localparam FALN_LATENCY = 1;
    localparam FACC_LATENCY = 1;
    localparam FRND_LATENCY = 1;
    localparam FEDP_LATENCY = FMUL_LATENCY + FALN_LATENCY + FACC_LATENCY + FRND_LATENCY;
`endif

    localparam PIPE_LATENCY = FEDP_LATENCY + 1;
    localparam MDATA_QUEUE_DEPTH = 1 << $clog2(PIPE_LATENCY);

    localparam LG_A_BS = $clog2(TCU_A_BLOCK_SIZE);
    localparam LG_B_BS = $clog2(TCU_B_BLOCK_SIZE);
    localparam OFF_W   = $clog2(TCU_BLOCK_CAP);

    wire [3:0] step_m = execute_if.data.op_args.tcu.step_m;
    wire [3:0] step_n = execute_if.data.op_args.tcu.step_n;

    wire [3:0] fmt_s = execute_if.data.op_args.tcu.fmt_s;
    wire [3:0] fmt_d = execute_if.data.op_args.tcu.fmt_d;

    `UNUSED_VAR ({step_m, step_n, fmt_s, fmt_d, execute_if.data});

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
        .data_in(execute_if.data.header),
        .data_out(result_if.data.header),
        `UNUSED_PIN(empty),
        `UNUSED_PIN(alm_empty),
        .full   (mdata_queue_full),
        `UNUSED_PIN(alm_full),
        `UNUSED_PIN(size)
    );

    wire [OFF_W-1:0] a_off = (OFF_W'(step_m) & OFF_W'(TCU_A_SUB_BLOCKS-1)) << LG_A_BS;
    wire [OFF_W-1:0] b_off = (OFF_W'(step_n) & OFF_W'(TCU_B_SUB_BLOCKS-1)) << LG_B_BS;

    wire [TCU_TC_M-1:0][TCU_TC_N-1:0][31:0] d_val;

    for (genvar i = 0; i < TCU_TC_M; ++i) begin : g_i
        for (genvar j = 0; j < TCU_TC_N; ++j) begin : g_j

            wire [TCU_TC_K-1:0][31:0] a_row = 32'(execute_if.data.rs1_data[a_off + i * TCU_TC_K +: TCU_TC_K][31:0]);
            wire [TCU_TC_K-1:0][31:0] b_cols = 32'(execute_if.data.rs2_data[b_off + j * TCU_TC_K +: TCU_TC_K][31:0]);
            wire [31:0] c_val = 32'(execute_if.data.rs3_data[i * TCU_TC_N + j]);
            wire [TCU_MAX_INPUTS-1:0] vld_mask = '1; // TODO: should connect to input source

            wire [3:0] fmt_s_r, fmt_d_r;
            wire [TCU_TC_K-1:0][31:0] a_row_r, b_col_r;
            wire [31:0] c_val_r;
            wire [TCU_MAX_INPUTS-1:0] vld_mask_r;

            `BUFFER_EX (
                {c_val_r, fmt_s_r, fmt_d_r, vld_mask_r},
                {c_val,   fmt_s,   fmt_d,   vld_mask},
                fedp_enable,
                0, // resetw
                1  // depth
            );

            // latch operands using per-element valid mask
            for (genvar k = 0; k < TCU_TC_K; ++k) begin : g_operands_latch
                for (genvar r = 0; r < TCU_MAX_ELT_RATIO; ++r) begin : g_elt
                    `BUFFER_EX (
                        {a_row_r[k][r * TCU_MIN_FMT_WIDTH +: TCU_MIN_FMT_WIDTH], b_col_r[k][r * TCU_MIN_FMT_WIDTH +: TCU_MIN_FMT_WIDTH]},
                        {a_row[k][r * TCU_MIN_FMT_WIDTH +: TCU_MIN_FMT_WIDTH],  b_cols[k][r * TCU_MIN_FMT_WIDTH +: TCU_MIN_FMT_WIDTH]},
                        fedp_enable && vld_mask[k * TCU_MAX_ELT_RATIO + r],
                        0, // resetw
                        1  // depth
                    );
                end
            end

        `ifdef TCU_TYPE_DPI
            VX_tcu_fedp_dpi #(
                .INSTANCE_ID (INSTANCE_ID),
                .LATENCY (FEDP_LATENCY),
                .N (TCU_TC_K)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .enable(fedp_enable),
                .vld_mask(vld_mask_r),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row(a_row_r),
                .b_cols(b_col_r),
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
                .vld_mask(vld_mask_r),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row(a_row_r),
                .b_cols(b_col_r),
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `elsif TCU_TYPE_DRL
            VX_tcu_fedp_drl #(
                .INSTANCE_ID (INSTANCE_ID),
                .LATENCY (FEDP_LATENCY),
                .N (TCU_TC_K)
            ) fedp (
                .clk   (clk),
                .reset (reset),
                .enable(fedp_enable),
                .vld_mask(vld_mask_r),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row(a_row_r),
                .b_cols(b_col_r),
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
                .vld_mask(vld_mask_r),
                .fmt_s (fmt_s_r),
                .fmt_d (fmt_d_r),
                .a_row(a_row_r),
                .b_cols(b_col_r),
                .c_val (c_val_r),
                .d_val (d_val[i][j])
            );
        `endif

        assign result_if.data.data[i * TCU_TC_N + j] = d_val[i][j];

        `ifdef DBG_TRACE_TCU
            always @(posedge clk) begin
                if (execute_if.valid && execute_if.ready) begin
                    `TRACE(3, ("%t: %s FEDP-enq: wid=%0d, i=%0d, j=%0d, m=%0d, n=%0d, a_row=", $time, INSTANCE_ID, execute_if.data.header.wid, i, j, step_m, step_n))
                    `TRACE_ARRAY1D(2, "0x%0h", a_row, TCU_TC_K)
                    `TRACE(3, (", b_cols="));
                    `TRACE_ARRAY1D(2, "0x%0h", b_cols, TCU_TC_K)
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
