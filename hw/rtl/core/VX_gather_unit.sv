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

module VX_gather_unit import VX_gpu_pkg::*; #(
    parameter BLOCK_SIZE = 1,
    parameter NUM_LANES  = 1,
    parameter OUT_BUF    = 0
) (
    input  wire         clk,
    input  wire         reset,

    // inputs
    VX_result_if.slave  result_if [BLOCK_SIZE],

    // outputs
    VX_commit_if.master commit_if [`ISSUE_WIDTH]
);
    `STATIC_ASSERT (`IS_DIVISBLE(`ISSUE_WIDTH, BLOCK_SIZE), ("invalid parameter"))
    `STATIC_ASSERT (`IS_DIVISBLE(`SIMD_WIDTH, NUM_LANES), ("invalid parameter"))
    localparam BLOCK_SIZE_W = `LOG2UP(BLOCK_SIZE);
    localparam NUM_PACKETS  = `SIMD_WIDTH / NUM_LANES;
    localparam LPID_BITS    = `CLOG2(NUM_PACKETS);
    localparam LPID_WIDTH   = `UP(LPID_BITS);
    localparam GPID_BITS    = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam GPID_WIDTH   = `UP(GPID_BITS);
    localparam DATAW        = UUID_WIDTH + NW_WIDTH + NUM_LANES + PC_BITS + 1 + NUM_REGS_BITS + NUM_LANES * `XLEN + GPID_WIDTH + 1 + 1;
    localparam DATA_WIS_OFF = DATAW - (UUID_WIDTH + NW_WIDTH);

    `DECL_RESULT_T (result_t, NUM_LANES);

    wire [BLOCK_SIZE-1:0] result_in_valid;
    wire [BLOCK_SIZE-1:0][DATAW-1:0] result_in_data;
    wire [BLOCK_SIZE-1:0] result_in_ready;
    wire [BLOCK_SIZE-1:0][ISSUE_ISW_W-1:0] result_in_isw;

    for (genvar i = 0; i < BLOCK_SIZE; ++i) begin : g_commit_in
        assign result_in_valid[i] = result_if[i].valid;
        assign result_in_data[i]  = result_if[i].data;
        assign result_if[i].ready = result_in_ready[i];
        if (BLOCK_SIZE != `ISSUE_WIDTH) begin : g_result_in_isw_partial
            if (BLOCK_SIZE != 1) begin : g_block
                assign result_in_isw[i] = {result_in_data[i][DATA_WIS_OFF+BLOCK_SIZE_W +: (ISSUE_ISW_W-BLOCK_SIZE_W)], BLOCK_SIZE_W'(i)};
            end else begin : g_no_block
                assign result_in_isw[i] = result_in_data[i][DATA_WIS_OFF +: ISSUE_ISW_W];
            end
        end else begin : g_result_in_isw_full
            assign result_in_isw[i] = BLOCK_SIZE_W'(i);
        end
    end

    reg [`ISSUE_WIDTH-1:0] result_out_valid;
    reg [`ISSUE_WIDTH-1:0][DATAW-1:0] result_out_data;
    wire [`ISSUE_WIDTH-1:0] result_out_ready;

    always @(*) begin
        result_out_valid = '0;
        for (integer i = 0; i < `ISSUE_WIDTH; ++i) begin
            result_out_data[i] = 'x;
        end
        for (integer i = 0; i < BLOCK_SIZE; ++i) begin
            result_out_valid[result_in_isw[i]] = result_in_valid[i];
            result_out_data[result_in_isw[i]] = result_in_data[i];
        end
    end

    for (genvar i = 0; i < BLOCK_SIZE; ++i) begin : g_result_in_ready
        assign result_in_ready[i] = result_out_ready[result_in_isw[i]];
    end

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin: g_out_bufs
        VX_result_if #(
            .data_t (result_t)
        ) result_tmp_if();

        VX_elastic_buffer #(
            .DATAW   (DATAW),
            .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
            .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF))
        ) out_buf (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (result_out_valid[i]),
            .ready_in   (result_out_ready[i]),
            .data_in    (result_out_data[i]),
            .data_out   (result_tmp_if.data),
            .valid_out  (result_tmp_if.valid),
            .ready_out  (result_tmp_if.ready)
        );

        logic [SIMD_IDX_W-1:0] commit_sid_w;
        logic [`SIMD_WIDTH-1:0] commit_tmask_w;
        logic [`SIMD_WIDTH-1:0][`XLEN-1:0] commit_data_w;

        if (LPID_BITS != 0) begin : g_lpid
            logic [LPID_WIDTH-1:0] lpid;
            if (SIMD_COUNT != 1) begin : g_simd
                assign {commit_sid_w, lpid} = result_tmp_if.data.pid;
            end else begin : g_no_simd
                assign commit_sid_w = 0;
                assign lpid = result_tmp_if.data.pid;
            end
            always @(*) begin
                commit_tmask_w = '0;
                commit_data_w  = 'x;
                for (integer j = 0; j < NUM_LANES; ++j) begin
                    commit_tmask_w[lpid * NUM_LANES + j] = result_tmp_if.data.tmask[j];
                    commit_data_w[lpid * NUM_LANES + j] = result_tmp_if.data.data[j];
                end
            end
        end else begin : g_no_lpid
            assign commit_sid_w   = result_tmp_if.data.pid;
            assign commit_tmask_w = result_tmp_if.data.tmask;
            assign commit_data_w  = result_tmp_if.data.data;
        end

        assign commit_if[i].valid = result_tmp_if.valid;
        assign commit_if[i].data = {
            result_tmp_if.data.uuid,
            result_tmp_if.data.wid,
            commit_sid_w,
            commit_tmask_w,
            result_tmp_if.data.PC,
            result_tmp_if.data.wb,
            result_tmp_if.data.rd,
            commit_data_w,
            result_tmp_if.data.sop,
            result_tmp_if.data.eop
        };
        assign result_tmp_if.ready = commit_if[i].ready;
    end

endmodule
