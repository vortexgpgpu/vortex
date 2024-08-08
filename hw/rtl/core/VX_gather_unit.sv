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
    VX_commit_if.slave  commit_in_if [BLOCK_SIZE],

    // outputs
    VX_commit_if.master commit_out_if [`ISSUE_WIDTH]

);
    `STATIC_ASSERT (`IS_DIVISBLE(`ISSUE_WIDTH, BLOCK_SIZE), ("invalid parameter"))
    `STATIC_ASSERT (`IS_DIVISBLE(`NUM_THREADS, NUM_LANES), ("invalid parameter"))
    localparam BLOCK_SIZE_W = `LOG2UP(BLOCK_SIZE);
    localparam PID_BITS     = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam PID_WIDTH    = `UP(PID_BITS);
    localparam DATAW        = `UUID_WIDTH + `NW_WIDTH + NUM_LANES + `PC_BITS + 1 + `NR_BITS + NUM_LANES * `XLEN + PID_WIDTH + 1 + 1;
    localparam DATA_WIS_OFF = DATAW - (`UUID_WIDTH + `NW_WIDTH);

    wire [BLOCK_SIZE-1:0] commit_in_valid;
    wire [BLOCK_SIZE-1:0][DATAW-1:0] commit_in_data;
    wire [BLOCK_SIZE-1:0] commit_in_ready;
    wire [BLOCK_SIZE-1:0][ISSUE_ISW_W-1:0] commit_in_isw;

    for (genvar i = 0; i < BLOCK_SIZE; ++i) begin
        assign commit_in_valid[i] = commit_in_if[i].valid;
        assign commit_in_data[i] = commit_in_if[i].data;
        assign commit_in_if[i].ready = commit_in_ready[i];
        if (BLOCK_SIZE != `ISSUE_WIDTH) begin
            if (BLOCK_SIZE != 1) begin
                assign commit_in_isw[i] = {commit_in_data[i][DATA_WIS_OFF+BLOCK_SIZE_W +: (ISSUE_ISW_W-BLOCK_SIZE_W)], BLOCK_SIZE_W'(i)};
            end else begin
                assign commit_in_isw[i] = commit_in_data[i][DATA_WIS_OFF +: ISSUE_ISW_W];
            end
        end else begin
            assign commit_in_isw[i] = BLOCK_SIZE_W'(i);
        end
    end

    reg [`ISSUE_WIDTH-1:0] commit_out_valid;
    reg [`ISSUE_WIDTH-1:0][DATAW-1:0] commit_out_data;
    wire [`ISSUE_WIDTH-1:0] commit_out_ready;

    always @(*) begin
        commit_out_valid = '0;
        for (integer i = 0; i < `ISSUE_WIDTH; ++i) begin
            commit_out_data[i] = 'x;
        end
        for (integer i = 0; i < BLOCK_SIZE; ++i) begin
            commit_out_valid[commit_in_isw[i]] = commit_in_valid[i];
            commit_out_data[commit_in_isw[i]] = commit_in_data[i];
        end
    end
    for (genvar i = 0; i < BLOCK_SIZE; ++i) begin
        assign commit_in_ready[i] = commit_out_ready[commit_in_isw[i]];
    end

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin
        VX_commit_if #(
            .NUM_LANES (NUM_LANES)
        ) commit_tmp_if();

        VX_elastic_buffer #(
            .DATAW   (DATAW),
            .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
            .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF))
        ) out_buf (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (commit_out_valid[i]),
            .ready_in   (commit_out_ready[i]),
            .data_in    (commit_out_data[i]),
            .data_out   (commit_tmp_if.data),
            .valid_out  (commit_tmp_if.valid),
            .ready_out  (commit_tmp_if.ready)
        );

        logic [`NUM_THREADS-1:0] commit_tmask_r;
        logic [`NUM_THREADS-1:0][`XLEN-1:0] commit_data_r;
        if (PID_BITS != 0) begin
            always @(*) begin
                commit_tmask_r = '0;
                commit_data_r  = 'x;
                for (integer j = 0; j < NUM_LANES; ++j) begin
                    commit_tmask_r[commit_tmp_if.data.pid * NUM_LANES + j] = commit_tmp_if.data.tmask[j];
                    commit_data_r[commit_tmp_if.data.pid * NUM_LANES + j] = commit_tmp_if.data.data[j];
                end
            end
        end else begin
            assign commit_tmask_r = commit_tmp_if.data.tmask;
            assign commit_data_r = commit_tmp_if.data.data;
        end

        assign commit_out_if[i].valid = commit_tmp_if.valid;
        assign commit_out_if[i].data = {
            commit_tmp_if.data.uuid,
            commit_tmp_if.data.wid,
            commit_tmask_r,
            commit_tmp_if.data.PC,
            commit_tmp_if.data.wb,
            commit_tmp_if.data.rd,
            commit_data_r,
            1'b0, // PID
            commit_tmp_if.data.sop,
            commit_tmp_if.data.eop
        };
        assign commit_tmp_if.ready = commit_out_if[i].ready;
    end

endmodule
