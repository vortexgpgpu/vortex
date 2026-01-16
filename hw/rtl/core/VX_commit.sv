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

module VX_commit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

    // inputs
    VX_commit_if.slave      commit_if [NUM_EX_UNITS * `ISSUE_WIDTH],

    // outputs
    VX_writeback_if.master  writeback_if  [`ISSUE_WIDTH],
    VX_commit_sched_if.master commit_sched_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam OUT_DATAW = $bits(commit_t);

    // commit arbitration

    VX_commit_if commit_arb_if[`ISSUE_WIDTH]();
    wire [`ISSUE_WIDTH-1:0] committed_warps;

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin : g_commit_arbs

        wire [NUM_EX_UNITS-1:0]            valid_in;
        wire [NUM_EX_UNITS-1:0][OUT_DATAW-1:0] data_in;
        wire [NUM_EX_UNITS-1:0]            ready_in;

        for (genvar j = 0; j < NUM_EX_UNITS; ++j) begin : g_data_in
            assign valid_in[j] = commit_if[j * `ISSUE_WIDTH + i].valid;
            assign data_in[j]  = commit_if[j * `ISSUE_WIDTH + i].data;
            assign commit_if[j * `ISSUE_WIDTH + i].ready = ready_in[j];
        end

        VX_stream_arb #(
            .NUM_INPUTS (NUM_EX_UNITS),
            .DATAW      (OUT_DATAW),
            .ARBITER    ("P"),
            .OUT_BUF    (1)
        ) commit_arb (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (valid_in),
            .ready_in   (ready_in),
            .data_in    (data_in),
            .data_out   (commit_arb_if[i].data),
            .valid_out  (commit_arb_if[i].valid),
            .ready_out  (commit_arb_if[i].ready),
            `UNUSED_PIN (sel_out)
        );

        wire commit_arb_fire = commit_arb_if[i].valid && commit_arb_if[i].ready;
        assign committed_warps[i] = commit_arb_fire && commit_arb_if[i].data.eop;
    end

    // notify scheduler
    wire [ISSUE_ISW_SIZEW-1:0] committed_warps_cnt, committed_warps_cnt_r;
    `POP_COUNT(committed_warps_cnt, committed_warps);
    `BUFFER(committed_warps_cnt_r, committed_warps_cnt);
    assign commit_sched_if.committed_warps_cnt = committed_warps_cnt_r;

    // Writeback

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin : g_writeback
        assign writeback_if[i].valid     = commit_arb_if[i].valid;
        assign writeback_if[i].data.uuid = commit_arb_if[i].data.uuid;
        assign writeback_if[i].data.wis  = wid_to_wis(commit_arb_if[i].data.wid);
        assign writeback_if[i].data.sid  = commit_arb_if[i].data.sid;
        assign writeback_if[i].data.PC   = commit_arb_if[i].data.PC;
        assign writeback_if[i].data.tmask= commit_arb_if[i].data.tmask;
        assign writeback_if[i].data.wb   = commit_arb_if[i].data.wb;
        assign writeback_if[i].data.wr_xregs = commit_arb_if[i].data.wr_xregs;
        assign writeback_if[i].data.rd   = commit_arb_if[i].data.rd;
        assign writeback_if[i].data.data = commit_arb_if[i].data.data;
        assign writeback_if[i].data.sop  = commit_arb_if[i].data.sop;
        assign writeback_if[i].data.eop  = commit_arb_if[i].data.eop;
        assign commit_arb_if[i].ready    = 1;
    end

`ifdef DBG_TRACE_PIPELINE
    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin : g_trace
        for (genvar j = 0; j < NUM_EX_UNITS; ++j) begin : g_j
            always @(posedge clk) begin
                if (commit_if[j * `ISSUE_WIDTH + i].valid && commit_if[j * `ISSUE_WIDTH + i].ready) begin
                    `TRACE(1, ("%t: %s: wid=%0d, sid=%0d, PC=0x%0h, ex=", $time, INSTANCE_ID, commit_if[j * `ISSUE_WIDTH + i].data.wid, commit_if[j * `ISSUE_WIDTH + i].data.sid, to_fullPC(commit_if[j * `ISSUE_WIDTH + i].data.PC)))
                    VX_trace_pkg::trace_ex_type(1, j);
                    `TRACE(1, (", tmask=%b, wb=%b, wr_xregs=%b, rd=%0d, sop=%b, eop=%b, data=", commit_if[j * `ISSUE_WIDTH + i].data.tmask, commit_if[j * `ISSUE_WIDTH + i].data.wb, commit_if[j * `ISSUE_WIDTH + i].data.wr_xregs, commit_if[j * `ISSUE_WIDTH + i].data.rd, commit_if[j * `ISSUE_WIDTH + i].data.sop, commit_if[j * `ISSUE_WIDTH + i].data.eop))
                    `TRACE_ARRAY1D(1, "0x%0h", commit_if[j * `ISSUE_WIDTH + i].data.data, `SIMD_WIDTH)
                    `TRACE(1, (" (#%0d)\n", commit_if[j * `ISSUE_WIDTH + i].data.uuid))
                end
            end
        end
    end
`endif

endmodule
