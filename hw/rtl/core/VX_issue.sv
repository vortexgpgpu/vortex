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

module VX_issue import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output issue_perf_t     issue_perf,
`endif

    VX_decode_if.slave      decode_if,
    VX_writeback_if.slave   writeback_if [`ISSUE_WIDTH],
    VX_dispatch_if.master   dispatch_if [`NUM_EX_UNITS * `ISSUE_WIDTH]
);
    `STATIC_ASSERT ((`ISSUE_WIDTH <= `NUM_WARPS), ("invalid parameter"))

`ifdef PERF_ENABLE
    issue_perf_t per_issue_perf [`ISSUE_WIDTH];
    `PERF_COUNTER_ADD (issue_perf, per_issue_perf, ibf_stalls, `PERF_CTR_BITS, `ISSUE_WIDTH, (`ISSUE_WIDTH > 2))
    `PERF_COUNTER_ADD (issue_perf, per_issue_perf, scb_stalls, `PERF_CTR_BITS, `ISSUE_WIDTH, (`ISSUE_WIDTH > 2))
    `PERF_COUNTER_ADD (issue_perf, per_issue_perf, opd_stalls, `PERF_CTR_BITS, `ISSUE_WIDTH, (`ISSUE_WIDTH > 2))
    for (genvar i = 0; i < `NUM_EX_UNITS; ++i) begin
        `PERF_COUNTER_ADD (issue_perf, per_issue_perf, units_uses[i], `PERF_CTR_BITS, `ISSUE_WIDTH, (`ISSUE_WIDTH > 2))
    end
    for (genvar i = 0; i < `NUM_SFU_UNITS; ++i) begin
        `PERF_COUNTER_ADD (issue_perf, per_issue_perf, sfu_uses[i], `PERF_CTR_BITS, `ISSUE_WIDTH, (`ISSUE_WIDTH > 2))
    end
`endif

    wire [ISSUE_ISW_W-1:0] decode_isw = wid_to_isw(decode_if.data.wid);
    wire [ISSUE_WIS_W-1:0] decode_wis = wid_to_wis(decode_if.data.wid);

    wire [`ISSUE_WIDTH-1:0] decode_ready_in;
    assign decode_if.ready = decode_ready_in[decode_isw];

    `SCOPE_IO_SWITCH (`ISSUE_WIDTH)

    for (genvar issue_id = 0; issue_id < `ISSUE_WIDTH; ++issue_id) begin : issue_slices
        VX_decode_if #(
            .NUM_WARPS (PER_ISSUE_WARPS)
        ) per_issue_decode_if();

        VX_dispatch_if per_issue_dispatch_if[`NUM_EX_UNITS]();

        assign per_issue_decode_if.valid = decode_if.valid && (decode_isw == ISSUE_ISW_W'(issue_id));
        assign per_issue_decode_if.data.uuid = decode_if.data.uuid;
        assign per_issue_decode_if.data.wid = decode_wis;
        assign per_issue_decode_if.data.tmask = decode_if.data.tmask;
        assign per_issue_decode_if.data.PC = decode_if.data.PC;
        assign per_issue_decode_if.data.ex_type = decode_if.data.ex_type;
        assign per_issue_decode_if.data.op_type = decode_if.data.op_type;
        assign per_issue_decode_if.data.op_args = decode_if.data.op_args;
        assign per_issue_decode_if.data.wb = decode_if.data.wb;
        assign per_issue_decode_if.data.rd = decode_if.data.rd;
        assign per_issue_decode_if.data.rs1 = decode_if.data.rs1;
        assign per_issue_decode_if.data.rs2 = decode_if.data.rs2;
        assign per_issue_decode_if.data.rs3 = decode_if.data.rs3;
        assign decode_ready_in[issue_id] = per_issue_decode_if.ready;
    `ifndef L1_ENABLE
        assign decode_if.ibuf_pop[issue_id * PER_ISSUE_WARPS +: PER_ISSUE_WARPS] = per_issue_decode_if.ibuf_pop;
    `endif

        VX_issue_slice #(
            .INSTANCE_ID ($sformatf("%s%0d", INSTANCE_ID, issue_id)),
            .ISSUE_ID (issue_id)
        ) issue_slice (
            `SCOPE_IO_BIND(issue_id)
            .clk          (clk),
            .reset        (reset),
        `ifdef PERF_ENABLE
            .issue_perf   (per_issue_perf[issue_id]),
        `endif
            .decode_if    (per_issue_decode_if),
            .writeback_if (writeback_if[issue_id]),
            .dispatch_if  (per_issue_dispatch_if)
        );

        // Assign transposed dispatch_if
        for (genvar ex_id = 0; ex_id < `NUM_EX_UNITS; ++ex_id) begin
            `ASSIGN_VX_IF(dispatch_if[ex_id * `ISSUE_WIDTH + issue_id], per_issue_dispatch_if[ex_id]);
        end
     end

endmodule
