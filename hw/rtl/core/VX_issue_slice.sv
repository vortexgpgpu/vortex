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

module VX_issue_slice import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter ISSUE_ID = 0
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output issue_perf_t     issue_perf,
`endif

    VX_decode_if.slave      decode_if,
    VX_writeback_if.slave   writeback_if,
    VX_dispatch_if.master   dispatch_if [NUM_EX_UNITS],
    VX_issue_sched_if.master issue_sched_if
);
    `UNUSED_PARAM (ISSUE_ID)

    VX_ibuffer_if ibuffer_if [PER_ISSUE_WARPS]();
    VX_scoreboard_if scoreboard_if();
    VX_operands_if operands_if();

    VX_ibuffer #(
        .INSTANCE_ID (`SFORMATF(("%s-ibuffer", INSTANCE_ID))),
        .ISSUE_ID (ISSUE_ID)
    ) ibuffer (
        .clk            (clk),
        .reset          (reset),
     `ifdef PERF_ENABLE
        .perf_stalls    (issue_perf.ibf_stalls),
     `endif
        .decode_if      (decode_if),
        .ibuffer_if     (ibuffer_if)
    );

    VX_scoreboard #(
        .INSTANCE_ID (`SFORMATF(("%s-scoreboard", INSTANCE_ID))),
        .ISSUE_ID (ISSUE_ID)
    ) scoreboard (
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        .perf_stalls    (issue_perf.scb_stalls),
        .perf_units_uses(issue_perf.units_uses),
        .perf_sfu_uses  (issue_perf.sfu_uses),
    `endif
        .writeback_if   (writeback_if),
        .ibuffer_if     (ibuffer_if),
        .scoreboard_if  (scoreboard_if)
    );

    VX_operands #(
        .INSTANCE_ID (`SFORMATF(("%s-operands", INSTANCE_ID))),
        .ISSUE_ID (ISSUE_ID)
    ) operands (
        .clk            (clk),
        .reset          (reset),
     `ifdef PERF_ENABLE
        .perf_stalls    (issue_perf.opd_stalls),
     `endif
        .writeback_if   (writeback_if),
        .scoreboard_if  (scoreboard_if),
        .operands_if    (operands_if)
    );

    VX_dispatch #(
        .INSTANCE_ID (`SFORMATF(("%s-dispatch", INSTANCE_ID))),
        .ISSUE_ID (ISSUE_ID)
    ) dispatch (
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        `UNUSED_PIN     (perf_stalls),
    `endif
        .operands_if    (operands_if),
        .dispatch_if    (dispatch_if)
    );

    // notify scheduler
    assign issue_sched_if.valid = operands_if.valid && operands_if.ready && operands_if.data.sop;
    assign issue_sched_if.wis = operands_if.data.wis;

`ifdef SCOPE
`ifdef DBG_SCOPE_ISSUE
    `SCOPE_IO_SWITCH (1);
    wire decode_fire = decode_if.valid && decode_if.ready;
    wire operands_fire = operands_if.valid && operands_if.ready;
    wire reset_negedge;
    `NEG_EDGE (reset_negedge, reset);
    `SCOPE_TAP_EX (0, 2, 4, 3, (
            UUID_WIDTH + NW_WIDTH + `NUM_THREADS + PC_BITS + EX_BITS + INST_OP_BITS + 1 + NUM_REGS_BITS * 4 +
            UUID_WIDTH + ISSUE_WIS_W + `SIMD_WIDTH + PC_BITS + EX_BITS + INST_OP_BITS + 1 + NUM_REGS_BITS + (3 * `XLEN) +
            UUID_WIDTH + ISSUE_WIS_W + `SIMD_WIDTH + NUM_REGS_BITS + (`SIMD_WIDTH * `XLEN) + 1
        ), {
            decode_if.valid,
            decode_if.ready,
            operands_if.valid,
            operands_if.ready
        }, {
            decode_fire,
            operands_fire,
            writeback_if.valid // ack-free
        }, {
            decode_if.data.uuid,
            decode_if.data.wid,
            decode_if.data.tmask,
            decode_if.data.PC,
            decode_if.data.ex_type,
            decode_if.data.op_type,
            decode_if.data.wb,
            decode_if.data.rd,
            decode_if.data.rs1,
            decode_if.data.rs2,
            decode_if.data.rs3,
            operands_if.data.uuid,
            operands_if.data.wis,
            operands_if.data.tmask,
            operands_if.data.PC,
            operands_if.data.ex_type,
            operands_if.data.op_type,
            operands_if.data.wb,
            operands_if.data.rd,
            operands_if.data.rs1_data[0],
            operands_if.data.rs2_data[0],
            operands_if.data.rs3_data[0],
            writeback_if.data.uuid,
            writeback_if.data.wis,
            writeback_if.data.tmask,
            writeback_if.data.rd,
            writeback_if.data.data,
            writeback_if.data.eop
        },
        reset_negedge, 1'b0, 4096
    );
`else
    `SCOPE_IO_UNUSED(0)
`endif
`endif

`ifdef CHIPSCOPE
`ifdef DBG_SCOPE_ISSUE
    ila_issue ila_issue_inst (
        .clk    (clk),
        .probe0 ({decode_if.valid, decode_if.data, decode_if.ready}),
        .probe1 ({scoreboard_if.valid, scoreboard_if.data, scoreboard_if.ready}),
        .probe2 ({operands_if.valid, operands_if.data, operands_if.ready}),
        .probe3 ({writeback_if.valid, writeback_if.data})
    );
`endif
`endif

`ifdef DBG_TRACE_PIPELINE
    for (genvar i = 0; i < PER_ISSUE_WARPS; ++i) begin : g_ibuffer_trace
        localparam wid = wis_to_wid(ISSUE_WIS_W'(i), ISSUE_ID);
        always @(posedge clk) begin
            if (ibuffer_if[i].valid && ibuffer_if[i].ready) begin
                `TRACE(1, ("%t: %s-ibuffer: wid=%0d, PC=0x%0h, ex=", $time, INSTANCE_ID, wid, to_fullPC(ibuffer_if[i].data.PC)))
                VX_trace_pkg::trace_ex_type(1, ibuffer_if[i].data.ex_type);
                `TRACE(1, (", op="))
                VX_trace_pkg::trace_ex_op(1, ibuffer_if[i].data.ex_type, ibuffer_if[i].data.op_type, ibuffer_if[i].data.op_args);
                `TRACE(1, (", tmask=%b, wb=%b, used_rs=%b, rd=", ibuffer_if[i].data.tmask, ibuffer_if[i].data.wb, ibuffer_if[i].data.used_rs))
                VX_trace_pkg::trace_reg_idx(1, ibuffer_if[i].data.rd);
                `TRACE(1, (", rs1="))
                VX_trace_pkg::trace_reg_idx(1, ibuffer_if[i].data.rs1);
                `TRACE(1, (", rs2="))
                VX_trace_pkg::trace_reg_idx(1, ibuffer_if[i].data.rs2);
                `TRACE(1, (", rs3="))
                VX_trace_pkg::trace_reg_idx(1, ibuffer_if[i].data.rs3);
                `TRACE(1, (", "))
                VX_trace_pkg::trace_op_args(1, ibuffer_if[i].data.ex_type, ibuffer_if[i].data.op_type, ibuffer_if[i].data.op_args);
                `TRACE(1, (" (#%0d)\n", ibuffer_if[i].data.uuid))
            end
        end
    end

    always @(posedge clk) begin
        if (operands_if.valid && operands_if.ready) begin
            `TRACE(1, ("%t: %s-dispatch: wid=%0d, sid=%0d, PC=0x%0h, ex=", $time, INSTANCE_ID, wis_to_wid(operands_if.data.wis, ISSUE_ID), operands_if.data.sid, to_fullPC(operands_if.data.PC)))
            VX_trace_pkg::trace_ex_type(1, operands_if.data.ex_type);
            `TRACE(1, (", op="))
            VX_trace_pkg::trace_ex_op(1, operands_if.data.ex_type, operands_if.data.op_type, operands_if.data.op_args);
            `TRACE(1, (", tmask=%b, wb=%b, rd=%0d, rs1_data=", operands_if.data.tmask, operands_if.data.wb, operands_if.data.rd))
            `TRACE_ARRAY1D(1, "0x%0h", operands_if.data.rs1_data, `SIMD_WIDTH)
            `TRACE(1, (", rs2_data="))
            `TRACE_ARRAY1D(1, "0x%0h", operands_if.data.rs2_data, `SIMD_WIDTH)
            `TRACE(1, (", rs3_data="))
            `TRACE_ARRAY1D(1, "0x%0h", operands_if.data.rs3_data, `SIMD_WIDTH)
            `TRACE(1, (", "))
           VX_trace_pkg::trace_op_args(1, operands_if.data.ex_type, operands_if.data.op_type, operands_if.data.op_args);
            `TRACE(1, (", sop=%b, eop=%b (#%0d)\n", operands_if.data.sop, operands_if.data.eop, operands_if.data.uuid))
        end
    end
`endif

endmodule
