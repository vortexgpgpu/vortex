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

module VX_issue_slice import VX_gpu_pkg::*, VX_trace_pkg::*; #(
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
    VX_dispatch_if.master   dispatch_if [`NUM_EX_UNITS]
);
    `UNUSED_PARAM (ISSUE_ID)

    VX_ibuffer_if ibuffer_if [PER_ISSUE_WARPS]();
    VX_scoreboard_if scoreboard_if();
    VX_operands_if operands_if();

    `RESET_RELAY (ibuf_reset, reset);
    `RESET_RELAY (scoreboard_reset, reset);
    `RESET_RELAY (operands_reset, reset);
    `RESET_RELAY (dispatch_reset, reset);

    VX_ibuffer #(
        .INSTANCE_ID ($sformatf("%s-ibuffer", INSTANCE_ID))
    ) ibuffer (
        .clk            (clk),
        .reset          (ibuf_reset),
     `ifdef PERF_ENABLE
        .perf_stalls    (issue_perf.ibf_stalls),
     `endif
        .decode_if      (decode_if),
        .ibuffer_if     (ibuffer_if)
    );

    VX_scoreboard #(
        .INSTANCE_ID ($sformatf("%s-scoreboard", INSTANCE_ID))
    ) scoreboard (
        .clk            (clk),
        .reset          (scoreboard_reset),
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
        .INSTANCE_ID ($sformatf("%s-operands", INSTANCE_ID))
    ) operands (
        .clk            (clk),
        .reset          (operands_reset),
     `ifdef PERF_ENABLE
        .perf_stalls    (issue_perf.opd_stalls),
     `endif
        .writeback_if   (writeback_if),
        .scoreboard_if  (scoreboard_if),
        .operands_if    (operands_if)
    );

    VX_dispatch #(
        .INSTANCE_ID ($sformatf("%s-dispatch", INSTANCE_ID))
    ) dispatch (
        .clk            (clk),
        .reset          (dispatch_reset),
    `ifdef PERF_ENABLE
        `UNUSED_PIN     (perf_stalls),
    `endif
        .operands_if    (operands_if),
        .dispatch_if    (dispatch_if)
    );

`ifdef DBG_SCOPE_ISSUE
    wire operands_if_fire = operands_if.valid && operands_if.ready;
    wire operands_if_not_ready = ~operands_if.ready;
    wire writeback_if_valid = writeback_if.valid;
    VX_scope_tap #(
        .SCOPE_ID (2),
        .TRIGGERW (4),
        .PROBEW (`UUID_WIDTH + `NUM_THREADS + `EX_BITS + `INST_OP_BITS +
            1 + `NR_BITS + (`NUM_THREADS * 3 * `XLEN) +
            `UUID_WIDTH + `NUM_THREADS + `NR_BITS + (`NUM_THREADS*`XLEN) + 1)
    ) scope_tap (
        .clk (clk),
        .reset (scope_reset),
        .start (1'b0),
        .stop (1'b0),
        .triggers ({
            reset,
            operands_if_fire,
            operands_if_not_ready,
            writeback_if_valid
        }),
        .probes ({
            operands_if.data.uuid,
            operands_if.data.tmask,
            operands_if.data.ex_type,
            operands_if.data.op_type,
            operands_if.data.wb,
            operands_if.data.rd,
            operands_if.data.rs1_data,
            operands_if.data.rs2_data,
            operands_if.data.rs3_data,
            writeback_if.data.uuid,
            writeback_if.data.tmask,
            writeback_if.data.rd,
            writeback_if.data.data,
            writeback_if.data.eop
        }),
        .bus_in (scope_bus_in),
        .bus_out (scope_bus_out)
    );
`else
    `SCOPE_IO_UNUSED()
`endif

`ifdef DBG_TRACE_PIPELINE
    always @(posedge clk) begin
        if (operands_if.valid && operands_if.ready) begin
            `TRACE(1, ("%d: %s wid=%0d, PC=0x%0h, ex=", $time, INSTANCE_ID, wis_to_wid(operands_if.data.wis, ISSUE_ID), {operands_if.data.PC, 1'b0}));
            trace_ex_type(1, operands_if.data.ex_type);
            `TRACE(1, (", op="));
            trace_ex_op(1, operands_if.data.ex_type, operands_if.data.op_type, operands_if.data.op_args);
            `TRACE(1, (", tmask=%b, wb=%b, rd=%0d, rs1_data=", operands_if.data.tmask, operands_if.data.wb, operands_if.data.rd));
            `TRACE_ARRAY1D(1, "0x%0h", operands_if.data.rs1_data, `NUM_THREADS);
            `TRACE(1, (", rs2_data="));
            `TRACE_ARRAY1D(1, "0x%0h", operands_if.data.rs2_data, `NUM_THREADS);
            `TRACE(1, (", rs3_data="));
            `TRACE_ARRAY1D(1, "0x%0h", operands_if.data.rs3_data, `NUM_THREADS);
            trace_op_args(1, operands_if.data.ex_type, operands_if.data.op_type, operands_if.data.op_args);
            `TRACE(1, (" (#%0d)\n", operands_if.data.uuid));
        end
    end
`endif

endmodule
