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
`include "VX_trace.vh"

module VX_dispatch import VX_gpu_pkg::*; #(
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output wire [`PERF_CTR_BITS-1:0] perf_stalls [`NUM_EX_UNITS],
`endif
    // inputs
    VX_operands_if.slave    operands_if [`ISSUE_WIDTH],

    // outputs
    VX_dispatch_if.master   dispatch_if [`NUM_EX_UNITS * `ISSUE_WIDTH]
);
    `UNUSED_PARAM (CORE_ID)

    localparam DATAW = `UUID_WIDTH + ISSUE_WIS_W + `NUM_THREADS + `PC_BITS + `INST_OP_BITS + `INST_ARGS_BITS + 1 + `NR_BITS + (3 * `NUM_THREADS * `XLEN) + `NT_WIDTH;

    wire [`NUM_THREADS-1:0][`NT_WIDTH-1:0] tids;
    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign tids[i] = `NT_WIDTH'(i);
    end

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin

        wire [`NT_WIDTH-1:0] last_active_tid;

        VX_find_first #(
            .N (`NUM_THREADS),
            .DATAW (`NT_WIDTH),
            .REVERSE (1)
        ) last_tid_select (
            .valid_in (operands_if[i].data.tmask),
            .data_in  (tids),
            .data_out (last_active_tid),
            `UNUSED_PIN (valid_out)
        );

        wire [`NUM_EX_UNITS-1:0] operands_reset;

        `RESET_RELAY (buf_reset, reset);

        for (genvar j = 0; j < `NUM_EX_UNITS; ++j) begin
            VX_elastic_buffer #(
                .DATAW   (DATAW),
                .SIZE    (2),
                .OUT_REG (2)
            ) buffer (
                .clk        (clk),
                .reset      (buf_reset),
                .valid_in   (operands_if[i].valid && (operands_if[i].data.ex_type == j)),
                .ready_in   (operands_reset[j]),
                .data_in    (`TO_DISPATCH_DATA(operands_if[i].data, last_active_tid)),
                .data_out   (dispatch_if[j * `ISSUE_WIDTH + i].data),
                .valid_out  (dispatch_if[j * `ISSUE_WIDTH + i].valid),
                .ready_out  (dispatch_if[j * `ISSUE_WIDTH + i].ready)
            );
        end

        assign operands_if[i].ready = operands_reset[operands_if[i].data.ex_type];
    end

`ifdef PERF_ENABLE
    wire [`NUM_EX_UNITS-1:0] perf_unit_stalls_per_cycle, perf_unit_stalls_per_cycle_r;
    reg [`ISSUE_WIDTH-1:0][`NUM_EX_UNITS-1:0] perf_issue_unit_stalls_per_cycle;
    reg [`NUM_EX_UNITS-1:0][`PERF_CTR_BITS-1:0] perf_stalls_r;

    for (genvar i=0; i < `ISSUE_WIDTH; ++i) begin
        always @(*) begin
            perf_issue_unit_stalls_per_cycle[i] = '0;
            if (operands_if[i].valid && ~operands_if[i].ready) begin
                perf_issue_unit_stalls_per_cycle[i][operands_if[i].data.ex_type] = 1;
            end
        end
    end

    VX_reduce #(
        .DATAW_IN (`NUM_EX_UNITS),
        .N  (`ISSUE_WIDTH),
        .OP ("|")
    ) reduce (
        .data_in (perf_issue_unit_stalls_per_cycle),
        .data_out (perf_unit_stalls_per_cycle)
    );

    `BUFFER(perf_unit_stalls_per_cycle_r, perf_unit_stalls_per_cycle);

    for (genvar i = 0; i < `NUM_EX_UNITS; ++i) begin
        always @(posedge clk) begin
            if (reset) begin
                perf_stalls_r[i] <= '0;
            end else begin
                perf_stalls_r[i] <= perf_stalls_r[i] + `PERF_CTR_BITS'(perf_unit_stalls_per_cycle_r[i]);
            end
        end
    end

    for (genvar i=0; i < `NUM_EX_UNITS; ++i) begin
        assign perf_stalls[i] = perf_stalls_r[i];
    end
`endif

`ifdef DBG_TRACE_PIPELINE
    for (genvar i=0; i < `ISSUE_WIDTH; ++i) begin
        always @(posedge clk) begin
            if (operands_if[i].valid && operands_if[i].ready) begin
                `TRACE(1, ("%d: core%0d-issue: wid=%0d, PC=0x%0h, ex=", $time, CORE_ID, wis_to_wid(operands_if[i].data.wis, i), {operands_if[i].data.PC, 1'b0}));
                trace_ex_type(1, operands_if[i].data.ex_type);
                `TRACE(1, (", op="));
                trace_ex_op(1, operands_if[i].data.ex_type, operands_if[i].data.op_type, operands_if[i].data.op_args);
                `TRACE(1, (", tmask=%b, wb=%b, rd=%0d, rs1_data=", operands_if[i].data.tmask, operands_if[i].data.wb, operands_if[i].data.rd));
                `TRACE_ARRAY1D(1, "0x%0h", operands_if[i].data.rs1_data, `NUM_THREADS);
                `TRACE(1, (", rs2_data="));
                `TRACE_ARRAY1D(1, "0x%0h", operands_if[i].data.rs2_data, `NUM_THREADS);
                `TRACE(1, (", rs3_data="));
                `TRACE_ARRAY1D(1, "0x%0h", operands_if[i].data.rs3_data, `NUM_THREADS);
                trace_op_args(1, operands_if[i].data.ex_type, operands_if[i].data.op_type, operands_if[i].data.op_args);
                `TRACE(1, (" (#%0d)\n", operands_if[i].data.uuid));
            end
        end
    end
`endif

endmodule
