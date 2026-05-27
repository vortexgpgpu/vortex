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

module VX_dispatch import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter ISSUE_ID = 0
) (
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] perf_stalls [NUM_EX_UNITS],
`endif
    // inputs
    VX_operands_if.slave    operands_if,

    // outputs
    VX_dispatch_if.master   dispatch_if [NUM_EX_UNITS]
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (ISSUE_ID)

    localparam OUT_DATAW = $bits(dispatch_t);

    wire [NUM_EX_UNITS-1:0] operands_ready_in;
    assign operands_if.ready = operands_ready_in[operands_if.data.ex_type];

    for (genvar i = 0; i < NUM_EX_UNITS; ++i) begin : g_buffers
        VX_elastic_buffer #(
            .DATAW   (OUT_DATAW),
            .SIZE    (2),
            .OUT_REG (1)
        ) buffer (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (operands_if.valid && (operands_if.data.ex_type == EX_BITS'(i))),
            .ready_in   (operands_ready_in[i]),
            .data_in    ({
                operands_if.data.uuid,
                operands_if.data.wis,
                operands_if.data.sid,
                operands_if.data.tmask,
                operands_if.data.PC,
                operands_if.data.op_type,
                operands_if.data.op_args,
                operands_if.data.wb,
                operands_if.data.rd,
                operands_if.data.rs1_data,
                operands_if.data.rs2_data,
                operands_if.data.rs3_data,
                operands_if.data.sop,
                operands_if.data.eop
            }),
            .data_out   (dispatch_if[i].data),
            .valid_out  (dispatch_if[i].valid),
            .ready_out  (dispatch_if[i].ready)
        );
    end

`ifdef PERF_ENABLE
    reg [NUM_EX_UNITS-1:0][PERF_CTR_BITS-1:0] perf_stalls_r;

    wire operands_if_stall = operands_if.valid && ~operands_if.ready;

    for (genvar i = 0; i < NUM_EX_UNITS; ++i) begin : g_perf_stalls
        always @(posedge clk) begin
            if (reset) begin
                perf_stalls_r[i] <= '0;
            end else begin
                perf_stalls_r[i] <= perf_stalls_r[i] + PERF_CTR_BITS'(operands_if_stall && operands_if.data.ex_type == EX_BITS'(i));
            end
        end
        assign perf_stalls[i] = perf_stalls_r[i];
    end
`endif

endmodule
