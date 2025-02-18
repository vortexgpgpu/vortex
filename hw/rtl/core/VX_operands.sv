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

// reset all GPRs in debug mode
`ifdef SIMULATION
`ifndef NDEBUG
`define GPR_RESET
`endif
`endif

module VX_operands import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] perf_stalls,
`endif

    VX_writeback_if.slave   writeback_if,
    VX_scoreboard_if.slave  scoreboard_if,
    VX_operands_if.master   operands_if [`NUM_OPCS]
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam NUM_OPDS = NUM_SRC_OPDS + 1;
    localparam SB_DATAW = UUID_WIDTH + ISSUE_WIS_W + `NUM_THREADS + PC_BITS + EX_BITS + INST_OP_BITS + INST_ARGS_BITS + NUM_OPDS + (REG_IDX_BITS * NUM_OPDS);

    VX_opc_if opc_if[`NUM_OPCS]();
    VX_scoreboard_if per_opc_scoreboard_if[`NUM_OPCS]();
    wire [ISSUE_WIS_W-1:0] per_opc_wis[`NUM_OPCS];
    wire [NUM_REGS-1:0] per_opc_pending_regs_n[`NUM_OPCS];

    `AOS_TO_ITF (per_opc_scoreboard, per_opc_scoreboard_if, `NUM_OPCS, SB_DATAW)

    VX_stream_arb #(
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (`NUM_OPCS),
        .DATAW       (SB_DATAW)
    ) scboard_arb (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (scoreboard_if.valid),
        .data_in    (scoreboard_if.data),
        .ready_in   (scoreboard_if.ready),
        .valid_out  (per_opc_scoreboard_valid),
        .data_out   (per_opc_scoreboard_data),
        .ready_out  (per_opc_scoreboard_ready),
        `UNUSED_PIN(sel_out)
    );

    for (genvar i = 0; i < `NUM_OPCS; ++i) begin : g_opc_units
        VX_opc_unit #(
            .INSTANCE_ID (INSTANCE_ID)
        ) opc_unit (
            .clk          (clk),
            .reset        (reset),
            .wis          (per_opc_wis[i]),
            .pending_regs_n(per_opc_pending_regs_n[i]),
            .scoreboard_if(per_opc_scoreboard_if[i]),
            .opc_if       (opc_if[i]),
            .operands_if  (operands_if[i])
        );
    end

    VX_gpr_unit #(
        .INSTANCE_ID (INSTANCE_ID),
        .NUM_REQS    (`NUM_OPCS),
        .NUM_BANKS   (`NUM_GPR_BANKS)
    ) gpr_unit (
        .clk          (clk),
        .reset        (reset),
    `ifdef PERF_ENABLE
        .perf_stalls  (perf_stalls),
    `endif
        .writeback_if (writeback_if),
        .opc_if       (opc_if)
    );

endmodule
