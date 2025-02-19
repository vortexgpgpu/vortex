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
    VX_operands_if.master   operands_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam NUM_OPDS = NUM_SRC_OPDS + 1;
    localparam SCB_DATAW = UUID_WIDTH + ISSUE_WIS_W + `NUM_THREADS + PC_BITS + EX_BITS + INST_OP_BITS + INST_ARGS_BITS + NUM_OPDS + (REG_IDX_BITS * NUM_OPDS);
    localparam OPD_DATAW = UUID_WIDTH + ISSUE_WIS_W + SIMD_IDX_W + `SIMD_WIDTH + PC_BITS + EX_BITS + INST_OP_BITS + INST_ARGS_BITS + 1 + NR_BITS + (NUM_SRC_OPDS * `SIMD_WIDTH * `XLEN);

    VX_opc_if opc_if[`NUM_OPCS]();
    VX_scoreboard_if per_opc_scoreboard_if[`NUM_OPCS]();
    VX_operands_if per_opc_operands_if[`NUM_OPCS]();

    wire [ISSUE_WIS_W-1:0] per_opc_wis[`NUM_OPCS];
    wire [SIMD_IDX_W-1:0] per_opc_sid[`NUM_OPCS];
    wire [NUM_REGS-1:0] per_opc_pending_regs[`NUM_OPCS];

    `AOS_TO_ITF (per_opc_scoreboard, per_opc_scoreboard_if, `NUM_OPCS, SCB_DATAW)

    VX_stream_arb #(
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (`NUM_OPCS),
        .DATAW       (SCB_DATAW)
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
            .sid          (per_opc_sid[i]),
            .wis          (per_opc_wis[i]),
            .pending_regs (per_opc_pending_regs[i]),
            .scoreboard_if(per_opc_scoreboard_if[i]),
            .opc_if       (opc_if[i]),
            .operands_if  (per_opc_operands_if[i])
        );
    end

    reg [`NUM_OPCS-1:0][NUM_REGS-1:0] per_opc_pending_regs_sel;
    for (genvar i = 0; i < `NUM_OPCS; ++i) begin : g_per_opc_pending_regs_sel
        wire opc_match = (per_opc_sid[i] == writeback_if.data.sid) && (per_opc_wis[i] == writeback_if.data.wis);
        assign per_opc_pending_regs_sel[i] = per_opc_pending_regs[i] & {NUM_REGS{opc_match}};
    end

    reg [NUM_REGS-1:0] opc_pending_regs;
    VX_reduce_tree #(
        .DATAW_IN  (NUM_REGS),
        .N         (`NUM_OPCS),
        .OP        ("|")
    ) reduce_opc_pending_regs (
        .data_in  (per_opc_pending_regs_sel),
        .data_out (opc_pending_regs)
    );

    wire war_dp_check = (opc_pending_regs[writeback_if.data.rd] == 0);

    VX_writeback_if writeback_if_s();
    assign writeback_if_s.valid = writeback_if.valid && war_dp_check;
    assign writeback_if_s.data = writeback_if.data;
    assign writeback_if.ready = war_dp_check;
    `UNUSED_VAR (writeback_if_s.ready)

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
        .writeback_if (writeback_if_s),
        .opc_if       (opc_if)
    );

    `ITF_TO_AOS (per_opc_operands_if, per_opc_operands, `NUM_OPCS, OPD_DATAW)

    VX_stream_arb #(
        .NUM_INPUTS  (`NUM_OPCS),
        .NUM_OUTPUTS (1),
        .DATAW       (OPD_DATAW)
    ) operands_arb (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (per_opc_operands_valid),
        .data_in    (per_opc_operands_data),
        .ready_in   (per_opc_operands_ready),
        .valid_out  (operands_if.valid),
        .data_out   (operands_if.data),
        .ready_out  (operands_if.ready),
        `UNUSED_PIN(sel_out)
    );

endmodule
