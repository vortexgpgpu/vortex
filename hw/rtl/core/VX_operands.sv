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
    parameter `STRING INSTANCE_ID = "",
    parameter ISSUE_ID = 0
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
    localparam NUM_OPDS  = NUM_SRC_OPDS + 1;
    localparam SCB_DATAW = UUID_WIDTH + ISSUE_WIS_W + `NUM_THREADS + PC_BITS + EX_BITS + INST_OP_BITS + INST_ARGS_BITS + NUM_OPDS + (REG_IDX_BITS * NUM_OPDS);
    localparam OPD_DATAW = UUID_WIDTH + ISSUE_WIS_W + SIMD_IDX_W + `SIMD_WIDTH + PC_BITS + EX_BITS + INST_OP_BITS + INST_ARGS_BITS + 1 + NR_BITS + (NUM_SRC_OPDS * `SIMD_WIDTH * `XLEN) + 1 + 1;

    VX_gpr_if per_opc_gpr_if[`NUM_OPCS]();
    VX_scoreboard_if per_opc_scoreboard_if[`NUM_OPCS]();
    VX_operands_if per_opc_operands_if[`NUM_OPCS]();

    wire [ISSUE_WIS_W-1:0] per_opc_pending_wis[`NUM_OPCS];
    wire [NUM_REGS-1:0] per_opc_pending_regs[`NUM_OPCS];

    // collector selection

    reg [`NUM_OPCS-1:0] select_opcs;
    always @(*) begin
        select_opcs = '1;
        // LSU cannot handle out of order LD/ST instructions: use same collector
        if (`NUM_OPCS > 1 && scoreboard_if.data.ex_type == EX_LSU) begin
            // select collector 0
            for (int i = 0; i < `NUM_OPCS; ++i) begin
                if (i != 0) select_opcs[i] = 0;
            end
        end
        // SFU cannot handle multiple inflight WCTL instructions
        // CSR access should be ordered as well: use same collector
        if (`NUM_OPCS > 1 && scoreboard_if.data.ex_type == EX_SFU) begin
            // select collector 1
            for (int i = 0; i < `NUM_OPCS; ++i) begin
                if (i != 1) select_opcs[i] = 0;
            end
        end
    end

`IGNORE_UNOPTFLAT_BEGIN
    `AOS_TO_ITF (per_opc_scoreboard, per_opc_scoreboard_if, `NUM_OPCS, SCB_DATAW)
`IGNORE_UNOPTFLAT_END

    VX_stream_arb #(
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (`NUM_OPCS),
        .DATAW       (SCB_DATAW),
        .ARBITER     ("P"),
        .OUT_BUF     (0)
    ) input_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (scoreboard_if.valid),
        .data_in   (scoreboard_if.data),
        .ready_in  (scoreboard_if.ready),
        .valid_out (per_opc_scoreboard_valid),
        .data_out  (per_opc_scoreboard_data),
        .ready_out (per_opc_scoreboard_ready & select_opcs),
        `UNUSED_PIN(sel_out)
    );

    for (genvar i = 0; i < `NUM_OPCS; ++i) begin : g_collectors
        wire [`UP(`NUM_OPCS-1)-1:0][ISSUE_WIS_W-1:0] pending_wis_in;
        wire [`UP(`NUM_OPCS-1)-1:0][NUM_REGS-1:0] pending_regs_in;

        for (genvar j = 1; j < `NUM_OPCS; ++j) begin : g_pending_wis_in
            localparam k = (i + j) % `NUM_OPCS;
            assign pending_wis_in[j-1] = per_opc_pending_wis[k];
            assign pending_regs_in[j-1] = per_opc_pending_regs[k];
        end

        VX_opc_unit #(
            .INSTANCE_ID (`SFORMATF(("%s-collector%0d", INSTANCE_ID, i))),
            .ISSUE_ID (ISSUE_ID)
        ) opc_unit (
            .clk          (clk),
            .reset        (reset),
            .pending_wis_in(pending_wis_in),
            .pending_regs_in(pending_regs_in),
            .pending_wis  (per_opc_pending_wis[i]),
            .pending_regs (per_opc_pending_regs[i]),
            .scoreboard_if(per_opc_scoreboard_if[i]),
            .gpr_if       (per_opc_gpr_if[i]),
            .operands_if  (per_opc_operands_if[i])
        );
    end

    VX_gpr_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-gpr", INSTANCE_ID))),
        .NUM_REQS    (`NUM_OPCS),
        .NUM_BANKS   (`NUM_GPR_BANKS)
    ) gpr_unit (
        .clk          (clk),
        .reset        (reset),
    `ifdef PERF_ENABLE
        .perf_stalls  (perf_stalls),
    `endif
        .writeback_if (writeback_if),
        .gpr_if       (per_opc_gpr_if)
    );

    `ITF_TO_AOS (per_opc_operands_if, per_opc_operands, `NUM_OPCS, OPD_DATAW)

    VX_stream_arb #(
        .NUM_INPUTS  (`NUM_OPCS),
        .NUM_OUTPUTS (1),
        .DATAW       (OPD_DATAW),
        .ARBITER     ("R"),
        .OUT_BUF     (3)
    ) output_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (per_opc_operands_valid),
        .data_in   (per_opc_operands_data),
        .ready_in  (per_opc_operands_ready),
        .valid_out (operands_if.valid),
        .data_out  (operands_if.data),
        .ready_out (operands_if.ready),
        `UNUSED_PIN(sel_out)
    );

endmodule
