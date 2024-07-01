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

module VX_scoreboard import VX_gpu_pkg::*; #(
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output reg [`PERF_CTR_BITS-1:0] perf_scb_stalls,
    output reg [`PERF_CTR_BITS-1:0] perf_units_uses [`NUM_EX_UNITS],
    output reg [`PERF_CTR_BITS-1:0] perf_sfu_uses [`NUM_SFU_UNITS],
`endif

    VX_writeback_if.slave   writeback_if [`ISSUE_WIDTH],
    VX_ibuffer_if.slave     ibuffer_if [`NUM_WARPS],
    VX_scoreboard_if.master scoreboard_if [`ISSUE_WIDTH]
);
    `UNUSED_PARAM (CORE_ID)
    localparam DATAW = `UUID_WIDTH + `NUM_THREADS + `PC_BITS + `EX_BITS + `INST_OP_BITS + `INST_ARGS_BITS + (`NR_BITS * 4) + 1;

`ifdef PERF_ENABLE
    reg [`NUM_WARPS-1:0][`NUM_EX_UNITS-1:0] perf_inuse_units_per_cycle;
    wire [`NUM_EX_UNITS-1:0] perf_units_per_cycle, perf_units_per_cycle_r;

    reg [`NUM_WARPS-1:0][`NUM_SFU_UNITS-1:0] perf_inuse_sfu_per_cycle;
    wire [`NUM_SFU_UNITS-1:0] perf_sfu_per_cycle, perf_sfu_per_cycle_r;

    wire [`NUM_WARPS-1:0] perf_issue_stalls_per_cycle;
    wire [`CLOG2(`NUM_WARPS+1)-1:0] perf_stalls_per_cycle, perf_stalls_per_cycle_r;

    `POP_COUNT(perf_stalls_per_cycle, perf_issue_stalls_per_cycle);

    VX_reduce #(
        .DATAW_IN (`NUM_EX_UNITS),
        .N  (`NUM_WARPS),
        .OP ("|")
    ) perf_units_reduce (
        .data_in  (perf_inuse_units_per_cycle),
        .data_out (perf_units_per_cycle)
    );

    VX_reduce #(
        .DATAW_IN (`NUM_SFU_UNITS),
        .N  (`NUM_WARPS),
        .OP ("|")
    ) perf_sfu_reduce (
        .data_in  (perf_inuse_sfu_per_cycle),
        .data_out (perf_sfu_per_cycle)
    );

    `BUFFER(perf_stalls_per_cycle_r, perf_stalls_per_cycle);
    `BUFFER_EX(perf_units_per_cycle_r, perf_units_per_cycle, 1'b1, `CDIV(`NUM_WARPS, `MAX_FANOUT));
    `BUFFER_EX(perf_sfu_per_cycle_r, perf_sfu_per_cycle, 1'b1, `CDIV(`NUM_WARPS, `MAX_FANOUT));

    always @(posedge clk) begin
        if (reset) begin
            perf_scb_stalls <= '0;
        end else begin
            perf_scb_stalls <= perf_scb_stalls + `PERF_CTR_BITS'(perf_stalls_per_cycle_r);
        end
    end

    for (genvar i = 0; i < `NUM_EX_UNITS; ++i) begin
        always @(posedge clk) begin
            if (reset) begin
                perf_units_uses[i] <= '0;
            end else begin
                perf_units_uses[i] <= perf_units_uses[i] + `PERF_CTR_BITS'(perf_units_per_cycle_r[i]);
            end
        end
    end

    for (genvar i = 0; i < `NUM_SFU_UNITS; ++i) begin
        always @(posedge clk) begin
            if (reset) begin
                perf_sfu_uses[i] <= '0;
            end else begin
                perf_sfu_uses[i] <= perf_sfu_uses[i] + `PERF_CTR_BITS'(perf_sfu_per_cycle_r[i]);
            end
        end
    end
`endif

    VX_ibuffer_if staging_if [`NUM_WARPS]();
    wire [`NUM_WARPS-1:0][3:0] staging_opds_busy;

    for (genvar i = 0; i < `NUM_WARPS; ++i) begin
        VX_elastic_buffer #(
            .DATAW (DATAW),
            .SIZE  (1)
        ) stanging_buf (
            .clk      (clk),
            .reset    (reset),
            .valid_in (ibuffer_if[i].valid),
            .data_in  (ibuffer_if[i].data),
            .ready_in (ibuffer_if[i].ready),
            .valid_out(staging_if[i].valid),
            .data_out (staging_if[i].data),
            .ready_out(staging_if[i].ready)
        );
    end

    for (genvar i = 0; i < `NUM_WARPS; ++i) begin
        reg [`NUM_REGS-1:0] inuse_regs;

        reg [3:0] operands_busy_r, operands_busy_n;

        localparam iw = i % `ISSUE_WIDTH;
        localparam wis = i / `ISSUE_WIDTH;

        wire ibuffer_fire = ibuffer_if[i].valid && ibuffer_if[i].ready;

        wire staging_fire = staging_if[i].valid && staging_if[i].ready;

        wire writeback_fire = writeback_if[iw].valid
                           && (writeback_if[iw].data.wis == ISSUE_WIS_W'(wis))
                           && writeback_if[iw].data.eop;

    `ifdef PERF_ENABLE
        reg [`NUM_REGS-1:0][`EX_WIDTH-1:0] inuse_units;
        reg [`NUM_REGS-1:0][`SFU_WIDTH-1:0] inuse_sfu;

        reg [`SFU_WIDTH-1:0] sfu_type;
        always @(*) begin
            case (staging_if[i].data.op_type)
            `INST_SFU_CSRRW,
            `INST_SFU_CSRRS,
            `INST_SFU_CSRRC: sfu_type = `SFU_CSRS;
            default: sfu_type = `SFU_WCTL;
            endcase
        end

        always @(*) begin
            perf_inuse_units_per_cycle[i] = '0;
            perf_inuse_sfu_per_cycle[i] = '0;
            if (staging_if[i].valid) begin
                if (operands_busy_r[0]) begin
                    perf_inuse_units_per_cycle[i][inuse_units[staging_if[i].data.rd]] = 1;
                    if (inuse_units[staging_if[i].data.rd] == `EX_SFU) begin
                        perf_inuse_sfu_per_cycle[i][inuse_sfu[staging_if[i].data.rd]] = 1;
                    end
                end
                if (operands_busy_r[1]) begin
                    perf_inuse_units_per_cycle[i][inuse_units[staging_if[i].data.rs1]] = 1;
                    if (inuse_units[staging_if[i].data.rs1] == `EX_SFU) begin
                        perf_inuse_sfu_per_cycle[i][inuse_sfu[staging_if[i].data.rs1]] = 1;
                    end
                end
                if (operands_busy_r[2]) begin
                    perf_inuse_units_per_cycle[i][inuse_units[staging_if[i].data.rs2]] = 1;
                    if (inuse_units[staging_if[i].data.rs2] == `EX_SFU) begin
                        perf_inuse_sfu_per_cycle[i][inuse_sfu[staging_if[i].data.rs2]] = 1;
                    end
                end
                if (operands_busy_r[3]) begin
                    perf_inuse_units_per_cycle[i][inuse_units[staging_if[i].data.rs3]] = 1;
                    if (inuse_units[staging_if[i].data.rs3] == `EX_SFU) begin
                        perf_inuse_sfu_per_cycle[i][inuse_sfu[staging_if[i].data.rs3]] = 1;
                    end
                end
            end
        end
        assign perf_issue_stalls_per_cycle[i] = staging_if[i].valid && ~staging_if[i].ready;
    `endif

        always @(*) begin
            operands_busy_n = operands_busy_r;
            if (ibuffer_fire) begin
                operands_busy_n = {
                    inuse_regs[ibuffer_if[i].data.rs3],
                    inuse_regs[ibuffer_if[i].data.rs2],
                    inuse_regs[ibuffer_if[i].data.rs1],
                    inuse_regs[ibuffer_if[i].data.rd]
                };
            end
            if (writeback_fire) begin
                if (ibuffer_fire) begin
                    if (writeback_if[iw].data.rd == ibuffer_if[i].data.rd) begin
                        operands_busy_n[0] = 0;
                    end
                    if (writeback_if[iw].data.rd == ibuffer_if[i].data.rs1) begin
                        operands_busy_n[1] = 0;
                    end
                    if (writeback_if[iw].data.rd == ibuffer_if[i].data.rs2) begin
                        operands_busy_n[2] = 0;
                    end
                    if (writeback_if[iw].data.rd == ibuffer_if[i].data.rs3) begin
                        operands_busy_n[3] = 0;
                    end
                end else begin
                    if (writeback_if[iw].data.rd == staging_if[i].data.rd) begin
                        operands_busy_n[0] = 0;
                    end
                    if (writeback_if[iw].data.rd == staging_if[i].data.rs1) begin
                        operands_busy_n[1] = 0;
                    end
                    if (writeback_if[iw].data.rd == staging_if[i].data.rs2) begin
                        operands_busy_n[2] = 0;
                    end
                    if (writeback_if[iw].data.rd == staging_if[i].data.rs3) begin
                        operands_busy_n[3] = 0;
                    end
                end
            end
            if (staging_fire && staging_if[i].data.wb) begin
                if (staging_if[i].data.rd == ibuffer_if[i].data.rd) begin
                    operands_busy_n[0] = 1;
                end
                if (staging_if[i].data.rd == ibuffer_if[i].data.rs1) begin
                    operands_busy_n[1] = 1;
                end
                if (staging_if[i].data.rd == ibuffer_if[i].data.rs2) begin
                    operands_busy_n[2] = 1;
                end
                if (staging_if[i].data.rd == ibuffer_if[i].data.rs3) begin
                    operands_busy_n[3] = 1;
                end
            end
        end

        always @(posedge clk) begin
            if (reset) begin
                inuse_regs <= '0;
            end else begin
                if (writeback_fire) begin
                    inuse_regs[writeback_if[iw].data.rd] <= 0;
                end
                if (staging_fire && staging_if[i].data.wb) begin
                    inuse_regs[staging_if[i].data.rd] <= 1;
                end
            end
            operands_busy_r <= operands_busy_n;
        `ifdef PERF_ENABLE
            if (staging_fire && staging_if[i].data.wb) begin
                inuse_units[staging_if[i].data.rd] <= staging_if[i].data.ex_type;
                if (staging_if[i].data.ex_type == `EX_SFU) begin
                    inuse_sfu[staging_if[i].data.rd] <= sfu_type;
                end
            end
        `endif
        end

        assign staging_opds_busy[i] = operands_busy_r;

    `ifdef SIMULATION
        reg [31:0] timeout_ctr;

        always @(posedge clk) begin
            if (reset) begin
                timeout_ctr <= '0;
            end else begin
                if (staging_if[i].valid && ~staging_if[i].ready) begin
                `ifdef DBG_TRACE_PIPELINE
                    `TRACE(3, ("%d: *** core%0d-scoreboard-stall: wid=%0d, PC=0x%0h, tmask=%b, cycles=%0d, inuse=%b (#%0d)\n",
                        $time, CORE_ID, i, {staging_if[i].data.PC, 1'b0}, staging_if[i].data.tmask, timeout_ctr,
                        operands_busy_r, staging_if[i].data.uuid));
                `endif
                    timeout_ctr <= timeout_ctr + 1;
                end else if (ibuffer_fire) begin
                    timeout_ctr <= '0;
                end
            end
        end

        `RUNTIME_ASSERT((timeout_ctr < `STALL_TIMEOUT),
                        ("%t: *** core%0d-scoreboard-timeout: wid=%0d, PC=0x%0h, tmask=%b, cycles=%0d, inuse=%b (#%0d)",
                            $time, CORE_ID, i, {staging_if[i].data.PC, 1'b0}, staging_if[i].data.tmask, timeout_ctr,
                            operands_busy_r, staging_if[i].data.uuid));

        `RUNTIME_ASSERT(~writeback_fire || inuse_regs[writeback_if[iw].data.rd] != 0,
            ("%t: *** core%0d: invalid writeback register: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d (#%0d)",
                $time, CORE_ID, i, {writeback_if[iw].data.PC, 1'b0}, writeback_if[iw].data.tmask, writeback_if[iw].data.rd, writeback_if[iw].data.uuid));
    `endif

    end

    `RESET_RELAY (arb_reset, reset);

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin
        wire [ISSUE_RATIO-1:0] valid_in;
        wire [ISSUE_RATIO-1:0][DATAW-1:0] data_in;
        wire [ISSUE_RATIO-1:0] ready_in;

        for (genvar j = 0; j < ISSUE_RATIO; ++j) begin
            wire operands_ready = ~(| staging_opds_busy[j * `ISSUE_WIDTH + i]);
            assign valid_in[j]  = staging_if[j * `ISSUE_WIDTH + i].valid && operands_ready;
            assign data_in[j]   = staging_if[j * `ISSUE_WIDTH + i].data;
            assign staging_if[j * `ISSUE_WIDTH + i].ready = ready_in[j] && operands_ready;
        end

        VX_stream_arb #(
            .NUM_INPUTS (ISSUE_RATIO),
            .DATAW      (DATAW),
            .ARBITER    ("R"),
            .OUT_BUF    (2)
        ) out_arb (
            .clk      (clk),
            .reset    (arb_reset),
            .valid_in (valid_in),
            .ready_in (ready_in),
            .data_in  (data_in),
            .data_out ({
                scoreboard_if[i].data.uuid,
                scoreboard_if[i].data.tmask,
                scoreboard_if[i].data.PC,
                scoreboard_if[i].data.ex_type,
                scoreboard_if[i].data.op_type,
                scoreboard_if[i].data.op_args,
                scoreboard_if[i].data.wb,
                scoreboard_if[i].data.rd,
                scoreboard_if[i].data.rs1,
                scoreboard_if[i].data.rs2,
                scoreboard_if[i].data.rs3
            }),
            .valid_out (scoreboard_if[i].valid),
            .ready_out (scoreboard_if[i].ready),
            .sel_out   (scoreboard_if[i].data.wis)
        );
    end

endmodule
