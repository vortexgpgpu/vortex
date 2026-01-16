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
    parameter `STRING INSTANCE_ID = "",
    parameter ISSUE_ID = 0
) (
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output reg [PERF_CTR_BITS-1:0] perf_stalls,
`endif

    VX_writeback_if.slave   writeback_if,
    VX_ibuffer_if.slave     ibuffer_if [PER_ISSUE_WARPS],
    VX_scoreboard_if.master scoreboard_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (ISSUE_ID)
    `UNUSED_VAR (writeback_if.data.sop)

    localparam NUM_OPDS  = NUM_SRC_OPDS + 1;
    localparam IN_DATAW  = $bits(ibuffer_t);
    localparam OUT_DATAW = $bits(scoreboard_t) - ISSUE_WIS_W;

    VX_ibuffer_if staging_if [PER_ISSUE_WARPS]();
    wire [PER_ISSUE_WARPS-1:0] operands_ready;

`ifdef PERF_ENABLE
    wire [PER_ISSUE_WARPS-1:0] stg_valid_in;
    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_stg_valid_in
        assign stg_valid_in[w] = staging_if[w].valid;
    end

    wire perf_stall_per_cycle = |(stg_valid_in & ~operands_ready);

    always @(posedge clk) begin : g_perf_stalls
        if (reset) begin
            perf_stalls <= '0;
        end else begin
            perf_stalls <= perf_stalls + PERF_CTR_BITS'(perf_stall_per_cycle);
        end
    end
`endif

    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_stanging_bufs
        VX_pipe_buffer #(
            .DATAW (IN_DATAW)
        ) stanging_buf (
            .clk      (clk),
            .reset    (reset),
            .valid_in (ibuffer_if[w].valid),
            .data_in  (ibuffer_if[w].data),
            .ready_in (ibuffer_if[w].ready),
            .valid_out(staging_if[w].valid),
            .data_out (staging_if[w].data),
            .ready_out(staging_if[w].ready)
        );
    end

    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_scoreboard
        reg [NUM_REGS-1:0] inuse_regs, inuse_regs_n;
        reg [NUM_XREGS-1:0] inuse_xregs, inuse_xregs_n;
        wire [NUM_OPDS-1:0] operands_busy;

        wire ibuffer_fire = ibuffer_if[w].valid && ibuffer_if[w].ready;
        wire staging_fire = staging_if[w].valid && staging_if[w].ready;

        wire writeback_fire = writeback_if.valid
                           && (writeback_if.data.wis == ISSUE_WIS_W'(w))
                           && writeback_if.data.eop;

        wire [NUM_OPDS-1:0] [NUM_REGS_BITS-1:0] ibf_opds, stg_opds;
        assign ibf_opds = {ibuffer_if[w].data.rs3, ibuffer_if[w].data.rs2, ibuffer_if[w].data.rs1, ibuffer_if[w].data.rd};
        assign stg_opds = {staging_if[w].data.rs3, staging_if[w].data.rs2, staging_if[w].data.rs1, staging_if[w].data.rd};

        wire [NUM_OPDS-1:0] ibf_used_rs = {ibuffer_if[w].data.used_rs, ibuffer_if[w].data.wb};
        wire [NUM_OPDS-1:0] stg_used_rs = {staging_if[w].data.used_rs, staging_if[w].data.wb};

        // Special-register dependency masks
        wire [NUM_XREGS-1:0] ibf_xregs_mask = ibuffer_if[w].data.rd_xregs | ibuffer_if[w].data.wr_xregs;
        wire [NUM_XREGS-1:0] stg_xregs_mask = staging_if[w].data.rd_xregs | staging_if[w].data.wr_xregs;

        wire [NUM_OPDS-1:0][REG_TYPES-1:0][RV_REGS-1:0] ibf_opd_mask, stg_opd_mask;

        for (genvar i = 0; i < NUM_OPDS; ++i) begin : g_opd_masks
            for (genvar j = 0; j < REG_TYPES; ++j) begin : g_j
                assign ibf_opd_mask[i][j] = (1 << get_reg_idx(ibf_opds[i])) & {RV_REGS{ibf_used_rs[i] && get_reg_type(ibf_opds[i]) == j}};
                assign stg_opd_mask[i][j] = (1 << get_reg_idx(stg_opds[i])) & {RV_REGS{stg_used_rs[i] && get_reg_type(stg_opds[i]) == j}};
            end
        end

        always @(*) begin
            inuse_regs_n  = inuse_regs;
            inuse_xregs_n = inuse_xregs;
            if (writeback_fire) begin
                if (writeback_if.data.wb) begin
                    inuse_regs_n[writeback_if.data.rd] = 0; // release rd
                end
                inuse_xregs_n &= ~writeback_if.data.wr_xregs; // release special regs
            end
            if (staging_fire) begin
                if (staging_if[w].data.wb) begin
                    inuse_regs_n |= stg_opd_mask[0]; // reserve rd
                end
                inuse_xregs_n |= staging_if[w].data.wr_xregs; // reserve special regs
            end
        end

        wire [REG_TYPES-1:0][RV_REGS-1:0] in_use_mask;
        for (genvar i = 0; i < REG_TYPES; ++i) begin : g_in_use_mask
            wire [RV_REGS-1:0] ibf_reg_mask = ibf_opd_mask[0][i] | ibf_opd_mask[1][i] | ibf_opd_mask[2][i] | ibf_opd_mask[3][i];
            wire [RV_REGS-1:0] stg_reg_mask = stg_opd_mask[0][i] | stg_opd_mask[1][i] | stg_opd_mask[2][i] | stg_opd_mask[3][i];
            wire [RV_REGS-1:0] regs_mask = ibuffer_fire ? ibf_reg_mask : stg_reg_mask;
            assign in_use_mask[i] = inuse_regs_n[i * RV_REGS +: RV_REGS] & regs_mask;
        end

        wire [REG_TYPES-1:0] regs_busy;
        for (genvar i = 0; i < REG_TYPES; ++i) begin : g_regs_busy
            assign regs_busy[i] = (in_use_mask[i] != 0);
        end

        for (genvar i = 0; i < NUM_OPDS; ++i) begin : g_operands_busy
            wire [REG_TYPE_BITS-1:0] rtype = get_reg_type(stg_opds[i]);
            assign operands_busy[i] = (in_use_mask[rtype] & stg_opd_mask[i][rtype]) != 0;
        end


        wire [NUM_XREGS-1:0] xregs_mask = ibuffer_fire ? ibf_xregs_mask : stg_xregs_mask;
        wire [NUM_XREGS-1:0] xregs_busy = inuse_xregs_n & xregs_mask;

        reg operands_ready_r;

        always @(posedge clk) begin
            if (reset) begin
                inuse_regs  <= '0;
                inuse_xregs <= '0;
            end else begin
                inuse_regs <= inuse_regs_n;
                inuse_xregs <= inuse_xregs_n;
            end
            operands_ready_r <= (regs_busy == 0) && (xregs_busy == 0);
        end

        assign operands_ready[w] = operands_ready_r;

    `ifdef SIMULATION
        reg [31:0] timeout_ctr;

        always @(posedge clk) begin
            if (reset) begin
                timeout_ctr <= '0;
            end else begin
                if (staging_if[w].valid && ~staging_if[w].ready) begin
                `ifdef DBG_TRACE_PIPELINE
                    `TRACE(4, ("%t: *** %s-stall: wid=%0d, PC=0x%0h, tmask=%b, cycles=%0d, opds_busy=%b, xregs_busy=%b (#%0d)\n",
                        $time, INSTANCE_ID, w, to_fullPC(staging_if[w].data.PC), staging_if[w].data.tmask, timeout_ctr,
                        operands_busy, xregs_busy, staging_if[w].data.uuid))
                `endif
                    timeout_ctr <= timeout_ctr + 1;
                end else if (ibuffer_fire) begin
                    timeout_ctr <= '0;
                end
            end
        end

        `RUNTIME_ASSERT((timeout_ctr < STALL_TIMEOUT),
            ("timeout: wid=%0d, PC=0x%0h, tmask=%b, cycles=%0d, inuse=%b (#%0d)",
                w, to_fullPC(staging_if[w].data.PC), staging_if[w].data.tmask, timeout_ctr,
                operands_busy, staging_if[w].data.uuid))

        `RUNTIME_ASSERT(~(writeback_fire && writeback_if.data.wb) || inuse_regs[writeback_if.data.rd] != 0,
            ("invalid writeback register: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d (#%0d)",
                w, to_fullPC(writeback_if.data.PC), writeback_if.data.tmask, writeback_if.data.rd, writeback_if.data.uuid))

        `RUNTIME_ASSERT(~writeback_fire || ((writeback_if.data.wr_xregs & ~inuse_xregs) == '0),
            ("invalid writeback special register: wid=%0d, PC=0x%0h, tmask=%b, xregs=%b (#%0d)",
                w, to_fullPC(writeback_if.data.PC), writeback_if.data.tmask, writeback_if.data.wr_xregs, writeback_if.data.uuid))
    `endif

    end

    wire [PER_ISSUE_WARPS-1:0] arb_valid_in;
    wire [PER_ISSUE_WARPS-1:0][OUT_DATAW-1:0] arb_data_in;
    wire [PER_ISSUE_WARPS-1:0] arb_ready_in;

    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_arb_data_in
        assign arb_valid_in[w] = staging_if[w].valid && operands_ready[w];
        assign arb_data_in[w] = {
            staging_if[w].data.uuid,
            staging_if[w].data.tmask,
            staging_if[w].data.PC,
            staging_if[w].data.ex_type,
            staging_if[w].data.op_type,
            staging_if[w].data.op_args,
            staging_if[w].data.wb,
            staging_if[w].data.wr_xregs,
            staging_if[w].data.used_rs,
            staging_if[w].data.rd,
            staging_if[w].data.rs1,
            staging_if[w].data.rs2,
            staging_if[w].data.rs3
        };
        assign staging_if[w].ready = arb_ready_in[w] && operands_ready[w];
    end

    VX_stream_arb #(
        .NUM_INPUTS (PER_ISSUE_WARPS),
        .DATAW      (OUT_DATAW),
        .ARBITER    ("C"),
        .OUT_BUF    (3)
    ) out_arb (
        .clk      (clk),
        .reset    (reset),
        .valid_in (arb_valid_in),
        .ready_in (arb_ready_in),
        .data_in  (arb_data_in),
        .data_out ({
            scoreboard_if.data.uuid,
            scoreboard_if.data.tmask,
            scoreboard_if.data.PC,
            scoreboard_if.data.ex_type,
            scoreboard_if.data.op_type,
            scoreboard_if.data.op_args,
            scoreboard_if.data.wb,
            scoreboard_if.data.wr_xregs,
            scoreboard_if.data.used_rs,
            scoreboard_if.data.rd,
            scoreboard_if.data.rs1,
            scoreboard_if.data.rs2,
            scoreboard_if.data.rs3
        }),
        .valid_out (scoreboard_if.valid),
        .ready_out (scoreboard_if.ready),
        .sel_out   (scoreboard_if.data.wis)
    );

endmodule
