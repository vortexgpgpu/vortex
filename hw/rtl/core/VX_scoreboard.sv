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

    input wire [NUM_EX_UNITS-1:0] fu_release,
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
    localparam OUT_BUF   = 3; // Use skid buffer (SIZE=2, OUT_REG=1)

    // Per-FU dispatch credits: spent at issue, reclaimed on FU accept, so a
    // credit covers ops still in operand collection (not yet at the queue).
    wire [NUM_EX_UNITS-1:0] fu_issue;
    wire [NUM_EX_UNITS-1:0] fu_goingfull;

    // going-full (not full): a 1-slot guard band keeps outstanding <= queue
    // depth despite the registered suppress lag, so an issued op never stalls
    // in the shared operand-collection path and HoL-blocks another FU.
    for (genvar e = 0; e < NUM_EX_UNITS; ++e) begin : g_fu_goingfull
        VX_pending_size #(
            .SIZE (DISPATCH_QSIZE)
        ) fu_pending (
            .clk        (clk),
            .reset      (reset),
            .incr       (fu_issue[e]),
            .decr       (fu_release[e]),
            `UNUSED_PIN (empty),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (full),
            .alm_full   (fu_goingfull[e]),
            `UNUSED_PIN (size)
        );
    end

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

        // Writeback release feeds wb_inuse_regs; the staging reserve is added on
        // top to form inuse_regs_n, which the busy check reads directly.
        reg [NUM_REGS-1:0]  wb_inuse_regs;
        reg [NUM_XREGS-1:0] wb_inuse_xregs;
        always @(*) begin
            wb_inuse_regs  = inuse_regs;
            wb_inuse_xregs = inuse_xregs;
            if (writeback_fire) begin
                if (writeback_if.data.wb) begin
                    wb_inuse_regs[writeback_if.data.rd] = 0; // release rd
                end
                wb_inuse_xregs &= ~writeback_if.data.wr_xregs; // release special regs
            end
        end

        always @(*) begin
            inuse_regs_n  = wb_inuse_regs;
            inuse_xregs_n = wb_inuse_xregs;
            if (staging_fire) begin
                if (staging_if[w].data.wb) begin
                    inuse_regs_n |= stg_opd_mask[0]; // reserve rd
                end
                inuse_xregs_n |= staging_if[w].data.wr_xregs; // reserve special regs
            end
        end

        // in_use_mask = inuse_regs_n masked by the operand-dependency set
        // (the ibuffer instr on a fire, else the staging instr), shared by the
        // regs_busy reduction and the per-operand operands_busy check.
        wire [REG_TYPES-1:0][RV_REGS-1:0] in_use_mask;
        for (genvar i = 0; i < REG_TYPES; ++i) begin : g_in_use_mask
            wire [RV_REGS-1:0] ibf_reg_mask = ibf_opd_mask[0][i] | ibf_opd_mask[1][i] | ibf_opd_mask[2][i] | ibf_opd_mask[3][i];
            wire [RV_REGS-1:0] stg_reg_mask = stg_opd_mask[0][i] | stg_opd_mask[1][i] | stg_opd_mask[2][i] | stg_opd_mask[3][i];
            wire [RV_REGS-1:0] regs_mask = ibuffer_fire ? ibf_reg_mask : stg_reg_mask;
            assign in_use_mask[i] = inuse_regs_n[i * RV_REGS +: RV_REGS] & regs_mask;
        end

        wire [REG_TYPES-1:0] regs_busy;
        for (genvar i = 0; i < REG_TYPES; ++i) begin : g_regs_busy
            assign regs_busy[i] = (| in_use_mask[i]);
        end

        for (genvar i = 0; i < NUM_OPDS; ++i) begin : g_operands_busy
            wire [REG_TYPE_BITS-1:0] rtype = get_reg_type(stg_opds[i]);
            assign operands_busy[i] = | (in_use_mask[rtype] & stg_opd_mask[i][rtype]);
        end

        wire [NUM_XREGS-1:0] xregs_mask = ibuffer_fire ? ibf_xregs_mask : stg_xregs_mask;
        wire xregs_busy = | (inuse_xregs_n & xregs_mask);

        wire [EX_BITS-1:0] ex_sel = ibuffer_fire ? ibuffer_if[w].data.ex_type : staging_if[w].data.ex_type;
        reg operands_ready_r;

        // Readiness folds data hazards, FU-congestion and FU-lock into one flop.
        // fu_locked_n is the next-state, look-ahead-aligned with ex_sel.
        wire fu_lock_sel = ibuffer_fire ? ibuffer_if[w].data.fu_lock : staging_if[w].data.fu_lock;
        wire data_ready  = ~((|regs_busy) || xregs_busy);
        wire operands_ready_n = data_ready
                             && ~fu_goingfull[ex_sel]
                             && ~(fu_locked_n[ex_sel] && fu_lock_sel);

        always @(posedge clk) begin
            if (reset) begin
                inuse_regs  <= '0;
                inuse_xregs <= '0;
            end else begin
                inuse_regs  <= inuse_regs_n;
                inuse_xregs <= inuse_xregs_n;
            end
            operands_ready_r <= operands_ready_n;
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

    reg  [NUM_EX_UNITS-1:0] fu_locked, fu_locked_n;

    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_arb_data_in
        // operands_ready carries data-hazard + FU-congestion + FU-lock (all folded
        // into operands_ready_r), so the request is just valid & operands_ready.
        assign arb_valid_in[w] = staging_if[w].valid && operands_ready[w];

        assign arb_data_in[w] = {
            staging_if[w].data.uuid,
            staging_if[w].data.cta_id,
            staging_if[w].data.tmask,
            staging_if[w].data.PC,
            staging_if[w].data.ex_type,
            staging_if[w].data.op_type,
            staging_if[w].data.op_args,
            staging_if[w].data.wb,
            staging_if[w].data.wr_xregs,
            staging_if[w].data.used_rs,
            staging_if[w].data.rd,
            staging_if[w].data.bytesel,
            staging_if[w].data.rs1,
            staging_if[w].data.rs2,
            staging_if[w].data.rs3
        };
        assign staging_if[w].ready = arb_ready_in[w] && operands_ready[w];
    end


    // Cyclic arbiter; STICKY holds the grant under backpressure. requests is a
    // pure flop (suppress already folded into operands_ready).

    localparam LOG_NUM_REQS = `LOG2UP(PER_ISSUE_WARPS);

    wire                    arb_valid;
    wire [LOG_NUM_REQS-1:0] arb_index;
    wire [PER_ISSUE_WARPS-1:0] arb_onehot;
    wire                    arb_ready;

    VX_cyclic_arbiter #(
        .NUM_REQS (PER_ISSUE_WARPS),
        .STICKY   (1) // Greedy
    ) out_arb (
        .clk          (clk),
        .reset        (reset),
        .requests     (arb_valid_in),
        .grant_valid  (arb_valid),
        .grant_index  (arb_index),
        .grant_onehot (arb_onehot),
        .grant_ready  (arb_ready)
    );

    wire valid_out_w;
    wire [OUT_DATAW-1:0] data_out_w;
    wire ready_out_w;

    assign valid_out_w = arb_valid;
    assign data_out_w  = arb_data_in[arb_index];

    for (genvar i = 0; i < PER_ISSUE_WARPS; ++i) begin : g_arb_ready_in
        assign arb_ready_in[i] = ready_out_w && arb_onehot[i];
    end

    assign arb_ready = ready_out_w;

    // FU lock: prevent warp interleaving during multi-uop sequences.
    // 10=acquire (first uop), 00=middle, 01=release (last uop), 11=default.

    wire issue_fire = valid_out_w && ready_out_w;

    wire [PER_ISSUE_WARPS-1:0][EX_BITS-1:0] staging_ex_vec;
    wire [PER_ISSUE_WARPS-1:0] staging_fu_lock_vec;
    wire [PER_ISSUE_WARPS-1:0] staging_fu_unlock_vec;
    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_staging_fu_lock
        assign staging_ex_vec[w]        = staging_if[w].data.ex_type;
        assign staging_fu_lock_vec[w]   = staging_if[w].data.fu_lock;
        assign staging_fu_unlock_vec[w] = staging_if[w].data.fu_unlock;
    end

    wire [EX_BITS-1:0] issue_ex = staging_ex_vec[arb_index];

    for (genvar e = 0; e < NUM_EX_UNITS; ++e) begin : g_fu_issue
        assign fu_issue[e] = issue_fire && (issue_ex == EX_BITS'(e));
    end

    // fu_locked next-state: the granted warp (arb_onehot is one-hot) acquires or
    // releases its FU. Built from arb_onehot -- not issue_ex -- so its arbiter
    // dependence matches ibuffer_fire's depth, keeping the fu_locked_n term folded
    // into operands_ready_n off the critical path (parallel to the look-ahead select).
    always @(*) begin
        fu_locked_n = fu_locked;
        for (integer i = 0; i < PER_ISSUE_WARPS; i = i + 1) begin
            if (arb_onehot[i] && ready_out_w) begin
                if (staging_fu_lock_vec[i] && ~staging_fu_unlock_vec[i]) begin
                    fu_locked_n[staging_ex_vec[i]] = 1'b1;
                end else if (~staging_fu_lock_vec[i] && staging_fu_unlock_vec[i]) begin
                    fu_locked_n[staging_ex_vec[i]] = 1'b0;
                end
            end
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            fu_locked <= '0;
        end else begin
            fu_locked <= fu_locked_n;
        end
    end

    VX_elastic_buffer #(
        .DATAW   (LOG_NUM_REQS + OUT_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF)),
        .LUTRAM  (`TO_OUT_BUF_LUTRAM(OUT_BUF))
    ) out_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (valid_out_w),
        .ready_in  (ready_out_w),
        .data_in   ({arb_index, data_out_w}),
        .data_out  ({
            scoreboard_if.data.wis,
            scoreboard_if.data.uuid,
            scoreboard_if.data.cta_id,
            scoreboard_if.data.tmask,
            scoreboard_if.data.PC,
            scoreboard_if.data.ex_type,
            scoreboard_if.data.op_type,
            scoreboard_if.data.op_args,
            scoreboard_if.data.wb,
            scoreboard_if.data.wr_xregs,
            scoreboard_if.data.used_rs,
            scoreboard_if.data.rd,
            scoreboard_if.data.bytesel,
            scoreboard_if.data.rs1,
            scoreboard_if.data.rs2,
            scoreboard_if.data.rs3
        }),
        .valid_out (scoreboard_if.valid),
        .ready_out (scoreboard_if.ready)
    );

endmodule
