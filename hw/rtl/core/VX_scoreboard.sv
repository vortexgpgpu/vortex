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
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output reg [PERF_CTR_BITS-1:0] perf_stalls,
`endif

    input wire [NUM_EX_UNITS-1:0] fu_release,
    VX_writeback_if.slave   writeback_if,

    // per warp: the instruction buffer's head entry and the one behind it
    input wire [PER_ISSUE_WARPS-1:0]      head_valid,
    input ibuf_hz_t [PER_ISSUE_WARPS-1:0] head_hz,
    input wire [PER_ISSUE_WARPS-1:0]      next_valid,
    input ibuf_hz_t [PER_ISSUE_WARPS-1:0] next_hz,

    // the granted warp's head instruction, popped when it issues
    output wire [ISSUE_WIS_W-1:0]         read_wis,
    input ibuffer_t                       read_data,
    output wire [PER_ISSUE_WARPS-1:0]     head_pop,

    VX_scoreboard_if.master scoreboard_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({writeback_if.data.sop, writeback_if.data.eop, writeback_if.data.wis})

    localparam NUM_OPDS  = NUM_SRC_OPDS + 1;
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

    wire [PER_ISSUE_WARPS-1:0] operands_ready;
    wire [PER_ISSUE_WARPS-1:0] arb_ready_in;

`ifdef PERF_ENABLE
    wire perf_stall_per_cycle = |(head_valid & ~operands_ready);

    always @(posedge clk) begin : g_perf_stalls
        if (reset) begin
            perf_stalls <= '0;
        end else begin
            perf_stalls <= perf_stalls + PERF_CTR_BITS'(perf_stall_per_cycle);
        end
    end
`endif

    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_scoreboard
        reg [NUM_REGS-1:0] inuse_regs, inuse_regs_n;
        reg [NUM_XREGS-1:0] inuse_xregs, inuse_xregs_n;

        // The head issues this cycle; the entry behind it becomes the head.
        wire staging_fire = head_pop[w];

        wire writeback_fire = writeback_if.valid && writeback_if.data.eop_wis[w];

        wire [NUM_OPDS-1:0] [NUM_REGS_BITS-1:0] ibf_opds, stg_opds;
        assign ibf_opds = {next_hz[w].rs3, next_hz[w].rs2, next_hz[w].rs1, next_hz[w].rd};
        assign stg_opds = {head_hz[w].rs3, head_hz[w].rs2, head_hz[w].rs1, head_hz[w].rd};

        wire [NUM_OPDS-1:0] ibf_used_rs = {next_hz[w].used_rs, next_hz[w].wb};
        wire [NUM_OPDS-1:0] stg_used_rs = {head_hz[w].used_rs, head_hz[w].wb};

        // Special-register dependency masks
        wire [NUM_XREGS-1:0] ibf_xregs_mask = next_hz[w].rd_xregs | next_hz[w].wr_xregs;
        wire [NUM_XREGS-1:0] stg_xregs_mask = head_hz[w].rd_xregs | head_hz[w].wr_xregs;

        wire [NUM_OPDS-1:0][REG_TYPES-1:0][RV_REGS-1:0] ibf_opd_mask, stg_opd_mask;

        for (genvar i = 0; i < NUM_OPDS; ++i) begin : g_opd_masks
            for (genvar j = 0; j < REG_TYPES; ++j) begin : g_j
                assign ibf_opd_mask[i][j] = (1 << get_reg_idx(ibf_opds[i])) & {RV_REGS{ibf_used_rs[i] && get_reg_type(ibf_opds[i]) == j}};
                assign stg_opd_mask[i][j] = (1 << get_reg_idx(stg_opds[i])) & {RV_REGS{stg_used_rs[i] && get_reg_type(stg_opds[i]) == j}};
            end
        end

        // Writeback release feeds wb_inuse_regs; the staging reserve is added on
        // top to form the next state, inuse_regs_n.
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
                if (head_hz[w].wb) begin
                    inuse_regs_n |= stg_opd_mask[0]; // reserve rd
                end
                inuse_xregs_n |= head_hz[w].wr_xregs; // reserve special regs
            end
        end

        // Readiness folds data hazards and FU-congestion into one flop; FU-lock
        // is enforced downstream by masking the arbiter requests.
        //
        // Whether this warp issues this cycle is decided by the out_arb grant,
        // which depends on every warp's operands_ready_r. Computing readiness
        // after that decision puts the whole arbitration in front of a
        // NUM_REGS-wide hazard reduction. Instead both outcomes are evaluated from state that
        // does not depend on the grant, and staging_fire only selects:
        //   hold: the head stays
        //   fire: the head issues, its rd is reserved, the entry behind it moves up
        // Either way the flop gets the value a post-grant evaluation would. An
        // entry written this cycle is checked the next, once it is in a slot.
        //
        // Either outcome checks one of only two instrs, so each is checked
        // once and the outcomes select bits.
        wire [NUM_REGS-1:0] ibf_regs_mask = ibf_opd_mask[0] | ibf_opd_mask[1] | ibf_opd_mask[2] | ibf_opd_mask[3];
        wire [NUM_REGS-1:0] stg_regs_mask = stg_opd_mask[0] | stg_opd_mask[1] | stg_opd_mask[2] | stg_opd_mask[3];
        // Each operand reads its in-use bit by index and compares its index with
        // this cycle's release, which is shallower than reducing the operand
        // masks against the whole released set.
        wire wb_release = writeback_fire && writeback_if.data.wb;
        wire [NUM_OPDS-1:0] ibf_opd_busy, stg_opd_busy;
        for (genvar i = 0; i < NUM_OPDS; ++i) begin : g_opd_busy
            assign ibf_opd_busy[i] = ibf_used_rs[i] && inuse_regs[ibf_opds[i]]
                                  && ~(wb_release && (writeback_if.data.rd == ibf_opds[i]));
            assign stg_opd_busy[i] = stg_used_rs[i] && inuse_regs[stg_opds[i]]
                                  && ~(wb_release && (writeback_if.data.rd == stg_opds[i]));
        end
        wire ibf_busy  = | ibf_opd_busy;
        wire stg_busy  = | stg_opd_busy;
        wire ibf_xbusy = | (wb_inuse_xregs & ibf_xregs_mask);
        wire stg_xbusy = | (wb_inuse_xregs & stg_xregs_mask);

        // On fire the staging instr also reserves its rd and wr_xregs. Its
        // successor sees that reservation only through a register it names,
        // and a lone staging instr always names its own rd.
        wire [NUM_OPDS-1:0] ibf_reads_rd;
        for (genvar i = 0; i < NUM_OPDS; ++i) begin : g_ibf_reads_rd
            assign ibf_reads_rd[i] = ibf_used_rs[i] && (ibf_opds[i] == stg_opds[0]);
        end
        wire stg_rsv_rd = head_hz[w].wb;
        wire ibf_rsv    = (stg_rsv_rd && (| ibf_reads_rd))
                       || (| (head_hz[w].wr_xregs & ibf_xregs_mask));

        wire ibf_goingfull = fu_goingfull[next_hz[w].ex_type];
        wire stg_goingfull = fu_goingfull[head_hz[w].ex_type];

        wire hold_ready = head_valid[w] && ~(stg_busy || stg_xbusy) && ~stg_goingfull;
        wire fire_ready = ISSUE_LOOKAHEAD && next_valid[w] && ~(ibf_busy || ibf_xbusy || ibf_rsv) && ~ibf_goingfull;

        wire operands_ready_n = staging_fire ? fire_ready : hold_ready;
        reg operands_ready_r;

    `ifdef SIMULATION
        // Per-operand busy view of the head, for the stall trace.
        wire [REG_TYPES-1:0][RV_REGS-1:0] in_use_mask = inuse_regs & stg_regs_mask;
        wire [NUM_OPDS-1:0] operands_busy;
        for (genvar i = 0; i < NUM_OPDS; ++i) begin : g_operands_busy
            wire [REG_TYPE_BITS-1:0] rtype = get_reg_type(stg_opds[i]);
            assign operands_busy[i] = | (in_use_mask[rtype] & stg_opd_mask[i][rtype]);
        end
        wire xregs_busy = | (inuse_xregs & stg_xregs_mask);
        `UNUSED_VAR ({ibf_regs_mask, xregs_busy})
    `else
        `UNUSED_VAR ({ibf_regs_mask, stg_regs_mask})
    `endif

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
                if (head_valid[w] && ~head_pop[w]) begin
                `ifdef DBG_TRACE_PIPELINE
                    `TRACE(4, ("%t: *** %s-stall: wid=%0d, PC=0x%0h, tmask=%b, cycles=%0d, opds_busy=%b, xregs_busy=%b (#%0d)\n",
                        $time, INSTANCE_ID, w, to_fullPC(head_hz[w].PC), head_hz[w].tmask, timeout_ctr,
                        operands_busy, xregs_busy, head_hz[w].uuid))
                `endif
                    timeout_ctr <= timeout_ctr + 1;
                end else begin
                    timeout_ctr <= '0;
                end
            end
        end

        `RUNTIME_ASSERT((timeout_ctr < STALL_TIMEOUT),
            ("timeout: wid=%0d, PC=0x%0h, tmask=%b, cycles=%0d, inuse=%b (#%0d)",
                w, to_fullPC(head_hz[w].PC), head_hz[w].tmask, timeout_ctr,
                operands_busy, head_hz[w].uuid))

        `RUNTIME_ASSERT(~(writeback_fire && writeback_if.data.wb) || inuse_regs[writeback_if.data.rd] != 0,
            ("invalid writeback register: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d (#%0d)",
                w, to_fullPC(writeback_if.data.PC), writeback_if.data.tmask, writeback_if.data.rd, writeback_if.data.uuid))

        `RUNTIME_ASSERT(~writeback_fire || ((writeback_if.data.wr_xregs & ~inuse_xregs) == '0),
            ("invalid writeback special register: wid=%0d, PC=0x%0h, tmask=%b, xregs=%b (#%0d)",
                w, to_fullPC(writeback_if.data.PC), writeback_if.data.tmask, writeback_if.data.wr_xregs, writeback_if.data.uuid))
    `endif

    end

    wire [PER_ISSUE_WARPS-1:0] arb_valid_in;

    // FU lock: a sequence must not interleave with another warp at the same FU.
    // fu_locked ('1 = open, one-hot = locked) gates arb_valid_in so only the lock
    // holder is requested while it holds the lock.
    reg [PER_ISSUE_WARPS-1:0] fu_locked;

`ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
    // A warp that issues a yielding uop while another warp also requests with
    // one sits out the next cycle, so the greedy arbiter moves on as it does
    // between scalar instructions that arrive one per fetch.
    reg [PER_ISSUE_WARPS-1:0] yield_r;
    wire [PER_ISSUE_WARPS-1:0] staging_yield_vec;
    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_staging_yield
        assign staging_yield_vec[w] = head_hz[w].issue_yield;
    end
    wire [PER_ISSUE_WARPS-1:0] yield_reqs = arb_valid_in & staging_yield_vec;
    wire [PER_ISSUE_WARPS-1:0] req_below;
    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_req_below
        if (w == 0) begin : g_first
            assign req_below[w] = 1'b0;
        end else begin : g_next
            assign req_below[w] = req_below[w-1] || yield_reqs[w-1];
        end
    end
    wire multi_req = | (yield_reqs & req_below);
`else
    wire [PER_ISSUE_WARPS-1:0] yield_r = '0;
`endif

    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_arb_valid_in
        // operands_ready carries data-hazard + FU-congestion; fu_locked adds the
        // FU-lock gate so only the lock holder is requested during a sequence.
        assign arb_valid_in[w] = head_valid[w] && operands_ready[w] && fu_locked[w] && ~yield_r[w];
        assign head_pop[w] = arb_valid_in[w] && arb_ready_in[w];
    end

    localparam LOG_NUM_REQS = `LOG2UP(PER_ISSUE_WARPS);

    wire                    arb_valid;
    wire [LOG_NUM_REQS-1:0] arb_index;
    wire [PER_ISSUE_WARPS-1:0] arb_onehot;
    wire                    arb_ready;

    // Matrix arbiter scales better past 8 requesters; RR is cheaper below that.
    // Past 16 the matrix goes two-level: its state would grow with the square
    // of the warps.
    if (PER_ISSUE_WARPS > 16) begin : g_out_arb2
        VX_matrix2_arbiter #(
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
    end else begin : g_out_arb
        VX_generic_arbiter #(
            .NUM_REQS (PER_ISSUE_WARPS),
            .TYPE     ((PER_ISSUE_WARPS > 8) ? "M" : "R"),
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
    end

    wire valid_out_w;
    wire [OUT_DATAW-1:0] data_out_w;
    wire ready_out_w;

    // The granted warp's head instruction is read from the buffer's RAM.
    assign read_wis    = ISSUE_WIS_W'(arb_index);
    assign valid_out_w = arb_valid;
    assign data_out_w  = {
        read_data.uuid,
        read_data.cta_id,
        read_data.tmask,
        read_data.PC,
        read_data.ex_type,
        read_data.op_type,
        read_data.op_args,
        read_data.wb,
        read_data.wr_xregs,
        read_data.used_rs,
        read_data.rd,
        read_data.bytesel,
        read_data.rs1,
        read_data.rs2,
        read_data.rs3
    };
    `UNUSED_VAR ({read_data.wid, read_data.rd_xregs, read_data.fu_lock, read_data.fu_unlock})
`ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
    `UNUSED_VAR (read_data.issue_yield)
`endif

    for (genvar i = 0; i < PER_ISSUE_WARPS; ++i) begin : g_arb_ready_in
        assign arb_ready_in[i] = ready_out_w && arb_onehot[i];
    end

    assign arb_ready = ready_out_w;

    // A sequence carries fu_lock on its first uop (acquire) and fu_unlock on its
    // last (release); 11 = single-uop default.

    wire issue_fire = valid_out_w && ready_out_w;

    wire [PER_ISSUE_WARPS-1:0][EX_BITS-1:0] staging_ex_vec;
    wire [PER_ISSUE_WARPS-1:0] staging_fu_lock_vec;
    wire [PER_ISSUE_WARPS-1:0] staging_fu_unlock_vec;
    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_staging_fu_lock
        assign staging_ex_vec[w]        = head_hz[w].ex_type;
        assign staging_fu_lock_vec[w]   = head_hz[w].fu_lock;
        assign staging_fu_unlock_vec[w] = head_hz[w].fu_unlock;
    end

    wire [EX_BITS-1:0] issue_ex = staging_ex_vec[arb_index];

    for (genvar e = 0; e < NUM_EX_UNITS; ++e) begin : g_fu_issue
        assign fu_issue[e] = issue_fire && (issue_ex == EX_BITS'(e));
    end

    // Lock to the granted warp on acquire, hold across its sequence, reopen on
    // release. arb_onehot is the granted warp, registered into fu_locked.
    wire issue_fu_lock   = staging_fu_lock_vec[arb_index];
    wire issue_fu_unlock = staging_fu_unlock_vec[arb_index];

    always @(posedge clk) begin
        if (reset) begin
            fu_locked <= '1;
        end else if (issue_fire) begin
            if (issue_fu_lock && ~issue_fu_unlock) begin
                fu_locked <= arb_onehot;
            end else if (issue_fu_unlock) begin
                fu_locked <= '1;
            end
        end
    end

`ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
    always @(posedge clk) begin
        if (reset) begin
            yield_r <= '0;
        end else begin
            yield_r <= arb_onehot & staging_yield_vec & {PER_ISSUE_WARPS{issue_fire && multi_req}};
        end
    end
`endif

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
