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
    input gpr_layout_t      gpr_layout,

    // the entry the buffer writes this cycle, and every warp's entries
    input wire                            ins_valid,
    input wire [ISSUE_WIS_W-1:0]          ins_wis,
    input ibuf_hz_t                       ins_hz,
    input ibuf_hz_t [PER_ISSUE_WARPS-1:0][`VX_CFG_IBUF_SIZE-1:0] slots_hz,
    input wire [PER_ISSUE_WARPS-1:0][`CLOG2(`VX_CFG_IBUF_SIZE + 1)-1:0] slots_count,

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
    `UNUSED_VAR ({writeback_if.data.sop, writeback_if.data.eop})

    localparam NUM_OPDS  = NUM_SRC_OPDS + 1;
    localparam OUT_DATAW = $bits(scoreboard_t) - ISSUE_WIS_W;
    localparam OUT_BUF   = 3; // Use skid buffer (SIZE=2, OUT_REG=1)
    localparam DEPTH     = `VX_CFG_IBUF_SIZE;
    localparam CNTW      = `CLOG2(DEPTH + 1);
    localparam NOPC      = `VX_CFG_NUM_OPCS;
    // one busy bit per register-file row of the slice's collectors
    localparam GROWS     = NOPC * `VX_CFG_GPR_ROWS;
    localparam GROW_W    = `LOG2UP(GROWS);
`ifdef VX_CFG_EXT_U_ENABLE
    localparam UROWS     = NOPC * `VX_CFG_UREG_ROWS;
`else
    localparam UROWS     = 1;
`endif
    localparam UROW_W    = `LOG2UP(UROWS);

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

    // A warp's registers sit at its set's base in its collector's file, as the
    // operand collectors lay them out: warps dealt round-robin to collectors.
    reg [PER_ISSUE_WARPS-1:0][GROW_W-1:0] xbase_r, fbase_r;
    reg [PER_ISSUE_WARPS-1:0][UROW_W-1:0] ubase_r;
    always @(posedge clk) begin
        for (integer w = 0; w < PER_ISSUE_WARPS; ++w) begin
            xbase_r[w] <= GROW_W'((w % NOPC) * `VX_CFG_GPR_ROWS + (w / NOPC) * gpr_layout.size4 * 4);
            fbase_r[w] <= GROW_W'((w % NOPC) * `VX_CFG_GPR_ROWS + (w / NOPC) * gpr_layout.size4 * 4 + gpr_layout.fp_off4 * 4);
        `ifdef VX_CFG_EXT_U_ENABLE
            // u1 takes the set's first row; u0 reads zero and takes none
            ubase_r[w] <= UROW_W'((w % NOPC) * `VX_CFG_UREG_ROWS + (w / NOPC) * gpr_layout.usize4 * 4);
        `else
            ubase_r[w] <= '0;
        `endif
        end
    end

    function automatic logic is_ureg(input logic [NUM_REGS_BITS-1:0] r);
        return `VX_CFG_EXT_U_ENABLED && (get_reg_type(r) == REG_TYPE_U);
    endfunction

    // x0 and u0 are never written, so they are never busy
    function automatic logic is_zero(input logic [NUM_REGS_BITS-1:0] r);
        return (get_reg_type(r) != REG_TYPE_F || GPR_TYPES == 1) && (get_reg_idx(r) == '0);
    endfunction

    function automatic logic [GROW_W-1:0] grow_of(input logic [ISSUE_WIS_W-1:0] wis,
                                                  input logic [NUM_REGS_BITS-1:0] r);
        logic is_f;
        is_f = (GPR_TYPES > 1) && (get_reg_type(r) == REG_TYPE_F);
        return (is_f ? fbase_r[wis] : xbase_r[wis]) + GROW_W'(get_reg_idx(r));
    endfunction

    function automatic logic [UROW_W-1:0] urow_of(input logic [ISSUE_WIS_W-1:0] wis,
                                                  input logic [NUM_REGS_BITS-1:0] r);
        return ubase_r[wis] + UROW_W'(get_reg_idx(r)) - UROW_W'(1);
    endfunction

    // Busy bits: a register with a write pending, from the writer's insertion
    // to its writeback. A warp issues in order, so a register reserved at
    // insertion holds every later entry of the warp as reserving it at issue
    // would; each entry looks its operands up once. The lookup and the
    // reservation take the cycle after the write, from a registered copy: the
    // write ends the whole front-end path, and the entry is checked for issue
    // only from that next cycle anyway.
    reg [GROWS-1:0] gbusy;
    reg [UROWS-1:0] ubusy;

    reg                   insq_valid;
    reg [ISSUE_WIS_W-1:0] insq_wis;
    ibuf_hz_t             insq_hz;
    reg [CNTW-1:0]        insq_slot;
    wire [CNTW-1:0] ins_slot = slots_count[ins_wis] - CNTW'(head_pop[ins_wis]);
    always @(posedge clk) begin
        insq_valid <= ~reset && ins_valid;
        insq_wis   <= ins_wis;
        insq_hz    <= ins_hz;
        insq_slot  <= ins_slot;
    end

    wire [NUM_REGS_BITS-1:0] wb_rd = writeback_if.data.rd;
    wire wb_any  = writeback_if.valid && (| writeback_if.data.eop_wis) && writeback_if.data.wb;
    wire wb_is_u = is_ureg(wb_rd);
    wire [GROW_W-1:0] wb_grow = grow_of(writeback_if.data.wis, wb_rd);
    wire [UROW_W-1:0] wb_urow = urow_of(writeback_if.data.wis, wb_rd);
    // an older entry of the writing warp, already reserved, writes the register too
    wire [PER_ISSUE_WARPS-1:0] wb_keep_w;
    wire wb_keep = | (wb_keep_w & writeback_if.data.eop_wis);
    wire gclear  = wb_any && ~wb_is_u && ~wb_keep;
    wire uclear  = wb_any && wb_is_u && ~wb_keep;

    // the entry written last cycle: its operands' busy bits, after this cycle's release
    wire [NUM_OPDS-1:0][NUM_REGS_BITS-1:0] ins_opds = {insq_hz.rs3, insq_hz.rs2, insq_hz.rs1, insq_hz.rd};
    wire [NUM_OPDS-1:0] ins_used = {insq_hz.used_rs, insq_hz.wb};
    wire [NUM_OPDS-1:0] ins_pend;
    wire [NUM_OPDS-1:0][GROW_W-1:0] ins_grow;
    wire [NUM_OPDS-1:0][UROW_W-1:0] ins_urow;
    for (genvar i = 0; i < NUM_OPDS; ++i) begin : g_ins
        assign ins_grow[i] = grow_of(insq_wis, ins_opds[i]);
        assign ins_urow[i] = urow_of(insq_wis, ins_opds[i]);
        wire gb = gbusy[ins_grow[i]] && ~(gclear && (wb_grow == ins_grow[i]));
        wire ub = ubusy[ins_urow[i]] && ~(uclear && (wb_urow == ins_urow[i]));
        assign ins_pend[i] = ins_used[i] && ~is_zero(ins_opds[i]) && (is_ureg(ins_opds[i]) ? ub : gb);
    end
    `UNUSED_VAR (insq_hz) // only its operands are looked up
    wire ins_rsv = insq_valid && insq_hz.wb && ~is_zero(insq_hz.rd);
    wire ins_rd_u = is_ureg(insq_hz.rd);

    // the busy bits change only on an insertion or a writeback
    wire busy_upd = insq_valid || wb_any;
`ifdef ASIC
    wire busy_clk;
    VX_clockgate busy_cg (.clk_in (clk), .en (reset || busy_upd), .clk_out (busy_clk));
`else
    `UNUSED_VAR (busy_upd)
`endif
    always @(posedge `GATED_CLK(clk, busy_clk)) begin
        if (reset) begin
            gbusy <= '0;
            ubusy <= '0;
        end else begin
            if (gclear) begin
                gbusy[wb_grow] <= 1'b0;
            end
            if (uclear) begin
                ubusy[wb_urow] <= 1'b0;
            end
            // an insertion's reservation wins over a release of the same row
            if (ins_rsv && ~ins_rd_u) begin
                gbusy[ins_grow[0]] <= 1'b1;
            end
            if (ins_rsv && ins_rd_u) begin
                ubusy[ins_urow[0]] <= 1'b1;
            end
        end
    end

`ifdef SIMULATION
    `RUNTIME_ASSERT(~wb_any || (wb_is_u ? ubusy[wb_urow] : gbusy[wb_grow]),
        ("invalid writeback register: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d (#%0d)",
            writeback_if.data.wis, to_fullPC(writeback_if.data.PC), writeback_if.data.tmask, wb_rd, writeback_if.data.uuid))
`endif

    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_scoreboard
        reg [NUM_XREGS-1:0] inuse_xregs, inuse_xregs_n;

        // The head issues this cycle; the entry behind it becomes the head.
        wire staging_fire = head_pop[w];

        wire writeback_fire = writeback_if.valid && writeback_if.data.eop_wis[w];
        wire wb_release = writeback_fire && writeback_if.data.wb;

        // Per entry, its operands still waiting on a write; a writeback clears
        // an operand unless an older entry of the warp writes the register too.
        reg  [DEPTH-1:0][NUM_OPDS-1:0] pend;
        wire [DEPTH-1:0][NUM_OPDS-1:0] pend_w;
        wire [CNTW-1:0] count = slots_count[w];
        // the entry written last cycle takes this cycle's lookup; it has not
        // reserved its own register yet
        wire [DEPTH-1:0] fresh;
        wire [DEPTH-1:0] slot_writes;
        for (genvar k = 0; k < DEPTH; ++k) begin : g_slot_writes
            assign fresh[k] = insq_valid && (insq_wis == ISSUE_WIS_W'(w)) && (insq_slot == CNTW'(k));
            assign slot_writes[k] = (CNTW'(k) < count) && ~fresh[k] && slots_hz[w][k].wb && (slots_hz[w][k].rd == wb_rd);
        end
        assign wb_keep_w[w] = | slot_writes;
        for (genvar k = 0; k < DEPTH; ++k) begin : g_wake
            wire older_writes;
            if (k == 0) begin : g_head
                assign older_writes = 1'b0;
            end else begin : g_behind
                assign older_writes = | slot_writes[k-1:0];
            end
            wire [NUM_OPDS-1:0][NUM_REGS_BITS-1:0] opds = {slots_hz[w][k].rs3, slots_hz[w][k].rs2,
                                                           slots_hz[w][k].rs1, slots_hz[w][k].rd};
            wire [NUM_OPDS-1:0] pend_k = fresh[k] ? ins_pend : pend[k];
            for (genvar i = 0; i < NUM_OPDS; ++i) begin : g_opd
                assign pend_w[k][i] = pend_k[i] && ~(wb_release && ~older_writes && (opds[i] == wb_rd));
            end
        end

        // an entry written this cycle gets its bits from next cycle's lookup
        for (genvar k = 0; k < DEPTH; ++k) begin : g_pend
            always @(posedge clk) begin
                if (staging_fire && (k + 1 < DEPTH)) begin
                    pend[k] <= pend_w[`MIN(k + 1, DEPTH - 1)];
                end else begin
                    pend[k] <= pend_w[k];
                end
            end
        end

        // Special-register dependency masks
        wire [NUM_XREGS-1:0] ibf_xregs_mask = next_hz[w].rd_xregs | next_hz[w].wr_xregs;
        wire [NUM_XREGS-1:0] stg_xregs_mask = head_hz[w].rd_xregs | head_hz[w].wr_xregs;

        // Special registers are reserved at issue and released at writeback.
        reg [NUM_XREGS-1:0] wb_inuse_xregs;
        always @(*) begin
            wb_inuse_xregs = inuse_xregs;
            if (writeback_fire) begin
                wb_inuse_xregs &= ~writeback_if.data.wr_xregs; // release special regs
            end
        end

        always @(*) begin
            inuse_xregs_n = wb_inuse_xregs;
            if (staging_fire) begin
                inuse_xregs_n |= head_hz[w].wr_xregs; // reserve special regs
            end
        end

        // Readiness folds data hazards and FU-congestion into one flop; FU-lock
        // is enforced downstream by masking the arbiter requests.
        //
        // Whether this warp issues this cycle is decided by the out_arb grant,
        // which depends on every warp's operands_ready_r. Both outcomes are
        // evaluated from state that does not depend on the grant, and
        // staging_fire only selects:
        //   hold: the head stays
        //   fire: the head issues, the entry behind it moves up
        // An entry written this cycle is checked the next, once it is in a slot.
        wire stg_busy  = | pend_w[0];
        wire ibf_busy  = | pend_w[`MIN(1, DEPTH - 1)];
        wire stg_xbusy = | (wb_inuse_xregs & stg_xregs_mask);
        wire ibf_xbusy = | (wb_inuse_xregs & ibf_xregs_mask);
        // On fire the head reserves its special registers.
        wire ibf_xrsv  = | (head_hz[w].wr_xregs & ibf_xregs_mask);

        wire ibf_goingfull = fu_goingfull[next_hz[w].ex_type];
        wire stg_goingfull = fu_goingfull[head_hz[w].ex_type];

        wire hold_ready = head_valid[w] && ~(stg_busy || stg_xbusy) && ~stg_goingfull;
        wire fire_ready = ISSUE_LOOKAHEAD && next_valid[w]
                       && ~(ibf_busy || ibf_xbusy || ibf_xrsv)
                       && ~ibf_goingfull;

        wire operands_ready_n = staging_fire ? fire_ready : hold_ready;
        reg operands_ready_r;

    `ifdef SIMULATION
        // Per-operand busy view of the head, for the stall trace.
        wire [NUM_OPDS-1:0] operands_busy = pend_w[0];
        wire xregs_busy = | (inuse_xregs & stg_xregs_mask);
        `UNUSED_VAR (xregs_busy)
    `endif

        // The in-use bits change only on this warp's writeback or issue; on
        // ASIC they take no clock edge otherwise.
        wire inuse_upd = writeback_fire || staging_fire;
        `RUNTIME_ASSERT (inuse_upd || (inuse_xregs_n == inuse_xregs),
            ("%t: scoreboard %0d in-use bits changed without writeback or issue", $time, w))
    `ifdef ASIC
        wire inuse_clk;
        VX_clockgate inuse_cg (.clk_in (clk), .en (reset || inuse_upd), .clk_out (inuse_clk));
    `endif

        always @(posedge `GATED_CLK(clk, inuse_clk)) begin
            if (reset) begin
                inuse_xregs <= '0;
            end else begin
                inuse_xregs <= inuse_xregs_n;
            end
        end

        always @(posedge clk) begin
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
