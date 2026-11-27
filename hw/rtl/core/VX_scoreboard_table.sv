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

// One warp's in-use register table: registers are reserved when the staged
// instruction issues and released at writeback; operands_ready tells whether
// the staged instruction can issue next cycle.
module VX_scoreboard_table import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

    input wire [ISSUE_WIS_W-1:0] wid,

    input wire [NUM_EX_UNITS-1:0] fu_goingfull,

    // the ibuffer's instruction entering staging
    input wire              ibf_valid,
    input wire              ibf_ready,
    input ibuffer_t         ibf_data,

    // the staged instruction
    input wire              stg_valid,
    input wire              stg_ready,
    input ibuffer_t         stg_data,

    // writeback
    input wire              wb_valid,
    input wire              wb_eop,
    input wire              wb_wb,
    input wire [NUM_REGS_BITS-1:0] wb_rd,
    // wb_rd decoded, shared by every warp's table
    input wire [NUM_REGS-1:0] wb_rd_onehot,
    input wire [NUM_XREGS-1:0] wb_wr_xregs,
    input wire [PC_BITS-1:0] wb_PC,
    input wire [`VX_CFG_SIMD_WIDTH-1:0] wb_tmask,
    input wire [UUID_WIDTH-1:0] wb_uuid,

    output wire             operands_ready
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({ibf_data.uuid, ibf_data.wid, ibf_data.cta_id, ibf_data.tmask, ibf_data.PC, ibf_data.op_type,
                  ibf_data.op_args, ibf_data.bytesel, ibf_data.fu_lock, ibf_data.fu_unlock})
    `UNUSED_VAR ({stg_data.wid, stg_data.cta_id, stg_data.op_type, stg_data.op_args, stg_data.bytesel,
                  stg_data.fu_lock, stg_data.fu_unlock})
`ifdef EXT_LDST_ANY_ENABLE
    `UNUSED_VAR ({ibf_data.issue_yield, stg_data.issue_yield})
`endif
`ifndef SIMULATION
    `UNUSED_VAR ({wid, stg_data.uuid, stg_data.tmask, stg_data.PC, wb_PC, wb_tmask, wb_uuid})
`endif

    localparam NUM_OPDS = NUM_SRC_OPDS + 1;

    reg [NUM_REGS-1:0] inuse_regs, inuse_regs_n;
    reg [NUM_XREGS-1:0] inuse_xregs, inuse_xregs_n;

    wire ibuffer_fire = ibf_valid && ibf_ready;
    wire staging_fire = stg_valid && stg_ready;

    wire writeback_fire = wb_valid && wb_eop;

    wire [NUM_OPDS-1:0] [NUM_REGS_BITS-1:0] ibf_opds, stg_opds;
    assign ibf_opds = {ibf_data.rs3, ibf_data.rs2, ibf_data.rs1, ibf_data.rd};
    assign stg_opds = {stg_data.rs3, stg_data.rs2, stg_data.rs1, stg_data.rd};

    wire [NUM_OPDS-1:0] ibf_used_rs = {ibf_data.used_rs, ibf_data.wb};
    wire [NUM_OPDS-1:0] stg_used_rs = {stg_data.used_rs, stg_data.wb};

    // Special-register dependency masks
    wire [NUM_XREGS-1:0] ibf_xregs_mask = ibf_data.rd_xregs | ibf_data.wr_xregs;
    wire [NUM_XREGS-1:0] stg_xregs_mask = stg_data.rd_xregs | stg_data.wr_xregs;

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
            if (wb_wb) begin
                wb_inuse_regs &= ~wb_rd_onehot; // release rd
            end
            wb_inuse_xregs &= ~wb_wr_xregs; // release special regs
        end
    end

    always @(*) begin
        inuse_regs_n  = wb_inuse_regs;
        inuse_xregs_n = wb_inuse_xregs;
        if (staging_fire) begin
            if (stg_data.wb) begin
                inuse_regs_n |= stg_opd_mask[0]; // reserve rd
            end
            inuse_xregs_n |= stg_data.wr_xregs; // reserve special regs
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
    //   hold: staging keeps its instr (or, if empty, takes the ibuffer's)
    //   fire: staging hands off, its rd is reserved, the ibuffer instr moves in
    // Either way the flop gets the value a post-grant evaluation would.
    //
    // Either outcome checks one of only two instrs, so each is checked
    // once and the outcomes select bits.
    wire [NUM_REGS-1:0] ibf_regs_mask = ibf_opd_mask[0] | ibf_opd_mask[1] | ibf_opd_mask[2] | ibf_opd_mask[3];
    wire [NUM_REGS-1:0] stg_regs_mask = stg_opd_mask[0] | stg_opd_mask[1] | stg_opd_mask[2] | stg_opd_mask[3];
    // Each operand reads its in-use bit by index and compares its index with
    // this cycle's release, which is shallower than reducing the operand
    // masks against the whole released set.
    wire wb_release = writeback_fire && wb_wb;
    wire [NUM_OPDS-1:0] ibf_opd_busy, stg_opd_busy;
    for (genvar i = 0; i < NUM_OPDS; ++i) begin : g_opd_busy
        assign ibf_opd_busy[i] = ibf_used_rs[i] && inuse_regs[ibf_opds[i]]
                              && ~(wb_release && (wb_rd == ibf_opds[i]));
        assign stg_opd_busy[i] = stg_used_rs[i] && inuse_regs[stg_opds[i]]
                              && ~(wb_release && (wb_rd == stg_opds[i]));
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
    wire stg_rsv_rd = stg_data.wb;
    wire ibf_rsv    = (stg_rsv_rd && (| ibf_reads_rd))
                   || (| (stg_data.wr_xregs & ibf_xregs_mask));
    wire stg_rsv    = stg_rsv_rd
                   || (| (stg_data.wr_xregs & stg_xregs_mask));

    wire hold_ibf = ibf_valid && ~stg_valid;
    wire fire_ibf = ibf_valid;

    wire hold_busy = hold_ibf ? (ibf_busy || ibf_xbusy)
                              : (stg_busy || stg_xbusy);
    wire fire_busy = fire_ibf ? (ibf_busy || ibf_xbusy || ibf_rsv)
                              : (stg_busy || stg_xbusy || stg_rsv);

    wire ibf_goingfull = fu_goingfull[ibf_data.ex_type];
    wire stg_goingfull = fu_goingfull[stg_data.ex_type];

    wire hold_ready = ~hold_busy && ~(hold_ibf ? ibf_goingfull : stg_goingfull);
    wire fire_ready = ~fire_busy && ~(fire_ibf ? ibf_goingfull : stg_goingfull);

    wire operands_ready_n = staging_fire ? fire_ready : hold_ready;
    reg operands_ready_r;

`ifdef SIMULATION
    // Per-operand busy view of the selected outcome, for the stall trace.
    wire [NUM_REGS-1:0] regs_mask = ibuffer_fire ? ibf_regs_mask : stg_regs_mask;
    wire [REG_TYPES-1:0][RV_REGS-1:0] in_use_mask = inuse_regs_n & regs_mask;
    wire [NUM_OPDS-1:0] operands_busy;
    for (genvar i = 0; i < NUM_OPDS; ++i) begin : g_operands_busy
        wire [REG_TYPE_BITS-1:0] rtype = get_reg_type(stg_opds[i]);
        assign operands_busy[i] = | (in_use_mask[rtype] & stg_opd_mask[i][rtype]);
    end
    wire xregs_busy = | (inuse_xregs_n & (ibuffer_fire ? ibf_xregs_mask : stg_xregs_mask));
    `UNUSED_VAR (xregs_busy)
`else
    `UNUSED_VAR (ibuffer_fire)
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

    assign operands_ready = operands_ready_r;

`ifdef SIMULATION
    reg [31:0] timeout_ctr;

    always @(posedge clk) begin
        if (reset) begin
            timeout_ctr <= '0;
        end else begin
            if (stg_valid && ~stg_ready) begin
            `ifdef DBG_TRACE_PIPELINE
                `TRACE(4, ("%t: *** %s-stall: wid=%0d, PC=0x%0h, tmask=%b, cycles=%0d, opds_busy=%b, xregs_busy=%b (#%0d)\n",
                    $time, INSTANCE_ID, wid, to_fullPC(stg_data.PC), stg_data.tmask, timeout_ctr,
                    operands_busy, xregs_busy, stg_data.uuid))
            `endif
                timeout_ctr <= timeout_ctr + 1;
            end else if (ibuffer_fire) begin
                timeout_ctr <= '0;
            end
        end
    end

    `RUNTIME_ASSERT((timeout_ctr < STALL_TIMEOUT),
        ("timeout: wid=%0d, PC=0x%0h, tmask=%b, cycles=%0d, inuse=%b (#%0d)",
            wid, to_fullPC(stg_data.PC), stg_data.tmask, timeout_ctr,
            operands_busy, stg_data.uuid))

    `RUNTIME_ASSERT(~(writeback_fire && wb_wb) || inuse_regs[wb_rd] != 0,
        ("invalid writeback register: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d (#%0d)",
            wid, to_fullPC(wb_PC), wb_tmask, wb_rd, wb_uuid))

    `RUNTIME_ASSERT(~writeback_fire || ((wb_wr_xregs & ~inuse_xregs) == '0),
        ("invalid writeback special register: wid=%0d, PC=0x%0h, tmask=%b, xregs=%b (#%0d)",
            wid, to_fullPC(wb_PC), wb_tmask, wb_wr_xregs, wb_uuid))
`endif

endmodule
