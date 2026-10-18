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

module VX_wctl_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = 1
) (
    input wire              clk,
    input wire              reset,

    // Inputs
    VX_execute_if.slave     execute_if,

    // Outputs
    VX_warp_ctl_if.master   warp_ctl_if,
    VX_warp_shared_if.master warp_shared_if,
    VX_result_if.master     result_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam LANE_BITS  = `CLOG2(NUM_LANES);
    localparam PID_BITS   = `CLOG2(`VX_CFG_NUM_THREADS / NUM_LANES);
    localparam WCTL_WIDTH = $bits(tmc_t) + $bits(split_t) + $bits(join_t);

    `UNUSED_VAR (execute_if.data.rs3_data)

    tmc_t       tmc;
    wspawn_t    wspawn;
    split_t     split;
    join_t      sjoin;
    barrier_t   bar;

    wire is_wspawn = (execute_if.data.op_type == INST_SFU_WSPAWN);
    wire is_tmc    = (execute_if.data.op_type == INST_SFU_TMC);
    wire is_pred   = (execute_if.data.op_type == INST_SFU_PRED);
    wire is_split  = (execute_if.data.op_type == INST_SFU_SPLIT);
    wire is_join   = (execute_if.data.op_type == INST_SFU_JOIN);
    wire is_bar    = (execute_if.data.op_type == INST_SFU_BAR);
    wire is_wsync  = (execute_if.data.op_type == INST_SFU_WSYNC);

    wire wctl_valid;
    wire tmc_valid, split_valid, sjoin_valid, wsync_valid;

    wire [`UP(LANE_BITS)-1:0] last_tid;
	    if (LANE_BITS != 0) begin : g_last_tid
	        VX_priority_encoder #(
	            .N (NUM_LANES),
	            .REVERSE (1)
	        ) last_tid_select (
	            .data_in (execute_if.data.header.tmask),
	            .index_out (last_tid),
	            `UNUSED_PIN (onehot_out),
            `UNUSED_PIN (valid_out)
        );
    end else begin : g_no_tid
        assign last_tid = 0;
    end

    wire [`VX_CFG_XLEN-1:0] rs1_data = execute_if.data.rs1_data[last_tid];
    wire [`VX_CFG_XLEN-1:0] rs2_data = execute_if.data.rs2_data[last_tid];
    `UNUSED_VAR (rs1_data)

    wire not_pred = execute_if.data.op_args.wctl.is_cond_neg;

    wire [NUM_LANES-1:0] taken;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_taken
        assign taken[i] = (execute_if.data.rs1_data[i][0] ^ not_pred);
    end

    logic [`VX_CFG_NUM_THREADS-1:0] then_tmask;
    logic [`VX_CFG_NUM_THREADS-1:0] else_tmask;

    if (PID_BITS != 0) begin : g_pid
        reg [SFU_BLOCK_WARPS-1:0][2*`VX_CFG_NUM_THREADS-1:0] tmask_table;

        wire [SFU_BLOCK_NW_W-1:0] tmask_lid = wid_to_sfu_local(execute_if.data.header.wid);
        wire [2*`VX_CFG_NUM_THREADS-1:0] tmask_r = tmask_table[tmask_lid];

        always @(*) begin
            {else_tmask, then_tmask} = execute_if.data.header.sop ? '0 : tmask_r;
            then_tmask[execute_if.data.header.pid * NUM_LANES +: NUM_LANES] = taken & execute_if.data.header.tmask;
            else_tmask[execute_if.data.header.pid * NUM_LANES +: NUM_LANES] = ~taken & execute_if.data.header.tmask;
        end

        always @(posedge clk) begin
            if (execute_if.valid) begin
                tmask_table[tmask_lid] <= {else_tmask, then_tmask};
            end
        end
    end else begin : g_no_pid
        assign then_tmask = taken & execute_if.data.header.tmask;
        assign else_tmask = ~taken & execute_if.data.header.tmask;
    end

    wire has_then = (then_tmask != 0);
    wire has_else = (else_tmask != 0);

    // tmc / pred

    wire [`VX_CFG_NUM_THREADS-1:0] pred_mask = has_then ? then_tmask : rs2_data[`VX_CFG_NUM_THREADS-1:0];
    assign tmc.tmask = is_pred ? pred_mask : rs1_data[`VX_CFG_NUM_THREADS-1:0];

    // split

    wire [`CLOG2(`VX_CFG_NUM_THREADS+1)-1:0] then_tmask_cnt, else_tmask_cnt;
    `POP_COUNT(then_tmask_cnt, then_tmask);
    `POP_COUNT(else_tmask_cnt, else_tmask);
    wire then_first = (then_tmask_cnt <= else_tmask_cnt);
    wire [`VX_CFG_NUM_THREADS-1:0] taken_tmask = then_first ? then_tmask : else_tmask;
    wire [`VX_CFG_NUM_THREADS-1:0] ntaken_tmask = then_first ? else_tmask : then_tmask;

    assign split_valid      = wctl_valid && is_split;
    assign split.is_dvg     = has_then && has_else;
    assign split.then_tmask = taken_tmask;
    assign split.else_tmask = ntaken_tmask;
    assign split.next_pc    = execute_if.data.header.PC + from_fullPC(`VX_CFG_XLEN'(4));

    // join

    assign sjoin_valid      = wctl_valid && is_join;
    assign sjoin.tmask      = then_tmask | else_tmask;
    assign sjoin.stack_ptr  = rs1_data[DV_STACK_SIZEW-1:0];

    // barrier

    wire [BAR_ADDR_W-1:0] wctl_bar_addr;
    `CONCAT(wctl_bar_addr, rs1_data[NW_BITS-1:0], rs1_data[BAR_ID_SHIFT +: NB_BITS], NW_BITS, NB_BITS)

    // expect_tx variant of barrier_arrive: rs2[31]=1 flags pre-registration
    // of pending transaction events. Routed through the existing event path
    // (is_event + phase=1) so VX_bar_unit's increment branch handles it.
    wire is_tx_expect   = is_bar
                       && execute_if.data.op_args.wctl.is_bar_arrive
                       && rs2_data[31];

    assign bar.id       = rs1_data[BAR_ID_SHIFT +: NB_BITS];
    assign bar.is_event = is_tx_expect;
    assign bar.is_sync  = execute_if.data.op_args.wctl.is_sync_bar;
    assign bar.is_global= rs1_data[31];
    // expect_tx is not a normal arrival (it does not advance arrive_count)
    assign bar.is_arrive= (execute_if.data.op_args.wctl.is_bar_arrive && ~rs2_data[31]) || execute_if.data.op_args.wctl.is_sync_bar;
    // expect_tx → phase=1 (attach/increment)
    assign bar.phase    = is_tx_expect ? 1'b1 : rs2_data[0];
    // For expect_tx: size_m1 carries (count - 1); for normal arrive: (num_warps - 1).
    // bar_unit will + 1 when consuming on the event path so the actual increment is `count`.
    assign bar.size_m1  = rs2_data[BAR_SIZE_W-1:0] - BAR_SIZE_W'(1);

    // wspawn

    wire [`VX_CFG_NUM_WARPS-1:0] wspawn_wmask;
    for (genvar i = 0; i < `VX_CFG_NUM_WARPS; ++i) begin : g_wspawn_wmask
        assign wspawn_wmask[i] = (i < rs1_data[NW_BITS:0]) && (i != execute_if.data.header.wid);
    end
    assign wspawn.wmask = wspawn_wmask;
    assign wspawn.pc    = from_fullPC(rs2_data);

    // Scheduler response setup

    assign warp_ctl_if.dvstack_wid = execute_if.data.header.wid;

    // WSYNC: hold execute until this warp's pipeline is drained (alm_empty=1
    // means only WSYNC itself is pending — all prior instructions committed).
    wire wsync_drain = execute_if.valid && is_wsync
                    && !warp_ctl_if.warp_pending_alm_empty[wid_to_sfu_local(execute_if.data.header.wid)];

    // BAR (vx_barrier or vx_barrier_arrive) drains LSU before continuing.
    wire bar_drain = execute_if.valid && is_bar && !warp_ctl_if.lsu_sched_drained;

    // Barriers, wspawn and warp retirement are core-wide requests to the shared
    // unit. Only a barrier arrive writes a result, the barrier phase, so it alone
    // completes once the unit has served it; every other request is posted,
    // completing as it is accepted, and the warp resumes through its unlock path.
    wire bar_rsp_valid = is_bar && execute_if.data.op_args.wctl.is_bar_arrive;
    wire is_exit   = (is_tmc || is_pred) && (tmc.tmask == 0);
    wire is_shared = execute_if.data.header.eop && (is_bar || is_wspawn || is_exit);
    wire sh_posted = is_shared && ~bar_rsp_valid;

    // an exit is retired by the shared unit, which deactivates the warp itself
    assign tmc_valid = wctl_valid && (is_tmc || is_pred) && !is_shared;

    localparam SH_IDLE = 2'd0, SH_WAIT = 2'd1, SH_DONE = 2'd2;
    reg [1:0] sh_state;
    reg       sh_phase;

    warp_shared_req_t sh_req;
    assign sh_req.kind     = is_bar ? WSH_BAR : (is_wspawn ? WSH_WSPAWN : WSH_EXIT);
    assign sh_req.has_rsp  = ~sh_posted;
    assign sh_req.wid      = execute_if.data.header.wid;
    assign sh_req.bar      = bar;
    assign sh_req.bar_addr = wctl_bar_addr;
    assign sh_req.wspawn   = wspawn;
    assign sh_req.mscratch = warp_ctl_if.mscratch;

    // A sole requester reaches the shared unit directly and is answered in the
    // cycle it asks; with several, requests are registered toward the unit's
    // arbiter and answered two cycles later.
    localparam SH_DIRECT = (`VX_CFG_NUM_SFU_BLOCKS == 1);

    wire rsp_buf_ready;
    // a posted or directly answered request is sent only in the cycle its instruction completes
    wire sh_req_valid = execute_if.valid && is_shared && (sh_state == SH_IDLE) && !bar_drain
                     && ((!sh_posted && !SH_DIRECT) || rsp_buf_ready);
    wire sh_req_ready;

    if (SH_DIRECT) begin : g_sh_direct
        assign warp_shared_if.req_valid = sh_req_valid;
        assign warp_shared_if.req_data  = sh_req;
        assign sh_req_ready = warp_shared_if.req_ready;
    end else begin : g_sh_buf
        VX_elastic_buffer #(
            .DATAW   ($bits(warp_shared_req_t)),
            .SIZE    (2),
            .OUT_REG (1)
        ) sh_req_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (sh_req_valid),
            .ready_in  (sh_req_ready),
            .data_in   (sh_req),
            .data_out  (warp_shared_if.req_data),
            .valid_out (warp_shared_if.req_valid),
            .ready_out (warp_shared_if.req_ready)
        );
    end

    wire sh_req_fire = sh_req_valid && sh_req_ready;
    wire sh_ok = ~is_shared || (sh_state == SH_DONE)
              || ((sh_state == SH_WAIT) && warp_shared_if.rsp_valid)
              || (sh_req_fire && (sh_posted || warp_shared_if.rsp_valid));
    wire exec_ok = rsp_buf_ready && !wsync_drain && sh_ok && (is_shared || !bar_drain);
    assign execute_if.ready = exec_ok;

    wire execute_fire = execute_if.valid && execute_if.ready;
    assign wctl_valid = execute_fire && execute_if.data.header.eop;

    always @(posedge clk) begin
        if (reset) begin
            sh_state <= SH_IDLE;
        end else begin
            case (sh_state)
                SH_IDLE: begin
                    if (sh_req_fire && !sh_posted && !execute_fire) begin
                        sh_state <= warp_shared_if.rsp_valid ? SH_DONE : SH_WAIT;
                    end
                end
                SH_WAIT: begin
                    if (warp_shared_if.rsp_valid) begin
                        sh_state <= execute_fire ? SH_IDLE : SH_DONE;
                    end
                end
                default: begin
                    if (execute_fire) begin
                        sh_state <= SH_IDLE;
                    end
                end
            endcase
        end
        if (warp_shared_if.rsp_valid) begin
            sh_phase <= warp_shared_if.rsp_phase;
        end
    end

    assign wsync_valid = wctl_valid && is_wsync;

    VX_pipe_register #(
        .DATAW (4 + NW_WIDTH + WCTL_WIDTH),
        .RESETW (1)
    ) wctl_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({
            tmc_valid,
            split_valid,
            sjoin_valid,
            wsync_valid,
            execute_if.data.header.wid,
            tmc,
            split,
            sjoin
        }),
        .data_out ({
            warp_ctl_if.tmc_valid,
            warp_ctl_if.split_valid,
            warp_ctl_if.sjoin_valid,
            warp_ctl_if.wsync_valid,
            warp_ctl_if.wid,
            warp_ctl_if.tmc,
            warp_ctl_if.split,
            warp_ctl_if.sjoin
        })
    );

    // Send result

    wire rsp_phase = warp_shared_if.rsp_valid ? warp_shared_if.rsp_phase : sh_phase;
    wire [DV_STACK_SIZEW-1:0] dvstack_ptr_r;
    wire bar_rsp_valid_r;
    wire bar_phase_r;

    VX_elastic_buffer #(
        .DATAW ($bits(sfu_header_t) + DV_STACK_SIZEW + 1 + 1),
        .SIZE  (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (execute_if.valid && exec_ok),
        .ready_in  (rsp_buf_ready),
        .data_in   ({execute_if.data.header, warp_ctl_if.dvstack_ptr, rsp_phase, bar_rsp_valid}),
        .data_out  ({result_if.data.header,  dvstack_ptr_r,           bar_phase_r, bar_rsp_valid_r}),
        .valid_out (result_if.valid),
        .ready_out (result_if.ready)
    );

    // Result data
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_result_if
        assign result_if.data.data[i] = bar_rsp_valid_r ? `VX_CFG_XLEN'(bar_phase_r) : `VX_CFG_XLEN'(dvstack_ptr_r);
    end

endmodule
