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
    VX_commit_if.master     commit_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam LANE_BITS  = `CLOG2(NUM_LANES);
    localparam PID_BITS   = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam PID_WIDTH  = `UP(PID_BITS);
    localparam WCTL_WIDTH = $bits(tmc_t) + $bits(wspawn_t) + $bits(split_t) + $bits(join_t) + $bits(barrier_t);
    localparam DATAW = `UUID_WIDTH + `NW_WIDTH + NUM_LANES + `PC_BITS + `NR_BITS + 1 + WCTL_WIDTH + PID_WIDTH + 1 + 1 + `DV_STACK_SIZEW;

    `UNUSED_VAR (execute_if.data.rs3_data)

    tmc_t       tmc, tmc_r;
    wspawn_t    wspawn, wspawn_r;
    split_t     split, split_r;
    join_t      sjoin, sjoin_r;
    barrier_t   barrier, barrier_r;

    wire is_wspawn = (execute_if.data.op_type == `INST_SFU_WSPAWN);
    wire is_tmc    = (execute_if.data.op_type == `INST_SFU_TMC);
    wire is_pred   = (execute_if.data.op_type == `INST_SFU_PRED);
    wire is_split  = (execute_if.data.op_type == `INST_SFU_SPLIT);
    wire is_join   = (execute_if.data.op_type == `INST_SFU_JOIN);
    wire is_bar    = (execute_if.data.op_type == `INST_SFU_BAR);

    wire [`UP(LANE_BITS)-1:0] tid;
    if (LANE_BITS != 0) begin : g_tid
        assign tid = execute_if.data.tid[0 +: LANE_BITS];
    end else begin : g_no_tid
        assign tid = 0;
    end

    wire [`XLEN-1:0] rs1_data = execute_if.data.rs1_data[tid];
    wire [`XLEN-1:0] rs2_data = execute_if.data.rs2_data[tid];
    `UNUSED_VAR (rs1_data)

    wire not_pred = execute_if.data.op_args.wctl.is_neg;

    wire [NUM_LANES-1:0] taken;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_taken
        assign taken[i] = (execute_if.data.rs1_data[i][0] ^ not_pred);
    end

    reg [`NUM_THREADS-1:0] then_tmask_r, then_tmask_n;
    reg [`NUM_THREADS-1:0] else_tmask_r, else_tmask_n;
    always @(*) begin
        then_tmask_n = then_tmask_r;
        else_tmask_n = else_tmask_r;
        if (execute_if.data.sop) begin
            then_tmask_n = '0;
            else_tmask_n = '0;
        end
        then_tmask_n[execute_if.data.pid * NUM_LANES +: NUM_LANES] = taken & execute_if.data.tmask;
        else_tmask_n[execute_if.data.pid * NUM_LANES +: NUM_LANES] = ~taken & execute_if.data.tmask;
    end
    always @(posedge clk) begin
        if (execute_if.valid) begin
            then_tmask_r <= then_tmask_n;
            else_tmask_r <= else_tmask_n;
        end
    end
    wire has_then = (then_tmask_n != 0);
    wire has_else = (else_tmask_n != 0);

    // tmc / pred

    wire [`NUM_THREADS-1:0] pred_mask = has_then ? then_tmask_n : rs2_data[`NUM_THREADS-1:0];
    assign tmc.valid = (is_tmc || is_pred);
    assign tmc.tmask = is_pred ? pred_mask : rs1_data[`NUM_THREADS-1:0];

    // split

    wire [`CLOG2(`NUM_THREADS+1)-1:0] then_tmask_cnt, else_tmask_cnt;
    `POP_COUNT(then_tmask_cnt, then_tmask_n);
    `POP_COUNT(else_tmask_cnt, else_tmask_n);
    wire then_first = (then_tmask_cnt >= else_tmask_cnt);
    wire [`NUM_THREADS-1:0] taken_tmask = then_first ? then_tmask_n : else_tmask_n;
    wire [`NUM_THREADS-1:0] ntaken_tmask = then_first ? else_tmask_n : then_tmask_n;

    assign split.valid      = is_split;
    assign split.is_dvg     = has_then && has_else;
    assign split.then_tmask = taken_tmask;
    assign split.else_tmask = ntaken_tmask;
    assign split.next_pc    = execute_if.data.PC + `PC_BITS'(2);

    assign warp_ctl_if.dvstack_wid = execute_if.data.wid;
    wire [`DV_STACK_SIZEW-1:0] dvstack_ptr;

    // join

    assign sjoin.valid      = is_join;
    assign sjoin.stack_ptr  = rs1_data[`DV_STACK_SIZEW-1:0];

    // barrier
    assign barrier.valid    = is_bar;
    assign barrier.id       = rs1_data[`NB_WIDTH-1:0];
`ifdef GBAR_ENABLE
    assign barrier.is_global = rs1_data[31];
`else
    assign barrier.is_global = 1'b0;
`endif
    assign barrier.size_m1  = rs2_data[$bits(barrier.size_m1)-1:0] - $bits(barrier.size_m1)'(1);
    assign barrier.is_noop  = (rs2_data[$bits(barrier.size_m1)-1:0] == $bits(barrier.size_m1)'(1));

    // wspawn

    wire [`NUM_WARPS-1:0] wspawn_wmask;
    for (genvar i = 0; i < `NUM_WARPS; ++i) begin : g_wspawn_wmask
        assign wspawn_wmask[i] = (i < rs1_data[`NW_BITS:0]) && (i != execute_if.data.wid);
    end
    assign wspawn.valid = is_wspawn;
    assign wspawn.wmask = wspawn_wmask;
    assign wspawn.pc    = rs2_data[1 +: `PC_BITS];

    // response

    VX_elastic_buffer #(
        .DATAW (DATAW),
        .SIZE  (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (execute_if.valid),
        .ready_in  (execute_if.ready),
        .data_in   ({execute_if.data.uuid, execute_if.data.wid, execute_if.data.tmask, execute_if.data.PC, execute_if.data.rd, execute_if.data.wb, execute_if.data.pid, execute_if.data.sop, execute_if.data.eop, {tmc, wspawn, split, sjoin, barrier}, warp_ctl_if.dvstack_ptr}),
        .data_out  ({commit_if.data.uuid, commit_if.data.wid, commit_if.data.tmask, commit_if.data.PC, commit_if.data.rd, commit_if.data.wb, commit_if.data.pid, commit_if.data.sop, commit_if.data.eop, {tmc_r, wspawn_r, split_r, sjoin_r, barrier_r}, dvstack_ptr}),
        .valid_out (commit_if.valid),
        .ready_out (commit_if.ready)
    );

    assign warp_ctl_if.valid   = commit_if.valid && commit_if.ready && commit_if.data.eop;
    assign warp_ctl_if.wid     = commit_if.data.wid;
    assign warp_ctl_if.tmc     = tmc_r;
    assign warp_ctl_if.wspawn  = wspawn_r;
    assign warp_ctl_if.split   = split_r;
    assign warp_ctl_if.sjoin   = sjoin_r;
    assign warp_ctl_if.barrier = barrier_r;

    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_commit_if
        assign commit_if.data.data[i] = `XLEN'(dvstack_ptr);
    end

endmodule
