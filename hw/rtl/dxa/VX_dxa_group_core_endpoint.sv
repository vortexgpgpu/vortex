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

`ifdef VX_CFG_EXT_DXA_GROUP_ENABLE

// One endpoint per SM core. Logical group state is indexed by warp id, while
// every warp shares the same bounded physical operation-context pool.
module VX_dxa_group_core_endpoint import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter NUM_WARPS    = `VX_CFG_NUM_WARPS,
    parameter RING_DEPTH   = `VX_CFG_DXA_GROUP_DEPTH,
    parameter NUM_CONTEXTS = `VX_CFG_DXA_GROUP_CONTEXTS,
    parameter CTX_GEN_W    = `VX_CFG_DXA_GROUP_CTX_GEN_BITS,
    parameter WID_W        = `UP(`CLOG2(NUM_WARPS)),
    parameter WAIT_N_W     = 5,
    parameter REMAIN_W     = `CLOG2(NUM_CONTEXTS + 1)
) (
    input  wire                         clk,
    input  wire                         reset,

    input  wire                         issue_valid,
    input  wire                         issue_query,
    input  wire [WID_W-1:0]             issue_wid,
    output wire [1:0]                   issue_result,
    output wire [DXA_GROUP_EPOCH_W-1:0] issue_epoch,
    output wire [DXA_GROUP_SEQ_W-1:0]   issue_seq,
    output wire [DXA_GROUP_OPID_W-1:0]  issue_op_id,

    input  wire                         commit_valid,
    input  wire [WID_W-1:0]             commit_wid,
    output wire [1:0]                   commit_result,

    input  wire                         completion_valid,
    output wire                         completion_ready,
    input  wire [WID_W-1:0]             completion_wid,
    input  wire [DXA_GROUP_EPOCH_W-1:0] completion_epoch,
    input  wire [DXA_GROUP_SEQ_W-1:0]   completion_seq,
    input  wire [DXA_GROUP_OPID_W-1:0]  completion_op,

    input  wire                         wq_valid,
    input  wire [WID_W-1:0]             wq_wid,
    input  wire [WAIT_N_W-1:0]          wq_n,
    output wire                         wq_satisfied,
    output wire [NUM_WARPS-1:0]         unlock_mask,

    input  wire                         poison_valid,
    input  wire [WID_W-1:0]             poison_wid,
    input  wire                         epoch_adv_valid,
    input  wire [WID_W-1:0]             epoch_adv_wid,
    output wire [NUM_WARPS-1:0]         drained_mask,
    output wire [1:0]                   sticky_status [NUM_WARPS]
);
    wire [DXA_GROUP_EPOCH_W-1:0] epoch [NUM_WARPS];
    wire [DXA_GROUP_SEQ_W-1:0] sealed_tail [NUM_WARPS];
    wire [DXA_GROUP_SEQ_W-1:0] read_head [NUM_WARPS];
    wire [REMAIN_W-1:0] open_ops [NUM_WARPS];
    wire [31:0] stale_drops [NUM_WARPS];
    wire [31:0] duplicate_drops [NUM_WARPS];
    wire [31:0] invalid_drops [NUM_WARPS];
    wire [RING_DEPTH-1:0] ring_live [NUM_WARPS];
    wire [RING_DEPTH*REMAIN_W-1:0] ring_remaining [NUM_WARPS];
    wire [NUM_CONTEXTS-1:0] context_live;
    wire [31:0] context_stalls;

    assign issue_epoch = epoch[issue_wid];
    assign issue_seq   = sealed_tail[issue_wid];

    VX_dxa_group_tracker #(
        .NUM_WARPS    (NUM_WARPS),
        .RING_DEPTH   (RING_DEPTH),
        .NUM_CONTEXTS (NUM_CONTEXTS),
        .CTX_GEN_W    (CTX_GEN_W),
        .WAIT_N_W     (WAIT_N_W),
        .WID_W        (WID_W),
        .OP_ID_W      (DXA_GROUP_OPID_W),
        .REMAIN_W     (REMAIN_W)
    ) tracker (
        .clk                 (clk),
        .reset               (reset),
        .assert_on_drop      (1'b1),
        .issue_valid         (issue_valid),
        .issue_query         (issue_query),
        .issue_wid           (issue_wid),
        .issue_result        (issue_result),
        .issue_op_id         (issue_op_id),
        .commit_valid        (commit_valid),
        .commit_wid          (commit_wid),
        .commit_result       (commit_result),
        .completion_valid    (completion_valid),
        .completion_ready    (completion_ready),
        .completion_wid      (completion_wid),
        .completion_epoch    (completion_epoch),
        .completion_seq      (completion_seq),
        .completion_op       (completion_op),
        .wq_valid            (wq_valid),
        .wq_wid              (wq_wid),
        .wq_n                (wq_n),
        .wq_satisfied        (wq_satisfied),
        .unlock_mask         (unlock_mask),
        .poison_valid        (poison_valid),
        .poison_wid          (poison_wid),
        .epoch_adv_valid     (epoch_adv_valid),
        .epoch_adv_wid       (epoch_adv_wid),
        .drained_mask        (drained_mask),
        .epoch               (epoch),
        .sticky_status       (sticky_status),
        .sticky_clear_valid  (1'b0),
        .sticky_clear_wid    ('0),
        .sticky_clear_mask   ('0),
        .obs_sealed_tail     (sealed_tail),
        .obs_read_head       (read_head),
        .obs_open_ops        (open_ops),
        .obs_stale_drops     (stale_drops),
        .obs_duplicate_drops (duplicate_drops),
        .obs_invalid_drops   (invalid_drops),
        .obs_ring_live       (ring_live),
        .obs_ring_remaining  (ring_remaining),
        .obs_context_live    (context_live),
        .obs_context_stalls  (context_stalls)
    );

    for (genvar w = 0; w < NUM_WARPS; ++w) begin : g_unused_observers
        wire unused_observers = |read_head[w]
                              | |open_ops[w]
                              | |stale_drops[w]
                              | |duplicate_drops[w]
                              | |invalid_drops[w]
                              | |ring_live[w]
                              | |ring_remaining[w];
        `UNUSED_VAR (unused_observers)
    end
    wire unused_context_observers = |context_live | |context_stalls;
    `UNUSED_VAR (unused_context_observers)

endmodule

`endif
