`include "VX_define.vh"

module VX_dxa_group_tracker_top (
    input  wire        clk,
    input  wire        reset,
    input  wire        assert_on_drop,
    input  wire        issue_valid,
    input  wire        issue_query,
    input  wire        issue_wid,
    output wire [1:0]  issue_result,
    output wire [5:0]  issue_op_id,
    input  wire        commit_valid,
    input  wire        commit_wid,
    output wire [1:0]  commit_result,
    input  wire        completion_valid,
    output wire        completion_ready,
    input  wire        completion_wid,
    input  wire [1:0]  completion_epoch,
    input  wire [7:0]  completion_seq,
    input  wire [5:0]  completion_op,
    input  wire        wq_valid,
    input  wire        wq_wid,
    input  wire [4:0]  wq_n,
    output wire        wq_satisfied,
    output wire [1:0]  unlock_mask,
    input  wire        poison_valid,
    input  wire        poison_wid,
    input  wire        epoch_adv_valid,
    input  wire        epoch_adv_wid,
    output wire [1:0]  drained_mask,
    output wire [3:0]  epoch_flat,
    output wire [3:0]  sticky_flat,
    input  wire        sticky_clear_valid,
    input  wire        sticky_clear_wid,
    input  wire [1:0]  sticky_clear_mask,
    output wire [15:0] sealed_tail_flat,
    output wire [15:0] read_head_flat,
    output wire [5:0]  open_ops_flat,
    output wire [63:0] stale_flat,
    output wire [63:0] duplicate_flat,
    output wire [63:0] invalid_flat,
    output wire [7:0]  ring_live_flat,
    output wire [23:0] ring_remaining_flat,
    output wire [3:0]  context_live,
    output wire [31:0] context_stalls
);
    localparam NUM_WARPS = 2;
    localparam RING_DEPTH = 4;
    localparam NUM_CONTEXTS = 4;
    localparam REMAIN_W = 3;

    wire unused_dxa_pkg = |32'(VX_dxa_pkg::DXA_LMEM_WORD_SIZE)
                        | |32'(VX_dxa_pkg::DXA_LMEM_ADDR_W)
                        | |32'(VX_dxa_pkg::DXA_DESC_SLOT_W)
                        | |32'(VX_dxa_pkg::DXA_DESC_META_TOTAL_BITS)
                        | |32'(VX_dxa_pkg::DXA_DEST_ROWMAJOR)
                        | |32'(VX_dxa_pkg::DXA_DEST_KMAJOR)
                        | |32'(VX_dxa_pkg::DXA_DEST_BLOCKMAJOR)
                        | |32'(VX_dxa_pkg::DXA_GROUP_DEPTH)
                        | |32'(VX_dxa_pkg::DXA_GROUP_COMPL_W);
    `UNUSED_VAR (unused_dxa_pkg)

    wire [1:0] epoch_w [NUM_WARPS];
    wire [1:0] sticky_w [NUM_WARPS];
    wire [7:0] tail_w [NUM_WARPS];
    wire [7:0] head_w [NUM_WARPS];
    wire [REMAIN_W-1:0] open_w [NUM_WARPS];
    wire [31:0] stale_w [NUM_WARPS];
    wire [31:0] duplicate_w [NUM_WARPS];
    wire [31:0] invalid_w [NUM_WARPS];
    wire [RING_DEPTH-1:0] ring_live_w [NUM_WARPS];
    wire [RING_DEPTH*REMAIN_W-1:0] ring_remaining_w [NUM_WARPS];

    VX_dxa_group_tracker #(
        .NUM_WARPS    (NUM_WARPS),
        .RING_DEPTH   (RING_DEPTH),
        .NUM_CONTEXTS (NUM_CONTEXTS),
        .CTX_GEN_W    (4),
        .OP_ID_W      (6)
    ) dut (
        .clk                 (clk),
        .reset               (reset),
        .assert_on_drop      (assert_on_drop),
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
        .epoch               (epoch_w),
        .sticky_status       (sticky_w),
        .sticky_clear_valid  (sticky_clear_valid),
        .sticky_clear_wid    (sticky_clear_wid),
        .sticky_clear_mask   (sticky_clear_mask),
        .obs_sealed_tail     (tail_w),
        .obs_read_head       (head_w),
        .obs_open_ops        (open_w),
        .obs_stale_drops     (stale_w),
        .obs_duplicate_drops (duplicate_w),
        .obs_invalid_drops   (invalid_w),
        .obs_ring_live       (ring_live_w),
        .obs_ring_remaining  (ring_remaining_w),
        .obs_context_live    (context_live),
        .obs_context_stalls  (context_stalls)
    );

    for (genvar w = 0; w < NUM_WARPS; ++w) begin : g_flatten
        assign epoch_flat[w*2 +: 2] = epoch_w[w];
        assign sticky_flat[w*2 +: 2] = sticky_w[w];
        assign sealed_tail_flat[w*8 +: 8] = tail_w[w];
        assign read_head_flat[w*8 +: 8] = head_w[w];
        assign open_ops_flat[w*REMAIN_W +: REMAIN_W] = open_w[w];
        assign stale_flat[w*32 +: 32] = stale_w[w];
        assign duplicate_flat[w*32 +: 32] = duplicate_w[w];
        assign invalid_flat[w*32 +: 32] = invalid_w[w];
        assign ring_live_flat[w*RING_DEPTH +: RING_DEPTH] = ring_live_w[w];
        assign ring_remaining_flat[w*RING_DEPTH*REMAIN_W +: RING_DEPTH*REMAIN_W]
            = ring_remaining_w[w];
    end

endmodule
