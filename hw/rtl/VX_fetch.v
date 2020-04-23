`include "VX_define.vh"

module VX_fetch (
    input  wire                   clk,
    input  wire                   reset,
    VX_wstall_if                  wstall_if,
    VX_join_if                    join_if,
    input  wire                   schedule_delay,
    input  wire                   icache_stage_delay,
    input  wire[`NW_BITS-1:0]     icache_stage_wid,
    input  wire[`NUM_THREADS-1:0] icache_stage_valids,

    output wire           ebreak,
    VX_jal_rsp_if         jal_rsp_if,
    VX_branch_rsp_if      branch_rsp_if,
    VX_inst_meta_if       fe_inst_meta_fi,
    VX_warp_ctl_if        warp_ctl_if
);

    wire[`NUM_THREADS-1:0] thread_mask;
    wire[`NW_BITS-1:0]     warp_num;
    wire[31:0]             warp_pc;
    wire                   scheduled_warp;

    wire pipe_stall;

    // Only reason this is there is because there is a hidden assumption that decode is exactly after fetch

    // Locals

    assign pipe_stall = schedule_delay || icache_stage_delay;

    VX_warp_sched warp_sched (
        .clk              (clk),
        .reset            (reset),
        .stall            (pipe_stall),

        .is_barrier       (warp_ctl_if.is_barrier),
        .barrier_id       (warp_ctl_if.barrier_id),
        .num_warps        (warp_ctl_if.num_warps),
        .barrier_warp_num (warp_ctl_if.warp_num),

        // Wspawn
        .wspawn           (warp_ctl_if.wspawn),
        .wsapwn_pc        (warp_ctl_if.wspawn_pc),
        .wspawn_new_active(warp_ctl_if.wspawn_new_active),
        
        // CTM
        .ctm              (warp_ctl_if.change_mask),
        .ctm_mask         (warp_ctl_if.thread_mask),
        .ctm_warp_num     (warp_ctl_if.warp_num),

        // WHALT
        .whalt            (warp_ctl_if.ebreak),
        .whalt_warp_num   (warp_ctl_if.warp_num),

        // Wstall
        .wstall           (wstall_if.wstall),
        .wstall_warp_num  (wstall_if.warp_num),

        // Lock/release Stuff
        .icache_stage_valids(icache_stage_valids),
        .icache_stage_wid   (icache_stage_wid),

        // Join
        .is_join           (join_if.is_join),
        .join_warp_num     (join_if.join_warp_num),

        // Split
        .is_split          (warp_ctl_if.is_split),
        .dont_split        (warp_ctl_if.dont_split),
        .split_new_mask    (warp_ctl_if.split_new_mask),
        .split_later_mask  (warp_ctl_if.split_later_mask),
        .split_save_pc     (warp_ctl_if.split_save_pc),
        .split_warp_num    (warp_ctl_if.warp_num),

        // JAL
        .jal              (jal_rsp_if.jal),
        .jal_dest         (jal_rsp_if.jal_dest),
        .jal_warp_num     (jal_rsp_if.jal_warp_num),

        // Branch
        .branch_valid     (branch_rsp_if.valid_branch),
        .branch_dir       (branch_rsp_if.branch_dir),
        .branch_dest      (branch_rsp_if.branch_dest),
        .branch_warp_num  (branch_rsp_if.branch_warp_num),

        // Outputs
        .thread_mask      (thread_mask),
        .warp_num         (warp_num),
        .warp_pc          (warp_pc),
        .ebreak           (ebreak),
        .scheduled_warp   (scheduled_warp)
    );

    assign fe_inst_meta_fi.warp_num    = warp_num;
    assign fe_inst_meta_fi.valid       = thread_mask;
    assign fe_inst_meta_fi.instruction = 32'h0;
    assign fe_inst_meta_fi.inst_pc     = warp_pc;
`DEBUG_BEGIN
    wire start_mat_add = scheduled_warp && (warp_pc == 32'h80000ed8) && (warp_num == 0);
    wire end_mat_add   = scheduled_warp && (warp_pc == 32'h80000fbc) && (warp_num == 0);
`DEBUG_END

endmodule