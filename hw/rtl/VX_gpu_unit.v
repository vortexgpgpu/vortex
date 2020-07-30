`include "VX_define.vh"

module VX_gpu_unit #(
    parameter CORE_ID = 0
) (
    // Input
    VX_gpu_req_if    gpu_req_if,

    // Output
    VX_warp_ctl_if   warp_ctl_if,
    VX_exu_to_cmt_if gpu_commit_if
);
    
    wire is_wspawn = (gpu_req_if.gpu_op == `GPU_WSPAWN);
    wire is_tmc    = (gpu_req_if.gpu_op == `GPU_TMC);
    wire is_split  = (gpu_req_if.gpu_op == `GPU_SPLIT);
    wire is_bar    = (gpu_req_if.gpu_op == `GPU_BAR);

    wire gpu_req_fire = gpu_req_if.valid && gpu_commit_if.ready;

    assign warp_ctl_if.warp_num = gpu_req_if.warp_num;
    
    // tmc

    genvar i;

    wire [`NUM_THREADS-1:0] tmc_new_mask;           
    for (i = 0; i < `NUM_THREADS; i++) begin
        assign tmc_new_mask[i] = (i < gpu_req_if.rs1_data[0]);
    end    
    assign warp_ctl_if.change_mask = is_tmc && gpu_req_fire;
    assign warp_ctl_if.thread_mask = tmc_new_mask;

    // barrier
    
    assign warp_ctl_if.is_barrier = is_bar && gpu_req_fire;
    assign warp_ctl_if.barrier_id = gpu_req_if.rs1_data[0][`NB_BITS-1:0];
    assign warp_ctl_if.barrier_num_warps = (`NW_BITS+1)'(gpu_req_if.rs2_data - 1);

    // wspawn

    wire [31:0] wspawn_pc = gpu_req_if.rs2_data;
    wire [`NUM_WARPS-1:0] wspawn_wmask;
    for (i = 0; i < `NUM_WARPS; i++) begin
        assign wspawn_wmask[i] = (i < gpu_req_if.rs1_data[0]);
    end
    assign warp_ctl_if.wspawn       = is_wspawn && gpu_req_fire;
    assign warp_ctl_if.wspawn_pc    = wspawn_pc;
    assign warp_ctl_if.wspawn_wmask = wspawn_wmask;

    // split

    wire[`NUM_THREADS-1:0] split_new_use_mask;
    wire[`NUM_THREADS-1:0] split_new_later_mask;

    for (i = 0; i < `NUM_THREADS; i++) begin
        wire curr_bool = (gpu_req_if.rs1_data[i] == 32'b1);
        assign split_new_use_mask[i]   = gpu_req_if.thread_mask[i] & (curr_bool);
        assign split_new_later_mask[i] = gpu_req_if.thread_mask[i] & (!curr_bool);
    end

    wire [`NT_BITS:0] num_valids;

    VX_countones #(
        .N(`NUM_THREADS)
    ) valids_counter (
        .valids(gpu_req_if.thread_mask),
        .count (num_valids)
    );
    
    assign warp_ctl_if.is_split         = is_split && (num_valids > 1) && gpu_req_fire;
    assign warp_ctl_if.do_split         = (split_new_use_mask != 0) && (split_new_use_mask != {`NUM_THREADS{1'b1}});
    assign warp_ctl_if.split_new_mask   = split_new_use_mask;
    assign warp_ctl_if.split_later_mask = split_new_later_mask;
    assign warp_ctl_if.split_save_pc    = gpu_req_if.next_PC;

    // commit
    assign gpu_commit_if.valid     = gpu_req_if.valid;
    assign gpu_commit_if.issue_tag = gpu_req_if.issue_tag;
    assign gpu_commit_if.data      = 0;
    assign gpu_req_if.ready = gpu_commit_if.ready;

endmodule