`include "VX_define.vh"

module VX_gpu_unit #(
    parameter CORE_ID = 0
) (
    // Input
    VX_gpu_req_if   gpu_req_if,

    // Output
    VX_warp_ctl_if  warp_ctl_if,
    VX_commit_if    gpu_commit_if
);
    wire [`NUM_THREADS-1:0] curr_valids = gpu_req_if.valid;
    wire is_wspawn = (gpu_req_if.gpu_op == `GPU_WSPAWN);
    wire is_tmc    = (gpu_req_if.gpu_op == `GPU_TMC);
    wire is_split  = (gpu_req_if.gpu_op == `GPU_SPLIT);
    wire is_bar    = (gpu_req_if.gpu_op == `GPU_BAR);

    wire [`NUM_THREADS-1:0] tmc_new_mask;
    wire all_threads = `NUM_THREADS < gpu_req_if.rs1_data[0];
    
    genvar i;
    for (i = 0; i < `NUM_THREADS; i++) begin : tmc_new_mask_init
        assign tmc_new_mask[i] = all_threads ? 1 : i < gpu_req_if.rs1_data[0];
    end

    wire valid_inst = (| curr_valids);

    assign warp_ctl_if.warp_num    = gpu_req_if.warp_num;
    
    assign warp_ctl_if.change_mask = is_tmc && valid_inst;
    assign warp_ctl_if.thread_mask = is_tmc ? tmc_new_mask : 0;

    assign warp_ctl_if.whalt = warp_ctl_if.change_mask && (0 == warp_ctl_if.thread_mask);

    wire wspawn = is_wspawn && valid_inst;
    wire [31:0] wspawn_pc = gpu_req_if.rs2_data;
    wire all_active = `NUM_WARPS < gpu_req_if.rs1_data[0];
    wire [`NUM_WARPS-1:0] wspawn_new_active;

    for (i = 0; i < `NUM_WARPS; i++) begin : wspawn_new_active_init
        assign wspawn_new_active[i] = all_active ? 1 : i < gpu_req_if.rs1_data[0];
    end

    assign warp_ctl_if.is_barrier = is_bar && valid_inst;
    assign warp_ctl_if.barrier_id = gpu_req_if.rs1_data[0][`NB_BITS-1:0];

    assign warp_ctl_if.num_warps = (`NW_BITS+1)'(gpu_req_if.rs2_data - 1);

    assign warp_ctl_if.wspawn            = wspawn;
    assign warp_ctl_if.wspawn_pc         = wspawn_pc;
    assign warp_ctl_if.wspawn_new_active = wspawn_new_active;

    wire[`NUM_THREADS-1:0] split_new_use_mask;
    wire[`NUM_THREADS-1:0] split_new_later_mask;

    for (i = 0; i < `NUM_THREADS; i++) begin : masks_init
        wire curr_bool = (gpu_req_if.rs1_data[i] == 32'b1);
        assign split_new_use_mask[i]   = curr_valids[i] & (curr_bool);
        assign split_new_later_mask[i] = curr_valids[i] & (!curr_bool);
    end

    wire [`NT_BITS:0] num_valids;

    VX_countones #(
        .N(`NUM_THREADS)
    ) valids_counter (
        .valids(curr_valids),
        .count (num_valids)
    );
    
    assign warp_ctl_if.is_split         = is_split && (num_valids > 1);
    assign warp_ctl_if.do_split         = (split_new_use_mask != 0) && (split_new_use_mask != {`NUM_THREADS{1'b1}});
    assign warp_ctl_if.split_new_mask   = split_new_use_mask;
    assign warp_ctl_if.split_later_mask = split_new_later_mask;
    assign warp_ctl_if.split_save_pc    = gpu_req_if.next_PC;

    assign gpu_req_if.ready = 1'b1; // has no stalls

    // commit
    assign gpu_commit_if.valid    = gpu_req_if.valid;
    assign gpu_commit_if.warp_num = gpu_req_if.warp_num;
    assign gpu_commit_if.curr_PC  = gpu_req_if.curr_PC;
    assign gpu_commit_if.wb       = `WB_NO;    

endmodule