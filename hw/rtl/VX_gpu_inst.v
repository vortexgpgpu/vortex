`include "VX_define.vh"

module VX_gpu_inst (
    // Input
    VX_gpu_inst_req_if    gpu_inst_req_if,

    // Output
    VX_warp_ctl_if        warp_ctl_if
);
    wire[`NUM_THREADS-1:0] curr_valids = gpu_inst_req_if.valid;
    wire is_split                      = gpu_inst_req_if.is_split;

    wire[`NUM_THREADS-1:0] tmc_new_mask;
    wire all_threads = `NUM_THREADS < gpu_inst_req_if.a_reg_data[0];
    
    genvar i;
    generate
    for (i = 0; i < `NUM_THREADS; i++) begin : tmc_new_mask_init
        assign tmc_new_mask[i] = all_threads ? 1 : i < gpu_inst_req_if.a_reg_data[0];
    end
    endgenerate

    wire valid_inst = (| curr_valids);

    assign warp_ctl_if.warp_num    = gpu_inst_req_if.warp_num;
    assign warp_ctl_if.change_mask = gpu_inst_req_if.is_tmc && valid_inst;
    assign warp_ctl_if.thread_mask = gpu_inst_req_if.is_tmc ? tmc_new_mask : 0;

    assign warp_ctl_if.whalt = warp_ctl_if.change_mask && (0 == warp_ctl_if.thread_mask);

    wire       wspawn     = gpu_inst_req_if.is_wspawn && valid_inst;
    wire[31:0] wspawn_pc  = gpu_inst_req_if.rd2;
    wire       all_active = `NUM_WARPS < gpu_inst_req_if.a_reg_data[0];
    wire[`NUM_WARPS-1:0] wspawn_new_active;

    generate
    for (i = 0; i < `NUM_WARPS; i++) begin : wspawn_new_active_init
        assign wspawn_new_active[i] = all_active ? 1 : i < gpu_inst_req_if.a_reg_data[0];
    end
    endgenerate

    assign warp_ctl_if.is_barrier = gpu_inst_req_if.is_barrier && valid_inst;
    assign warp_ctl_if.barrier_id = gpu_inst_req_if.a_reg_data[0];

`DEBUG_BEGIN
    wire[31:0] num_warps_m1 = gpu_inst_req_if.rd2 - 1;
`DEBUG_END

    assign warp_ctl_if.num_warps  = num_warps_m1[$clog2(`NUM_WARPS):0];

    assign warp_ctl_if.wspawn            = wspawn;
    assign warp_ctl_if.wspawn_pc         = wspawn_pc;
    assign warp_ctl_if.wspawn_new_active = wspawn_new_active;

    wire[`NUM_THREADS-1:0] split_new_use_mask;
    wire[`NUM_THREADS-1:0] split_new_later_mask;

    generate
    for (i = 0; i < `NUM_THREADS; i++) begin : masks_init
        wire curr_bool = (gpu_inst_req_if.a_reg_data[i] == 32'b1);
        assign split_new_use_mask[i]   = curr_valids[i] & (curr_bool);
        assign split_new_later_mask[i] = curr_valids[i] & (!curr_bool);
    end
    endgenerate

    wire[$clog2(`NUM_THREADS):0] num_valids;

    VX_countones #(
        .N(`NUM_THREADS)
    ) valids_counter (
        .valids(curr_valids),
        .count (num_valids)
    );

    // wire[`NW_BITS-1:0] num_valids = $countones(curr_valids);
    
    assign warp_ctl_if.is_split         = is_split && (num_valids > 1);
    assign warp_ctl_if.dont_split       = warp_ctl_if.is_split && ((split_new_use_mask == 0) || (split_new_use_mask == {`NUM_THREADS{1'b1}}));
    assign warp_ctl_if.split_new_mask   = split_new_use_mask;
    assign warp_ctl_if.split_later_mask = split_new_later_mask;
    assign warp_ctl_if.split_save_pc    = gpu_inst_req_if.pc_next;
    assign warp_ctl_if.split_warp_num   = gpu_inst_req_if.warp_num;

    // gpu_inst_req_if.is_wspawn
    // gpu_inst_req_if.is_split
    // gpu_inst_req_if.is_barrier

endmodule