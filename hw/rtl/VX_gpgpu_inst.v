`include "VX_define.vh"

module VX_gpgpu_inst (
    // Input
    VX_gpu_inst_req_if    gpu_inst_req_if,

    // Output
    VX_warp_ctl_if        warp_ctl_if
);
    wire[`NUM_THREADS-1:0] curr_valids = gpu_inst_req_if.valid;
    wire is_split                      = (gpu_inst_req_if.is_split);

    wire[`NUM_THREADS-1:0] tmc_new_mask;
    wire all_threads = `NUM_THREADS < gpu_inst_req_if.a_reg_data[0];
    
    genvar curr_t;
    generate
    for (curr_t = 0; curr_t < `NUM_THREADS; curr_t=curr_t+1) begin : tmc_new_mask_init
        assign tmc_new_mask[curr_t] = all_threads ? 1 : curr_t < gpu_inst_req_if.a_reg_data[0];
    end
    endgenerate

    wire valid_inst = (|curr_valids);

    assign warp_ctl_if.warp_num    = gpu_inst_req_if.warp_num;
    assign warp_ctl_if.change_mask = (gpu_inst_req_if.is_tmc) && valid_inst;
    assign warp_ctl_if.thread_mask = gpu_inst_req_if.is_tmc ? tmc_new_mask : 0;

    // assign warp_ctl_if.ebreak = (gpu_inst_req_if.a_reg_data[0] == 0) && valid_inst;
    assign warp_ctl_if.ebreak = warp_ctl_if.change_mask && (warp_ctl_if.thread_mask == 0);

    wire       wspawn     = gpu_inst_req_if.is_wspawn;
    wire[31:0] wspawn_pc  = gpu_inst_req_if.rd2;
    wire       all_active = `NUM_WARPS < gpu_inst_req_if.a_reg_data[0];
    wire[`NUM_WARPS-1:0] wspawn_new_active;

    genvar curr_w;
    generate
    for (curr_w = 0; curr_w < `NUM_WARPS; curr_w=curr_w+1) begin : wspawn_new_active_init
        assign wspawn_new_active[curr_w] = all_active ? 1 : curr_w < gpu_inst_req_if.a_reg_data[0];
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

    // VX_gpu_inst_req.pc
    genvar curr_s_t;
    generate
    for (curr_s_t = 0; curr_s_t < `NUM_THREADS; curr_s_t=curr_s_t+1) begin : masks_init
        wire curr_bool = (gpu_inst_req_if.a_reg_data[curr_s_t] == 32'b1);

        assign split_new_use_mask[curr_s_t]   = curr_valids[curr_s_t] & (curr_bool);
        assign split_new_later_mask[curr_s_t] = curr_valids[curr_s_t] & (!curr_bool);
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