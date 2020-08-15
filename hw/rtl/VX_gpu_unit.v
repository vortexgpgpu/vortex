`include "VX_define.vh"

module VX_gpu_unit #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

    // Input
    VX_gpu_req_if    gpu_req_if,

    // Output
    VX_warp_ctl_if   warp_ctl_if,
    VX_exu_to_cmt_if gpu_commit_if
);
    gpu_tmc_t       tmc;
    gpu_wspawn_t    wspawn;
    gpu_barrier_t   barrier;
    gpu_split_t     split;
    
    wire is_wspawn = (gpu_req_if.op == `GPU_WSPAWN);
    wire is_tmc    = (gpu_req_if.op == `GPU_TMC);
    wire is_split  = (gpu_req_if.op == `GPU_SPLIT);
    wire is_bar    = (gpu_req_if.op == `GPU_BAR);

    wire gpu_req_fire = gpu_req_if.valid;
    
    // tmc

    wire [`NUM_THREADS-1:0] tmc_new_mask;           
    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        assign tmc_new_mask[i] = (i < gpu_req_if.rs1_data[0]);
    end    
    assign tmc.valid       = gpu_req_fire && is_tmc;
    assign tmc.thread_mask = tmc_new_mask;

    // wspawn

    wire [31:0] wspawn_pc = gpu_req_if.rs2_data;
    wire [`NUM_WARPS-1:0] wspawn_wmask;
    for (genvar i = 0; i < `NUM_WARPS; i++) begin
        assign wspawn_wmask[i] = (i < gpu_req_if.rs1_data[0]);
    end
    assign wspawn.valid = gpu_req_fire && is_wspawn;
    assign wspawn.wmask = wspawn_wmask;
    assign wspawn.pc    = wspawn_pc;

    // split

    wire [`NUM_THREADS-1:0] split_then_mask;
    wire [`NUM_THREADS-1:0] split_else_mask;

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        wire taken = gpu_req_if.rs1_data[i][0];
        assign split_then_mask[i] = gpu_req_if.thread_mask[i] & taken;
        assign split_else_mask[i] = gpu_req_if.thread_mask[i] & ~taken;
    end

    assign split.valid     = gpu_req_fire && is_split;
    assign split.diverged  = (| split_then_mask) && (| split_else_mask);
    assign split.then_mask = split_then_mask;
    assign split.else_mask = split_else_mask;
    assign split.pc        = gpu_req_if.curr_PC + 4;

    // barrier
    
    assign barrier.valid     = is_bar && gpu_req_fire;
    assign barrier.id        = gpu_req_if.rs1_data[0][`NB_BITS-1:0];
    assign barrier.num_warps = (`NW_BITS+1)'(gpu_req_if.rs2_data - 1);

    // output

    VX_generic_register #(
        .N(1 + `ISTAG_BITS + `NW_BITS + $bits(gpu_tmc_t) + $bits(gpu_wspawn_t) + $bits(gpu_split_t) + $bits(gpu_barrier_t))
    ) gpu_reg (
        .clk   (clk),
        .reset (reset),
        .stall (0),
        .flush (0),
        .in    ({gpu_req_if.valid,    gpu_req_if.issue_tag,    gpu_req_if.wid,  tmc,             wspawn,             split,             barrier}),
        .out   ({gpu_commit_if.valid, gpu_commit_if.issue_tag, warp_ctl_if.wid, warp_ctl_if.tmc, warp_ctl_if.wspawn, warp_ctl_if.split, warp_ctl_if.barrier})
    );

    assign gpu_req_if.ready = 1'b1;

endmodule