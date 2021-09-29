`include "VX_define.vh"

module VX_gpu_unit #(
    parameter CORE_ID = 0
) (
    `SCOPE_IO_VX_gpu_unit
    
    input wire          clk,
    input wire          reset,

    // Inputs
    VX_gpu_req_if.slave gpu_req_if,

    // Outputs
    VX_warp_ctl_if.master warp_ctl_if,
    VX_commit_if.master gpu_commit_if
);
    import gpu_types::*;

    `UNUSED_PARAM (CORE_ID)
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

    gpu_tmc_t       tmc;
    gpu_wspawn_t    wspawn;
    gpu_barrier_t   barrier;
    gpu_split_t     split;
    
    wire is_wspawn = (gpu_req_if.op_type == `INST_GPU_WSPAWN);
    wire is_tmc    = (gpu_req_if.op_type == `INST_GPU_TMC);
    wire is_split  = (gpu_req_if.op_type == `INST_GPU_SPLIT);
    wire is_bar    = (gpu_req_if.op_type == `INST_GPU_BAR);
    wire is_pred   = (gpu_req_if.op_type == `INST_GPU_PRED);

    wire [31:0] rs1_data = gpu_req_if.rs1_data[gpu_req_if.tid];

    wire [`NUM_THREADS-1:0] taken_tmask;
    wire [`NUM_THREADS-1:0] not_taken_tmask;

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        wire taken = gpu_req_if.rs1_data[i][0];
        assign taken_tmask[i]     = gpu_req_if.tmask[i] & taken;
        assign not_taken_tmask[i] = gpu_req_if.tmask[i] & ~taken;
    end

    // tmc

    wire [`NUM_THREADS-1:0] pred_mask = (taken_tmask != 0) ? taken_tmask : gpu_req_if.tmask;

    assign tmc.valid = is_tmc || is_pred;
    assign tmc.tmask = is_pred ? pred_mask : rs1_data[`NUM_THREADS-1:0];

    // wspawn

    wire [31:0] wspawn_pc = gpu_req_if.rs2_data;
    wire [`NUM_WARPS-1:0] wspawn_wmask;
    for (genvar i = 0; i < `NUM_WARPS; i++) begin
        assign wspawn_wmask[i] = (i < rs1_data);
    end
    assign wspawn.valid = is_wspawn;
    assign wspawn.wmask = wspawn_wmask;
    assign wspawn.pc    = wspawn_pc;

    // split

    assign split.valid      = is_split;
    assign split.diverged   = (| taken_tmask) && (| not_taken_tmask);
    assign split.then_tmask = taken_tmask;
    assign split.else_tmask = not_taken_tmask;
    assign split.pc         = gpu_req_if.next_PC;

    // barrier
    
    assign barrier.valid   = is_bar;
    assign barrier.id      = rs1_data[`NB_BITS-1:0];
    assign barrier.size_m1 = (`NW_BITS)'(gpu_req_if.rs2_data - 1);

    // output
    
    wire stall = ~gpu_commit_if.ready && gpu_commit_if.valid;

    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + `GPU_TMC_BITS + `GPU_WSPAWN_BITS + `GPU_SPLIT_BITS + `GPU_BARRIER_BITS),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall),
        .data_in  ({gpu_req_if.valid,    gpu_req_if.wid,    gpu_req_if.tmask,    gpu_req_if.PC,    gpu_req_if.rd,    gpu_req_if.wb,    tmc,             wspawn,             split,             barrier}),
        .data_out ({gpu_commit_if.valid, gpu_commit_if.wid, gpu_commit_if.tmask, gpu_commit_if.PC, gpu_commit_if.rd, gpu_commit_if.wb, warp_ctl_if.tmc, warp_ctl_if.wspawn, warp_ctl_if.split, warp_ctl_if.barrier})
    );

    assign gpu_commit_if.eop = 1'b1;

    assign warp_ctl_if.valid = gpu_commit_if.valid && gpu_commit_if.ready;
    assign warp_ctl_if.wid   = gpu_commit_if.wid;

    // can accept new request?
    assign gpu_req_if.ready = ~stall;

    `SCOPE_ASSIGN (gpu_rsp_valid, warp_ctl_if.valid);
    `SCOPE_ASSIGN (gpu_rsp_wid, warp_ctl_if.wid);
    `SCOPE_ASSIGN (gpu_rsp_tmc, warp_ctl_if.tmc);
    `SCOPE_ASSIGN (gpu_rsp_wspawn, warp_ctl_if.wspawn);          
    `SCOPE_ASSIGN (gpu_rsp_split, warp_ctl_if.split);
    `SCOPE_ASSIGN (gpu_rsp_barrier, warp_ctl_if.barrier);

endmodule