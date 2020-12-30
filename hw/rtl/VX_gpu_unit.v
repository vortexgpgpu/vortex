`include "VX_define.vh"

module VX_gpu_unit #(
    parameter CORE_ID = 0
) (
    `SCOPE_IO_VX_gpu_unit
    
    input wire      clk,
    input wire      reset,

    // Inputs
    VX_gpu_req_if   gpu_req_if,

    // Outputs
    VX_warp_ctl_if  warp_ctl_if,
    VX_commit_if    gpu_commit_if
);
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

    gpu_tmc_t       tmc;
    gpu_wspawn_t    wspawn;
    gpu_barrier_t   barrier;
    gpu_split_t     split;
    
    wire is_wspawn = (gpu_req_if.op_type == `GPU_WSPAWN);
    wire is_tmc    = (gpu_req_if.op_type == `GPU_TMC);
    wire is_split  = (gpu_req_if.op_type == `GPU_SPLIT);
    wire is_bar    = (gpu_req_if.op_type == `GPU_BAR);

    // tmc

    wire [`NUM_THREADS-1:0] tmc_new_mask;           
    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        assign tmc_new_mask[i] = (i < gpu_req_if.rs1_data[0]);
    end    
    assign tmc.valid = is_tmc;
    assign tmc.tmask = tmc_new_mask;

    // wspawn

    wire [31:0] wspawn_pc = gpu_req_if.rs2_data;
    wire [`NUM_WARPS-1:0] wspawn_wmask;
    for (genvar i = 0; i < `NUM_WARPS; i++) begin
        assign wspawn_wmask[i] = (i < gpu_req_if.rs1_data[0]);
    end
    assign wspawn.valid = is_wspawn;
    assign wspawn.wmask = wspawn_wmask;
    assign wspawn.pc    = wspawn_pc;

    // split

    wire [`NUM_THREADS-1:0] split_then_mask;
    wire [`NUM_THREADS-1:0] split_else_mask;

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        wire taken = gpu_req_if.rs1_data[i][0];
        assign split_then_mask[i] = gpu_req_if.tmask[i] & taken;
        assign split_else_mask[i] = gpu_req_if.tmask[i] & ~taken;
    end

    assign split.valid     = is_split;
    assign split.diverged  = (| split_then_mask) && (| split_else_mask);
    assign split.then_mask = split_then_mask;
    assign split.else_mask = split_else_mask;
    assign split.pc        = gpu_req_if.next_PC;

    // barrier
    
    assign barrier.valid   = is_bar;
    assign barrier.id      = gpu_req_if.rs1_data[0][`NB_BITS-1:0];
    assign barrier.size_m1 = (`NW_BITS)'(gpu_req_if.rs2_data - 1);

    // output
    
    wire stall = ~gpu_commit_if.ready && gpu_commit_if.valid;

    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + `GPU_TMC_SIZE + `GPU_WSPAWN_SIZE + `GPU_SPLIT_SIZE + `GPU_BARRIER_SIZE),
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

    `SCOPE_ASSIGN (gpu_req_fire, gpu_req_if.valid && gpu_req_if.ready);
    `SCOPE_ASSIGN (gpu_req_wid, gpu_req_if.wid);
    `SCOPE_ASSIGN (gpu_req_tmask, gpu_req_if.tmask);
    `SCOPE_ASSIGN (gpu_req_op_type, gpu_req_if.op_type);
    `SCOPE_ASSIGN (gpu_req_rs1, gpu_req_if.rs1_data[0]); 
    `SCOPE_ASSIGN (gpu_req_rs2, gpu_req_if.rs2_data);
    `SCOPE_ASSIGN (gpu_rsp_valid, warp_ctl_if.valid);
    `SCOPE_ASSIGN (gpu_rsp_wid, warp_ctl_if.wid);
    `SCOPE_ASSIGN (gpu_rsp_tmc, warp_ctl_if.tmc);
    `SCOPE_ASSIGN (gpu_rsp_wspawn, warp_ctl_if.wspawn);          
    `SCOPE_ASSIGN (gpu_rsp_split, warp_ctl_if.split);
    `SCOPE_ASSIGN (gpu_rsp_barrier, warp_ctl_if.barrier);

endmodule