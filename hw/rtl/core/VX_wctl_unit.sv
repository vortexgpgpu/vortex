`include "VX_define.vh"
`include "VX_gpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
`IGNORE_WARNINGS_END

module VX_wctl_unit #(
    parameter OUTPUT_REG = 0
) (
    input wire              clk,
    input wire              reset,

    // Inputs
    VX_gpu_exe_if.slave     gpu_exe_if,
    
    // Outputs
    VX_warp_ctl_if.master   warp_ctl_if,
    VX_commit_if.master     commit_if
);

    localparam UUID_WIDTH = `UP(`UUID_BITS);
    localparam NW_WIDTH   = `UP(`NW_BITS);
    
    gpu_tmc_t       tmc;
    gpu_wspawn_t    wspawn;    
    gpu_split_t     split;
    gpu_join_t      sjoin;
    gpu_barrier_t   barrier;
    
    wire [`XLEN-1:0] rs1_data = gpu_exe_if.rs1_data[gpu_exe_if.tid];
    wire [`XLEN-1:0] rs2_data = gpu_exe_if.rs2_data[gpu_exe_if.tid];
    
    wire [`NUM_THREADS-1:0] taken;
    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign taken[i] = gpu_exe_if.rs1_data[i][0];
    end

    wire is_wspawn = (gpu_exe_if.op_type == `INST_GPU_WSPAWN);
    wire is_tmc    = (gpu_exe_if.op_type == `INST_GPU_TMC);
    wire is_pred   = (gpu_exe_if.op_type == `INST_GPU_PRED);
    wire is_split  = (gpu_exe_if.op_type == `INST_GPU_SPLIT);
    wire is_join   = (gpu_exe_if.op_type == `INST_GPU_JOIN);
    wire is_bar    = (gpu_exe_if.op_type == `INST_GPU_BAR);

    assign warp_ctl_if.valid   = gpu_exe_if.valid && gpu_exe_if.ready;
    assign warp_ctl_if.wid     = gpu_exe_if.wid;
    assign warp_ctl_if.tmc     = tmc;
    assign warp_ctl_if.wspawn  = wspawn;
    assign warp_ctl_if.split   = split;
    assign warp_ctl_if.sjoin   = sjoin;
    assign warp_ctl_if.barrier = barrier;

    // tmc

    wire [`NUM_THREADS-1:0] then_tmask = gpu_exe_if.tmask & taken;
    wire [`NUM_THREADS-1:0] pred_mask = (then_tmask != 0) ? then_tmask : gpu_exe_if.tmask;

    assign tmc.valid = is_tmc || is_pred;
    assign tmc.tmask = is_pred ? pred_mask : rs1_data[`NUM_THREADS-1:0];

    // wspawn

    wire [`XLEN-1:0] wspawn_pc = rs2_data;
    wire [`NUM_WARPS-1:0] wspawn_wmask;
    for (genvar i = 0; i < `NUM_WARPS; ++i) begin
        assign wspawn_wmask[i] = (i < rs1_data[31:0]);
    end
    assign wspawn.valid     = is_wspawn;
    assign wspawn.wmask     = wspawn_wmask;
    assign wspawn.pc        = wspawn_pc;

    // split
    
    assign split.valid      = is_split;
    assign split.taken      = taken;
    assign split.tmask      = gpu_exe_if.tmask;
    assign split.next_pc    = gpu_exe_if.next_PC;

    // join

    assign sjoin.valid      = is_join;   
    assign sjoin.stack_ptr  = `PD_STACK_SIZEW'(rs1_data);

    // barrier
    assign barrier.valid    = is_bar;
    assign barrier.id       = rs1_data[`NB_BITS-1:0];
    assign barrier.is_global = rs1_data[31];
    assign barrier.size_m1  = $bits(barrier.size_m1)'(rs2_data[31:0] - 1);

    // response

    wire [`PD_STACK_SIZEW-1:0] rsp_data;
    
    VX_skid_buffer #(
        .DATAW    (UUID_WIDTH + NW_WIDTH + `NUM_THREADS + `XLEN + `NR_BITS + 1 + `PD_STACK_SIZEW),
        .PASSTHRU (OUTPUT_REG == 0)
    ) rsp_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (gpu_exe_if.valid),
        .ready_in  (gpu_exe_if.ready),
        .data_in   ({gpu_exe_if.uuid, gpu_exe_if.wid, gpu_exe_if.tmask, gpu_exe_if.PC, gpu_exe_if.rd, gpu_exe_if.wb, warp_ctl_if.split_ret}),
        .data_out  ({commit_if.uuid,  commit_if.wid,  commit_if.tmask,  commit_if.PC,  commit_if.rd,  commit_if.wb,  rsp_data}),
        .valid_out (commit_if.valid),
        .ready_out (commit_if.ready)
    );
    
    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign commit_if.data[i] = `XLEN'(rsp_data);
    end
    
endmodule
