`include "VX_define.vh"

module VX_gpu_unit #(
    parameter CORE_ID = 0
) (
    `SCOPE_IO_VX_gpu_unit
    
    input wire      clk,
    input wire      reset,

    // Inputs
    VX_gpu_req_if   gpu_req_if,

`ifdef EXT_TEX_ENABLE
    VX_tex_csr_if   tex_csr_if,

    VX_dcache_core_req_if dcache_req_if,
    VX_dcache_core_rsp_if dcache_rsp_if,
`endif

    // Outputs
    VX_warp_ctl_if  warp_ctl_if,
    VX_commit_if    gpu_commit_if
);

    `UNUSED_PARAM (CORE_ID)

    wire                          rsp_valid;
    wire [`NW_BITS-1:0]           rsp_wid;
    wire [`NUM_THREADS-1:0]       rsp_tmask;
    wire [31:0]                   rsp_PC;
    wire [`NR_BITS-1:0]           rsp_rd;   
    wire                          rsp_wb; 
    wire [`NUM_THREADS-1:0][31:0] rsp_data;

    gpu_tmc_t       tmc;
    gpu_wspawn_t    wspawn;
    gpu_barrier_t   barrier;
    gpu_split_t     split;
    
    wire [(`NUM_THREADS * 32)-1:0] warp_ctl_data;
    wire is_warp_ctl;
    
    wire stall_in, stall_out;
    
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

    wire [31:0] wspawn_pc = gpu_req_if.rs2_data[0];
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
    assign barrier.size_m1 = (`NW_BITS)'(gpu_req_if.rs2_data[0] - 1);       

    // pack warp ctl result
    `IGNORE_WARNINGS_BEGIN
    assign warp_ctl_data = {tmc, wspawn, barrier, split};
    `IGNORE_WARNINGS_END

    // texture

`ifdef EXT_TEX_ENABLE
    
    VX_tex_req_if   tex_req_if;
    VX_tex_rsp_if   tex_rsp_if;    

    wire is_tex = (gpu_req_if.op_type == `GPU_TEX);

    assign tex_req_if.valid = gpu_req_if.valid && is_tex;
    assign tex_req_if.wid   = gpu_req_if.wid;
    assign tex_req_if.tmask = gpu_req_if.tmask;
    assign tex_req_if.PC    = gpu_req_if.PC;
    assign tex_req_if.rd    = gpu_req_if.rd;
    assign tex_req_if.wb    = gpu_req_if.wb;

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        assign tex_req_if.u[i] = gpu_req_if.rs1_data[i];
        assign tex_req_if.v[i] = gpu_req_if.rs2_data[i];
        assign tex_req_if.lod[i] = gpu_req_if.rs3_data[i][31:8];
        assign tex_req_if.t[i] = gpu_req_if.rs3_data[i][7:0];
    end

    VX_tex_unit #(
        .CORE_ID(CORE_ID)
    ) texture_unit (
        .clk        (clk),
        .reset      (reset),
        .tex_req_if (tex_req_if),
        .tex_csr_if (tex_csr_if),
        .tex_rsp_if (tex_rsp_if),
        .dcache_req_if (dcache_req_if),
        .dcache_rsp_if (dcache_rsp_if)
    );

    assign tex_rsp_if.ready = !stall_out;

    assign stall_in = (is_tex && ~tex_req_if.ready)
                   || (~is_tex && (tex_rsp_if.valid || stall_out));

    assign is_warp_ctl = !(is_tex || tex_rsp_if.valid);

    assign rsp_valid = tex_rsp_if.valid || (gpu_req_if.valid && ~is_tex);
    assign rsp_wid   = tex_rsp_if.valid ? tex_rsp_if.wid : gpu_req_if.wid;
    assign rsp_tmask = tex_rsp_if.valid ? tex_rsp_if.tmask : gpu_req_if.tmask;
    assign rsp_PC    = tex_rsp_if.valid ? tex_rsp_if.PC : gpu_req_if.PC;
    assign rsp_rd    = tex_rsp_if.rd;
    assign rsp_wb    = tex_rsp_if.valid && tex_rsp_if.wb;
    assign rsp_data  = tex_rsp_if.valid ? tex_rsp_if.data : warp_ctl_data;
    
`else

    assign stall_in = stall_out;
    assign is_warp_ctl = 1;

    assign rsp_valid = gpu_req_if.valid;
    assign rsp_wid   = gpu_req_if.wid;
    assign rsp_tmask = gpu_req_if.tmask;
    assign rsp_PC    = gpu_req_if.PC;
    assign rsp_rd    = 0;
    assign rsp_wb    = 0;
    assign rsp_data  = warp_ctl_data;   

    `UNUSED_VAR (gpu_req_if.rs2_data)
    `UNUSED_VAR (gpu_req_if.rs3_data)
    `UNUSED_VAR (gpu_req_if.wb)
    `UNUSED_VAR (gpu_req_if.rd)

`endif

    wire is_warp_ctl_r;

    // output
    assign stall_out = ~gpu_commit_if.ready && gpu_commit_if.valid;

    VX_pipe_register #(
        .DATAW  (1 + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32) + 1),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall_out),
        .data_in  ({rsp_valid,           rsp_wid,           rsp_tmask,           rsp_PC,           rsp_rd,           rsp_wb,           rsp_data,           is_warp_ctl}),
        .data_out ({gpu_commit_if.valid, gpu_commit_if.wid, gpu_commit_if.tmask, gpu_commit_if.PC, gpu_commit_if.rd, gpu_commit_if.wb, gpu_commit_if.data, is_warp_ctl_r})
    );  

    assign gpu_commit_if.eop = 1'b1;

    // warp control reponse
     
    `IGNORE_WARNINGS_BEGIN
    assign {warp_ctl_if.tmc, warp_ctl_if.wspawn, warp_ctl_if.barrier, warp_ctl_if.split} = gpu_commit_if.data;
    `IGNORE_WARNINGS_END
    assign warp_ctl_if.valid = gpu_commit_if.valid && gpu_commit_if.ready && is_warp_ctl_r;
    assign warp_ctl_if.wid   = gpu_commit_if.wid;    

    // can accept new request?
    assign gpu_req_if.ready = ~stall_in;

    `SCOPE_ASSIGN (gpu_rsp_valid, warp_ctl_if.valid);
    `SCOPE_ASSIGN (gpu_rsp_wid, warp_ctl_if.wid);
    `SCOPE_ASSIGN (gpu_rsp_tmc, warp_ctl_if.tmc);
    `SCOPE_ASSIGN (gpu_rsp_wspawn, warp_ctl_if.wspawn);          
    `SCOPE_ASSIGN (gpu_rsp_split, warp_ctl_if.split);
    `SCOPE_ASSIGN (gpu_rsp_barrier, warp_ctl_if.barrier);

endmodule