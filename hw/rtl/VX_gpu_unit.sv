`include "VX_define.vh"

module VX_gpu_unit #(
    parameter CORE_ID = 0
) (
    `SCOPE_IO_VX_gpu_unit
    
    input wire          clk,
    input wire          reset,

    // Inputs
    VX_gpu_req_if.slave gpu_req_if,

`ifdef EXT_TEX_ENABLE
    // PERF
`ifdef PERF_ENABLE
    VX_perf_tex_if.master   perf_tex_if,
`endif
    VX_tex_dcr_if.slave     tex_dcr_if,
    VX_dcache_req_if.master tcache_req_if,
    VX_dcache_rsp_if.slave  tcache_rsp_if,
`endif

    // Outputs
    VX_warp_ctl_if.master warp_ctl_if,
    VX_commit_if.master gpu_commit_if
);
    import gpu_types::*;

    `UNUSED_PARAM (CORE_ID)

    localparam WCTL_DATAW = `GPU_TMC_BITS + `GPU_WSPAWN_BITS + `GPU_SPLIT_BITS + `GPU_BARRIER_BITS;
    localparam RSP_DATAW  = `MAX(`NUM_THREADS * 32, WCTL_DATAW);
    localparam MUX_DATAW  = `UUID_BITS + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + RSP_DATAW + 1 + 1;

    wire                    rsp_valid;
    wire [`UUID_BITS-1:0]   rsp_uuid;
    wire [`NW_BITS-1:0]     rsp_wid;
    wire [`NUM_THREADS-1:0] rsp_tmask;
    wire [31:0]             rsp_PC;
    wire [`NR_BITS-1:0]     rsp_rd;   
    wire                    rsp_wb;
    wire [RSP_DATAW-1:0]    rsp_data, rsp_data_r;
    wire                    rsp_eop;
    wire                    rsp_is_wctl, rsp_is_wctl_r;

    wire stall_out;

    // Warp control block

    gpu_tmc_t       tmc;
    gpu_wspawn_t    wspawn;
    gpu_barrier_t   barrier;
    gpu_split_t     split;
    
    wire is_wspawn = (gpu_req_if.op_type == `INST_GPU_WSPAWN);
    wire is_tmc    = (gpu_req_if.op_type == `INST_GPU_TMC);
    wire is_split  = (gpu_req_if.op_type == `INST_GPU_SPLIT);
    wire is_bar    = (gpu_req_if.op_type == `INST_GPU_BAR);
    wire is_pred   = (gpu_req_if.op_type == `INST_GPU_PRED);

    wire is_wctl   = is_wspawn | is_tmc | is_split | is_bar | is_pred;

    wire [31:0] rs1_data = gpu_req_if.rs1_data[gpu_req_if.tid];
    wire [31:0] rs2_data = gpu_req_if.rs2_data[gpu_req_if.tid];
    
    wire [`NUM_THREADS-1:0] taken_tmask;
    wire [`NUM_THREADS-1:0] not_taken_tmask;

    for (genvar i = 0; i < `NUM_THREADS; i++) begin
        wire taken = (gpu_req_if.rs1_data[i] != 0);
        assign taken_tmask[i]     = gpu_req_if.tmask[i] & taken;
        assign not_taken_tmask[i] = gpu_req_if.tmask[i] & ~taken;
    end

    // tmc

    wire [`NUM_THREADS-1:0] pred_mask = (taken_tmask != 0) ? taken_tmask : gpu_req_if.tmask;

    assign tmc.valid = is_tmc || is_pred;
    assign tmc.tmask = is_pred ? pred_mask : rs1_data[`NUM_THREADS-1:0];

    // wspawn

    wire [31:0] wspawn_pc = rs2_data;
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
    assign barrier.size_m1 = (`NW_BITS)'(rs2_data - 1);       

    // Warp control response
    wire wctl_rsp_valid = gpu_req_if.valid & is_wctl;
    wire [WCTL_DATAW-1:0] wctl_rsp_data = {tmc, wspawn, split, barrier};
    wire wctl_rsp_ready;

    `UNUSED_VAR (gpu_req_if.op_mod)
    `UNUSED_VAR (gpu_req_if.rs3_data)
    `UNUSED_VAR (gpu_req_if.wb)
    `UNUSED_VAR (gpu_req_if.rd)
    
`ifdef EXT_TEX_ENABLE
    
    VX_tex_req_if tex_req_if();
    VX_commit_if  tex_rsp_if();

    assign tex_req_if.valid = gpu_req_if.valid && (gpu_req_if.op_type == `INST_GPU_TEX);
    assign tex_req_if.uuid  = gpu_req_if.uuid;
    assign tex_req_if.wid   = gpu_req_if.wid;
    assign tex_req_if.tmask = gpu_req_if.tmask;
    assign tex_req_if.PC    = gpu_req_if.PC;
    assign tex_req_if.rd    = gpu_req_if.rd;
    assign tex_req_if.wb    = gpu_req_if.wb;
    
    assign tex_req_if.stage     = gpu_req_if.op_mod[`NTEX_BITS-1:0];
    assign tex_req_if.coords[0] = gpu_req_if.rs1_data;
    assign tex_req_if.coords[1] = gpu_req_if.rs2_data;
    assign tex_req_if.lod       = gpu_req_if.rs3_data;        

    VX_tex_unit #(
        .CORE_ID (CORE_ID)
    ) tex_unit (
        .clk           (clk),
        .reset         (reset),
    `ifdef PERF_ENABLE
        .perf_tex_if   (perf_tex_if),
    `endif
        .tex_req_if    (tex_req_if),
        .tex_rsp_if    (tex_rsp_if),
        .tex_dcr_if    (tex_dcr_if),
        .cache_req_if  (tcache_req_if),
        .cache_rsp_if  (tcache_rsp_if)
    );        
`endif

    // can accept new request?
    reg gpu_req_ready;
    always @(*) begin
        case (gpu_req_if.op_type)
    `ifdef EXT_TEX_ENABLE
        `INST_GPU_TEX: gpu_req_ready = tex_req_if.ready;
    `endif
        default: gpu_req_ready = wctl_rsp_ready;
        endcase
    end   
    assign gpu_req_if.ready = gpu_req_ready;

    VX_stream_mux #(            
        .NUM_REQS (1
    `ifdef EXT_TEX_ENABLE
        +1
    `endif    
        ),
        .DATAW    (MUX_DATAW),
        .BUFFERED (1),
        .TYPE     ("R")
    ) rsp_mux (
        .clk       (clk),
        .reset     (reset),
        .valid_in  ({
            wctl_rsp_valid
        `ifdef EXT_TEX_ENABLE
          , tex_rsp_if.valid
        `endif
        }),
        .data_in   ({
            {gpu_req_if.uuid, gpu_req_if.wid, gpu_req_if.tmask, gpu_req_if.PC, `NR_BITS'(0),  1'b0,          RSP_DATAW'(wctl_rsp_data),   1'b1,           1'b1}
        `ifdef EXT_TEX_ENABLE
          , {tex_rsp_if.uuid, tex_rsp_if.wid, tex_rsp_if.tmask, tex_rsp_if.PC, tex_rsp_if.rd, tex_rsp_if.wb, RSP_DATAW'(tex_rsp_if.data), tex_rsp_if.eop, 1'b0}
        `endif
        }),
        .ready_in  ({
            wctl_rsp_ready
        `ifdef EXT_TEX_ENABLE
          , tex_rsp_if.ready
        `endif
        }),
        .valid_out (rsp_valid),
        .data_out  ({rsp_uuid, rsp_wid, rsp_tmask, rsp_PC, rsp_rd, rsp_wb, rsp_data, rsp_eop, rsp_is_wctl}),
        .ready_out (~stall_out)
    );

    // output
    assign stall_out = ~gpu_commit_if.ready && gpu_commit_if.valid;

    VX_pipe_register #(
        .DATAW  (1 + MUX_DATAW),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall_out),
        .data_in  ({rsp_valid,           rsp_uuid,           rsp_wid,           rsp_tmask,           rsp_PC,           rsp_rd,           rsp_wb,           rsp_data,   rsp_eop,           rsp_is_wctl}),
        .data_out ({gpu_commit_if.valid, gpu_commit_if.uuid, gpu_commit_if.wid, gpu_commit_if.tmask, gpu_commit_if.PC, gpu_commit_if.rd, gpu_commit_if.wb, rsp_data_r, gpu_commit_if.eop, rsp_is_wctl_r})
    );  

    assign gpu_commit_if.data = rsp_data_r[(`NUM_THREADS * 32)-1:0];

    // warp control reponse
     
    assign {warp_ctl_if.tmc, warp_ctl_if.wspawn, warp_ctl_if.split, warp_ctl_if.barrier} = rsp_data_r[WCTL_DATAW-1:0];    
    assign warp_ctl_if.valid = gpu_commit_if.valid && gpu_commit_if.ready && rsp_is_wctl_r;
    assign warp_ctl_if.wid   = gpu_commit_if.wid;    

    `SCOPE_ASSIGN (gpu_rsp_valid, warp_ctl_if.valid);
    `SCOPE_ASSIGN (gpu_rsp_uuid, gpu_commit_if.uuid);
    `SCOPE_ASSIGN (gpu_rsp_tmc, warp_ctl_if.tmc.valid);
    `SCOPE_ASSIGN (gpu_rsp_wspawn, warp_ctl_if.wspawn.valid);          
    `SCOPE_ASSIGN (gpu_rsp_split, warp_ctl_if.split.valid);
    `SCOPE_ASSIGN (gpu_rsp_barrier, warp_ctl_if.barrier.valid);

endmodule