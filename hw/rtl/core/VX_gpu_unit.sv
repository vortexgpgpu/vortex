`include "VX_define.vh"
`include "VX_gpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
`IGNORE_WARNINGS_END

module VX_gpu_unit #(
    parameter CORE_ID = 0
) (    
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    VX_perf_gpu_if.master   perf_gpu_if,
`endif

    // Inputs
    VX_gpu_req_if.slave     gpu_req_if,

`ifdef EXT_TEX_ENABLE
    VX_gpu_csr_if.slave     tex_csr_if,
    VX_tex_req_if.master    tex_req_if,
    VX_tex_rsp_if.slave     tex_rsp_if,
`endif

`ifdef EXT_RASTER_ENABLE        
    VX_gpu_csr_if.slave     raster_csr_if,
    VX_raster_req_if.slave  raster_req_if,
`endif

`ifdef EXT_ROP_ENABLE        
    VX_gpu_csr_if.slave     rop_csr_if,
    VX_rop_req_if.master    rop_req_if,
`endif

    // Outputs
    VX_warp_ctl_if.master   warp_ctl_if,
    VX_commit_if.master     gpu_commit_if,

    input wire              csr_pending,
    output wire             req_pending
);
    `UNUSED_PARAM (CORE_ID)

    localparam UUID_WIDTH    = `UP(`UUID_BITS);
    localparam NW_WIDTH      = `UP(`NW_BITS);
    localparam WCTL_DATAW    = `GPU_TMC_BITS + `GPU_WSPAWN_BITS + `GPU_SPLIT_BITS + `GPU_BARRIER_BITS;
    localparam RSP_DATAW     = `MAX(`NUM_THREADS * `XLEN, WCTL_DATAW);
    localparam RSP_ARB_DATAW = UUID_WIDTH + NW_WIDTH + `NUM_THREADS + `XLEN + `NR_BITS + 1 + RSP_DATAW + 1 + 1;
    localparam RSP_ARB_SIZE  = 1 + `EXT_TEX_ENABLED + `EXT_RASTER_ENABLED + `EXT_ROP_ENABLED + `EXT_IMADD_ENABLED;

    localparam RSP_ARB_IDX_GPU    = 0;
    localparam RSP_ARB_IDX_RASTER = RSP_ARB_IDX_GPU + 1;
    localparam RSP_ARB_IDX_ROP    = RSP_ARB_IDX_RASTER + `EXT_RASTER_ENABLED;    
    localparam RSP_ARB_IDX_TEX    = RSP_ARB_IDX_ROP + `EXT_ROP_ENABLED;    
    localparam RSP_ARB_IDX_IMADD  = RSP_ARB_IDX_TEX + `EXT_TEX_ENABLED;
    `UNUSED_PARAM (RSP_ARB_IDX_RASTER)
    `UNUSED_PARAM (RSP_ARB_IDX_ROP)
    `UNUSED_PARAM (RSP_ARB_IDX_TEX)
    `UNUSED_PARAM (RSP_ARB_IDX_IMADD)

    wire [RSP_ARB_SIZE-1:0] rsp_arb_valid_in;
    wire [RSP_ARB_SIZE-1:0] rsp_arb_ready_in;
    wire [RSP_ARB_SIZE-1:0][RSP_ARB_DATAW-1:0] rsp_arb_data_in;

    wire [RSP_DATAW-1:0] rsp_data;
    wire                 rsp_is_wctl;

    wire gpu_req_valid;
    reg gpu_req_ready;

    wire csr_ready = ~csr_pending;
    assign gpu_req_valid = gpu_req_if.valid && csr_ready;

    // Warp control block

    gpu_tmc_t       tmc;
    gpu_wspawn_t    wspawn;
    gpu_barrier_t   barrier;
    gpu_split_t     split;
    
    wire is_wspawn = (gpu_req_if.op_type == `INST_GPU_WSPAWN);
    wire is_tmc    = (gpu_req_if.op_type == `INST_GPU_TMC);
    wire is_split  = (gpu_req_if.op_type == `INST_GPU_SPLIT);
    wire is_join   = (gpu_req_if.op_type == `INST_GPU_JOIN);
    wire is_bar    = (gpu_req_if.op_type == `INST_GPU_BAR);
    wire is_pred   = (gpu_req_if.op_type == `INST_GPU_PRED);

    wire [`XLEN-1:0] rs1_data = gpu_req_if.rs1_data[gpu_req_if.tid];
    wire [`XLEN-1:0] rs2_data = gpu_req_if.rs2_data[gpu_req_if.tid];
    
    wire [`NUM_THREADS-1:0] taken_tmask;
    wire [`NUM_THREADS-1:0] not_taken_tmask;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        wire taken = (gpu_req_if.rs1_data[i] != 0);
        assign taken_tmask[i]     = gpu_req_if.tmask[i] && taken;
        assign not_taken_tmask[i] = gpu_req_if.tmask[i] && ~taken;
    end

    // tmc

    wire [`NUM_THREADS-1:0] pred_mask = (taken_tmask != 0) ? taken_tmask : gpu_req_if.tmask;

    assign tmc.valid = is_tmc || is_pred;
    assign tmc.tmask = is_pred ? pred_mask : rs1_data[`NUM_THREADS-1:0];

    // wspawn

    wire [`XLEN-1:0] wspawn_pc = rs2_data;
    wire [`NUM_WARPS-1:0] wspawn_wmask;
    for (genvar i = 0; i < `NUM_WARPS; ++i) begin
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
    
    assign barrier.valid    = is_bar;
    assign barrier.id       = rs1_data[`NB_BITS-1:0];
    assign barrier.is_global = rs1_data[31];
    assign barrier.size_m1  = $bits(barrier.size_m1)'(rs2_data - 1);       

    // Warp control response
    wire wctl_req_valid = gpu_req_valid && (is_wspawn || is_tmc || is_split || is_join || is_bar || is_pred);
    wire wctl_rsp_valid = wctl_req_valid;
    wire [WCTL_DATAW-1:0] wctl_rsp_data = {tmc, wspawn, split, barrier};
    wire wctl_rsp_ready;
    wire wctl_req_ready = wctl_rsp_ready;

    assign rsp_arb_valid_in[RSP_ARB_IDX_GPU] = wctl_rsp_valid;
    assign rsp_arb_data_in[RSP_ARB_IDX_GPU] = {gpu_req_if.uuid, gpu_req_if.wid, gpu_req_if.tmask, gpu_req_if.PC, `NR_BITS'(0), 1'b0, RSP_DATAW'(wctl_rsp_data), 1'b1, ~is_join};
    assign wctl_rsp_ready = rsp_arb_ready_in[RSP_ARB_IDX_GPU];

    `UNUSED_VAR (gpu_req_if.op_mod)
    `UNUSED_VAR (gpu_req_if.rs3_data)
    `UNUSED_VAR (gpu_req_if.wb)
    `UNUSED_VAR (gpu_req_if.rd)
    
`ifdef EXT_TEX_ENABLE

    VX_tex_agent_if tex_agent_if();
    VX_commit_if    tex_commit_if();

    assign tex_agent_if.valid = gpu_req_valid && (gpu_req_if.op_type == `INST_GPU_TEX);
    assign tex_agent_if.uuid  = gpu_req_if.uuid;
    assign tex_agent_if.wid   = gpu_req_if.wid;
    assign tex_agent_if.tmask = gpu_req_if.tmask;
    assign tex_agent_if.PC    = gpu_req_if.PC;
    assign tex_agent_if.rd    = gpu_req_if.rd;
    assign tex_agent_if.stage = gpu_req_if.op_mod[`TEX_STAGE_BITS-1:0];

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign tex_agent_if.coords[0][i] = gpu_req_if.rs1_data[i];
        assign tex_agent_if.coords[1][i] = gpu_req_if.rs2_data[i];
        assign tex_agent_if.lod[i]       = gpu_req_if.rs3_data[i][0 +: `TEX_LOD_BITS];        
    end

    `RESET_RELAY (tex_reset, reset);

    VX_tex_agent #(
        .CORE_ID (CORE_ID)
    ) tex_agent (
        .clk           (clk),
        .reset         (tex_reset),
        .tex_csr_if    (tex_csr_if),
        .tex_agent_if  (tex_agent_if),        
        .tex_commit_if (tex_commit_if),
        .tex_req_if    (tex_req_if),
        .tex_rsp_if    (tex_rsp_if)
    );     

    assign rsp_arb_valid_in[RSP_ARB_IDX_TEX] = tex_commit_if.valid;
    assign rsp_arb_data_in[RSP_ARB_IDX_TEX] = {tex_commit_if.uuid, tex_commit_if.wid, tex_commit_if.tmask, tex_commit_if.PC, tex_commit_if.rd, tex_commit_if.wb, RSP_DATAW'(tex_commit_if.data), tex_commit_if.eop, 1'b0};
    assign tex_commit_if.ready = rsp_arb_ready_in[RSP_ARB_IDX_TEX];

`endif

`ifdef EXT_RASTER_ENABLE
    
    VX_raster_agent_if raster_agent_if();
    VX_commit_if       raster_commit_if();

    assign raster_agent_if.valid = gpu_req_valid && (gpu_req_if.op_type == `INST_GPU_RASTER);
    assign raster_agent_if.uuid  = gpu_req_if.uuid;
    assign raster_agent_if.wid   = gpu_req_if.wid;
    assign raster_agent_if.tmask = gpu_req_if.tmask;
    assign raster_agent_if.PC    = gpu_req_if.PC;
    assign raster_agent_if.rd    = gpu_req_if.rd;

    `RESET_RELAY (raster_reset, reset);

    VX_raster_agent #(
        .CORE_ID (CORE_ID)
    ) raster_agent (
        .clk              (clk),
        .reset            (raster_reset),
        .raster_csr_if    (raster_csr_if),
        .raster_req_if    (raster_req_if),
        .raster_agent_if  (raster_agent_if),        
        .raster_commit_if (raster_commit_if)        
    );

    assign rsp_arb_valid_in[RSP_ARB_IDX_RASTER] = raster_commit_if.valid;
    assign rsp_arb_data_in[RSP_ARB_IDX_RASTER] = {raster_commit_if.uuid, raster_commit_if.wid, raster_commit_if.tmask, raster_commit_if.PC, raster_commit_if.rd, raster_commit_if.wb, RSP_DATAW'(raster_commit_if.data), raster_commit_if.eop, 1'b0};
    assign raster_commit_if.ready = rsp_arb_ready_in[RSP_ARB_IDX_RASTER];

`endif

`ifdef EXT_ROP_ENABLE
    
    VX_rop_agent_if rop_agent_if();
    VX_commit_if    rop_commit_if();

    assign rop_agent_if.valid = gpu_req_valid && (gpu_req_if.op_type == `INST_GPU_ROP);
    assign rop_agent_if.uuid  = gpu_req_if.uuid;
    assign rop_agent_if.wid   = gpu_req_if.wid;
    assign rop_agent_if.tmask = gpu_req_if.tmask;
    assign rop_agent_if.PC    = gpu_req_if.PC;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign rop_agent_if.face[i]  = gpu_req_if.rs1_data[i][0];
        assign rop_agent_if.pos_x[i] = gpu_req_if.rs1_data[i][1 +: `ROP_DIM_BITS];
        assign rop_agent_if.pos_y[i] = gpu_req_if.rs1_data[i][16 +: `ROP_DIM_BITS];
        assign rop_agent_if.color[i] = gpu_req_if.rs2_data[i];
        assign rop_agent_if.depth[i] = gpu_req_if.rs3_data[i][`ROP_DEPTH_BITS-1:0];
    end

    `RESET_RELAY (rop_reset, reset);
            
    VX_rop_agent #(
        .CORE_ID (CORE_ID)
    ) rop_agent (
        .clk           (clk),
        .reset         (rop_reset),
        .rop_csr_if    (rop_csr_if),
        .rop_agent_if  (rop_agent_if),
        .rop_commit_if (rop_commit_if),
        .rop_req_if    (rop_req_if)        
    );

    assign rsp_arb_valid_in[RSP_ARB_IDX_ROP] = rop_commit_if.valid;
    assign rsp_arb_data_in[RSP_ARB_IDX_ROP] = {rop_commit_if.uuid, rop_commit_if.wid, rop_commit_if.tmask, rop_commit_if.PC, rop_commit_if.rd, rop_commit_if.wb, RSP_DATAW'(rop_commit_if.data), rop_commit_if.eop, 1'b0};
    assign rop_commit_if.ready = rsp_arb_ready_in[RSP_ARB_IDX_ROP];

`endif

`ifdef EXT_IMADD_ENABLE    

    wire                          imadd_valid_in;
    wire                          imadd_ready_in;

    wire                          imadd_valid_out;
    wire [UUID_WIDTH-1:0]         imadd_uuid_out;
    wire [NW_WIDTH-1:0]           imadd_wid_out;
    wire [`NUM_THREADS-1:0]       imadd_tmask_out;
    wire [`XLEN-1:0]              imadd_PC_out;
    wire [`NR_BITS-1:0]           imadd_rd_out; 
    wire [`NUM_THREADS-1:0][31:0] imadd_data_out;
    wire                          imadd_ready_out;

    assign imadd_valid_in = gpu_req_valid && (gpu_req_if.op_type == `INST_GPU_IMADD);

    `RESET_RELAY (imadd_reset, reset);

    VX_imadd #(
        .NUM_LANES  (`NUM_THREADS),
        .DATA_WIDTH (32),
        .MAX_SHIFT  (24),
        .SIGNED     (1),
        .TAG_WIDTH  (UUID_WIDTH + NW_WIDTH + `NUM_THREADS + `XLEN + `NR_BITS)
    ) imadd (
        .clk        (clk),
        .reset      (imadd_reset),
        
        // Inputs
        .valid_in   (imadd_valid_in),
        .shift_in   ({gpu_req_if.op_mod[1:0], 3'b0}),
        .data1_in   (gpu_req_if.rs1_data),
        .data2_in   (gpu_req_if.rs2_data),
        .data3_in   (gpu_req_if.rs3_data),
        .tag_in     ({gpu_req_if.uuid, gpu_req_if.wid, gpu_req_if.tmask, gpu_req_if.PC, gpu_req_if.rd}),
        .ready_in   (imadd_ready_in),

        // Outputs
        .valid_out  (imadd_valid_out),
        .tag_out    ({imadd_uuid_out, imadd_wid_out, imadd_tmask_out, imadd_PC_out, imadd_rd_out}),
        .data_out   (imadd_data_out),
        .ready_out  (imadd_ready_out)
    );

    assign rsp_arb_valid_in[RSP_ARB_IDX_IMADD] = imadd_valid_out;
    assign rsp_arb_data_in[RSP_ARB_IDX_IMADD] = {imadd_uuid_out, imadd_wid_out, imadd_tmask_out, imadd_PC_out, imadd_rd_out, 1'b1, RSP_DATAW'(imadd_data_out), 1'b1, 1'b0};
    assign imadd_ready_out = rsp_arb_ready_in[RSP_ARB_IDX_IMADD];

`endif

    // can accept new request?
    
    always @(*) begin
        case (gpu_req_if.op_type)
    `ifdef EXT_TEX_ENABLE
        `INST_GPU_TEX: gpu_req_ready = tex_agent_if.ready;
    `endif
    `ifdef EXT_RASTER_ENABLE
        `INST_GPU_RASTER: gpu_req_ready = raster_agent_if.ready;
    `endif
    `ifdef EXT_ROP_ENABLE
        `INST_GPU_ROP: gpu_req_ready = rop_agent_if.ready;
    `endif
    `ifdef EXT_IMADD_ENABLE
        `INST_GPU_IMADD: gpu_req_ready = imadd_ready_in;
    `endif
        default: gpu_req_ready = wctl_req_ready;
        endcase
    end   
    assign gpu_req_if.ready = gpu_req_ready && csr_ready;

    // response arbitration

    VX_stream_arb #(
        .NUM_INPUTS (RSP_ARB_SIZE),
        .DATAW      (RSP_ARB_DATAW),
        .ARBITER    ("R"),
        .BUFFERED   (1)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),        
        .valid_in  (rsp_arb_valid_in),
        .ready_in  (rsp_arb_ready_in),
        .data_in   (rsp_arb_data_in),
        .data_out  ({gpu_commit_if.uuid, gpu_commit_if.wid, gpu_commit_if.tmask, gpu_commit_if.PC, gpu_commit_if.rd, gpu_commit_if.wb, rsp_data, gpu_commit_if.eop, rsp_is_wctl}),
        .valid_out (gpu_commit_if.valid),
        .ready_out (gpu_commit_if.ready)
    );

    assign gpu_commit_if.data = rsp_data[(`NUM_THREADS * `XLEN)-1:0];

    // warp control reponse

    wire gpu_req_fire = gpu_req_if.valid && gpu_req_if.ready;
    wire gpu_commit_fire = gpu_commit_if.valid && gpu_commit_if.ready;
         
    assign warp_ctl_if.valid = gpu_commit_fire && rsp_is_wctl;
    assign warp_ctl_if.wid   = gpu_commit_if.wid;    
    assign {warp_ctl_if.tmc, warp_ctl_if.wspawn, warp_ctl_if.split, warp_ctl_if.barrier} = rsp_data[WCTL_DATAW-1:0];

    // pending request

    reg req_pending_r;
    always @(posedge clk) begin
        if (reset) begin
            req_pending_r <= 0;
        end else begin                      
            if (gpu_req_fire) begin
                 req_pending_r <= 1;
            end
            if (gpu_commit_fire) begin
                 req_pending_r <= 0;
            end
        end
    end
    assign req_pending = req_pending_r;

`ifdef PERF_ENABLE
`ifdef EXT_TEX_ENABLE
    reg [`PERF_CTR_BITS-1:0] perf_tex_stalls;
    always @(posedge clk) begin
        if (reset) begin
            perf_tex_stalls <= '0;
        end else begin
            perf_tex_stalls <= perf_tex_stalls + `PERF_CTR_BITS'(tex_agent_if.valid && ~tex_agent_if.ready);
        end
    end
    assign perf_gpu_if.tex_stalls = perf_tex_stalls;
`endif
`ifdef EXT_RASTER_ENABLE
    reg [`PERF_CTR_BITS-1:0] perf_raster_stalls;
    always @(posedge clk) begin
        if (reset) begin
            perf_raster_stalls <= '0;
        end else begin
            perf_raster_stalls <= perf_raster_stalls + `PERF_CTR_BITS'(raster_agent_if.valid && ~raster_agent_if.ready);
        end
    end
    assign perf_gpu_if.raster_stalls = perf_raster_stalls;
`endif
`ifdef EXT_ROP_ENABLE
    reg [`PERF_CTR_BITS-1:0] perf_rop_stalls;
    always @(posedge clk) begin
        if (reset) begin
            perf_rop_stalls <= '0;
        end else begin
            perf_rop_stalls <= perf_rop_stalls + `PERF_CTR_BITS'(rop_agent_if.valid && ~rop_agent_if.ready);
        end
    end
    assign perf_gpu_if.rop_stalls = perf_rop_stalls;
`endif
`ifdef EXT_IMADD_ENABLE
    reg [`PERF_CTR_BITS-1:0] perf_imadd_stalls;
    always @(posedge clk) begin
        if (reset) begin
            perf_imadd_stalls <= '0;
        end else begin
            perf_imadd_stalls <= perf_imadd_stalls + `PERF_CTR_BITS'(imadd_valid_in && ~imadd_ready_in);
        end
    end
    assign perf_gpu_if.imadd_stalls = perf_imadd_stalls;
`endif
    reg [`PERF_CTR_BITS-1:0] perf_wctl_stalls;
    always @(posedge clk) begin
        if (reset) begin
            perf_wctl_stalls <= '0;
        end else begin
            perf_wctl_stalls <= perf_wctl_stalls + `PERF_CTR_BITS'(wctl_req_valid && ~wctl_req_ready);
        end
    end
    assign perf_gpu_if.wctl_stalls = perf_wctl_stalls;
`endif

endmodule
