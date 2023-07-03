`include "VX_define.vh"

module VX_gpu_unit #(
    parameter CORE_ID = 0
) (    
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    VX_gpu_perf_if.master   gpu_perf_if,
`endif

    // Inputs
    VX_gpu_exe_if.slave     gpu_exe_if,

`ifdef EXT_TEX_ENABLE
    VX_gpu_csr_if.slave     tex_csr_if,
    VX_tex_bus_if.master    tex_bus_if,
`endif

`ifdef EXT_RASTER_ENABLE        
    VX_gpu_csr_if.slave     raster_csr_if,
    VX_raster_bus_if.slave  raster_bus_if,
`endif

`ifdef EXT_ROP_ENABLE        
    VX_gpu_csr_if.slave     rop_csr_if,
    VX_rop_bus_if.master    rop_bus_if,
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
    localparam RSP_ARB_DATAW = UUID_WIDTH + NW_WIDTH + `NUM_THREADS + (`NUM_THREADS * `XLEN) + `NR_BITS + 1 + `XLEN + 1;
    localparam RSP_ARB_SIZE  = 1 + `EXT_TEX_ENABLED + `EXT_RASTER_ENABLED + `EXT_ROP_ENABLED + `EXT_IMADD_ENABLED;

    localparam RSP_ARB_IDX_WCTL   = 0;
    localparam RSP_ARB_IDX_RASTER = RSP_ARB_IDX_WCTL + 1;
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

    wire gpu_req_valid;
    reg gpu_req_ready;

    wire csr_ready = ~csr_pending;
    assign gpu_req_valid = gpu_exe_if.valid && csr_ready;

    // Warp control block    
    VX_gpu_exe_if wctl_exe_if();
    VX_commit_if wctl_commit_if();
    
    `ASSIGN_VX_GPU_EXE_IF_V(wctl_exe_if, gpu_exe_if, gpu_req_valid && `INST_GPU_IS_WCTL(gpu_exe_if.op_type));
    
    VX_wctl_unit #(
        .OUTPUT_REG (RSP_ARB_SIZE > 1)
    ) wctl_unit (
        .clk        (clk),
        .reset      (reset),
        .gpu_exe_if (wctl_exe_if),  
        .warp_ctl_if(warp_ctl_if),      
        .commit_if  (wctl_commit_if)
    );

    assign rsp_arb_valid_in[RSP_ARB_IDX_WCTL] = wctl_commit_if.valid;
    assign rsp_arb_data_in[RSP_ARB_IDX_WCTL] = {wctl_commit_if.uuid, wctl_commit_if.wid, wctl_commit_if.tmask, wctl_commit_if.PC, wctl_commit_if.rd, wctl_commit_if.wb, wctl_commit_if.data, 1'b1};
    assign wctl_commit_if.ready = rsp_arb_ready_in[RSP_ARB_IDX_WCTL];
    
`ifdef EXT_TEX_ENABLE

    VX_gpu_exe_if tex_exe_if();
    VX_commit_if tex_commit_if();

    `ASSIGN_VX_GPU_EXE_IF_V(tex_exe_if, gpu_exe_if, gpu_req_valid && (gpu_exe_if.op_type == `INST_GPU_TEX));

    `RESET_RELAY (tex_reset, reset);

    VX_tex_agent #(
        .CORE_ID (CORE_ID)
    ) tex_agent (
        .clk        (clk),
        .reset      (tex_reset),
        .gpu_exe_if (tex_exe_if),
        .tex_csr_if (tex_csr_if),
        .tex_bus_if (tex_bus_if),
        .commit_if  (tex_commit_if)        
    );     

    assign rsp_arb_valid_in[RSP_ARB_IDX_TEX] = tex_commit_if.valid;
    assign rsp_arb_data_in[RSP_ARB_IDX_TEX] = {tex_commit_if.uuid, tex_commit_if.wid, tex_commit_if.tmask, tex_commit_if.PC, tex_commit_if.rd, tex_commit_if.wb, tex_commit_if.data, tex_commit_if.eop};
    assign tex_commit_if.ready = rsp_arb_ready_in[RSP_ARB_IDX_TEX];

`endif

`ifdef EXT_RASTER_ENABLE
    
    VX_gpu_exe_if raster_exe_if();
    VX_commit_if raster_commit_if();

    `ASSIGN_VX_GPU_EXE_IF_V(raster_exe_if, gpu_exe_if, gpu_req_valid && (gpu_exe_if.op_type == `INST_GPU_RASTER));

    `RESET_RELAY (raster_reset, reset);

    VX_raster_agent #(
        .CORE_ID (CORE_ID)
    ) raster_agent (
        .clk            (clk),
        .reset          (raster_reset),
        .gpu_exe_if     (raster_exe_if),
        .raster_csr_if  (raster_csr_if),
        .raster_bus_if  (raster_bus_if),        
        .commit_if      (raster_commit_if)       
    );

    assign rsp_arb_valid_in[RSP_ARB_IDX_RASTER] = raster_commit_if.valid;
    assign rsp_arb_data_in[RSP_ARB_IDX_RASTER] = {raster_commit_if.uuid, raster_commit_if.wid, raster_commit_if.tmask, raster_commit_if.PC, raster_commit_if.rd, raster_commit_if.wb, raster_commit_if.data, raster_commit_if.eop};
    assign raster_commit_if.ready = rsp_arb_ready_in[RSP_ARB_IDX_RASTER];

`endif

`ifdef EXT_ROP_ENABLE
    
    VX_gpu_exe_if rop_exe_if();
    VX_commit_if rop_commit_if();

    `ASSIGN_VX_GPU_EXE_IF_V(rop_exe_if, gpu_exe_if, gpu_req_valid && (gpu_exe_if.op_type == `INST_GPU_ROP));

    `RESET_RELAY (rop_reset, reset);
            
    VX_rop_agent #(
        .CORE_ID (CORE_ID)
    ) rop_agent (
        .clk        (clk),
        .reset      (rop_reset),
        .gpu_exe_if (rop_exe_if),
        .rop_csr_if (rop_csr_if),
        .rop_bus_if (rop_bus_if),
        .commit_if  (rop_commit_if)
    );

    assign rsp_arb_valid_in[RSP_ARB_IDX_ROP] = rop_commit_if.valid;
    assign rsp_arb_data_in[RSP_ARB_IDX_ROP] = {rop_commit_if.uuid, rop_commit_if.wid, rop_commit_if.tmask, rop_commit_if.PC, rop_commit_if.rd, rop_commit_if.wb, rop_commit_if.data, rop_commit_if.eop};
    assign rop_commit_if.ready = rsp_arb_ready_in[RSP_ARB_IDX_ROP];

`endif

`ifdef EXT_IMADD_ENABLE    

    wire                          imadd_valid_in;
    wire                          imadd_ready_in;
    wire [`NUM_THREADS-1:0][31:0] imadd_data_in [3];

    wire                          imadd_valid_out;
    wire [UUID_WIDTH-1:0]         imadd_uuid_out;
    wire [NW_WIDTH-1:0]           imadd_wid_out;
    wire [`NUM_THREADS-1:0]       imadd_tmask_out;
    wire [`XLEN-1:0]              imadd_PC_out;
    wire [`NR_BITS-1:0]           imadd_rd_out; 
    wire [`NUM_THREADS-1:0][31:0] imadd_data_out;
    wire                          imadd_ready_out;

    assign imadd_valid_in = gpu_req_valid && (gpu_exe_if.op_type == `INST_GPU_IMADD);

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign imadd_data_in[0][i] = gpu_exe_if.rs1_data[i][31:0];
        assign imadd_data_in[1][i] = gpu_exe_if.rs2_data[i][31:0];
        assign imadd_data_in[2][i] = gpu_exe_if.rs3_data[i][31:0];
    end

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
        .shift_in   ({gpu_exe_if.op_mod[1:0], 3'b0}),
        .data1_in   (imadd_data_in[0]),
        .data2_in   (imadd_data_in[1]),
        .data3_in   (imadd_data_in[2]),
        .tag_in     ({gpu_exe_if.uuid, gpu_exe_if.wid, gpu_exe_if.tmask, gpu_exe_if.PC, gpu_exe_if.rd}),
        .ready_in   (imadd_ready_in),

        // Outputs
        .valid_out  (imadd_valid_out),
        .tag_out    ({imadd_uuid_out, imadd_wid_out, imadd_tmask_out, imadd_PC_out, imadd_rd_out}),
        .data_out   (imadd_data_out),
        .ready_out  (imadd_ready_out)
    );

    wire [`NUM_THREADS-1:0][`XLEN-1:0] imadd_data_out_x;
    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign imadd_data_out_x[i] = `XLEN'(imadd_data_out[i]);
    end

    assign rsp_arb_valid_in[RSP_ARB_IDX_IMADD] = imadd_valid_out;
    assign rsp_arb_data_in[RSP_ARB_IDX_IMADD] = {imadd_uuid_out, imadd_wid_out, imadd_tmask_out, imadd_PC_out, imadd_rd_out, 1'b1, imadd_data_out_x, 1'b1};
    assign imadd_ready_out = rsp_arb_ready_in[RSP_ARB_IDX_IMADD];

`endif

    // can accept new request?

    always @(*) begin
        case (gpu_exe_if.op_type)
    `ifdef EXT_TEX_ENABLE
        `INST_GPU_TEX: gpu_req_ready = tex_exe_if.ready;
    `endif
    `ifdef EXT_RASTER_ENABLE
        `INST_GPU_RASTER: gpu_req_ready = raster_exe_if.ready;
    `endif
    `ifdef EXT_ROP_ENABLE
        `INST_GPU_ROP: gpu_req_ready = rop_exe_if.ready;
    `endif
    `ifdef EXT_IMADD_ENABLE
        `INST_GPU_IMADD: gpu_req_ready = imadd_ready_in;
    `endif
        default: gpu_req_ready = wctl_exe_if.ready;
        endcase
    end   
    assign gpu_exe_if.ready = gpu_req_ready && csr_ready;

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
        .data_out  ({gpu_commit_if.uuid, gpu_commit_if.wid, gpu_commit_if.tmask, gpu_commit_if.PC, gpu_commit_if.rd, gpu_commit_if.wb, gpu_commit_if.data, gpu_commit_if.eop}),
        .valid_out (gpu_commit_if.valid),
        .ready_out (gpu_commit_if.ready)
    );

    wire gpu_req_fire = gpu_exe_if.valid && gpu_exe_if.ready;
    wire gpu_commit_fire = gpu_commit_if.valid && gpu_commit_if.ready;

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
            perf_tex_stalls <= perf_tex_stalls + `PERF_CTR_BITS'(tex_exe_if.valid && ~tex_exe_if.ready);
        end
    end
    assign gpu_perf_if.tex_stalls = perf_tex_stalls;
`endif
`ifdef EXT_RASTER_ENABLE
    reg [`PERF_CTR_BITS-1:0] perf_raster_stalls;
    always @(posedge clk) begin
        if (reset) begin
            perf_raster_stalls <= '0;
        end else begin
            perf_raster_stalls <= perf_raster_stalls + `PERF_CTR_BITS'(raster_exe_if.valid && ~raster_exe_if.ready);
        end
    end
    assign gpu_perf_if.raster_stalls = perf_raster_stalls;
`endif
`ifdef EXT_ROP_ENABLE
    reg [`PERF_CTR_BITS-1:0] perf_rop_stalls;
    always @(posedge clk) begin
        if (reset) begin
            perf_rop_stalls <= '0;
        end else begin
            perf_rop_stalls <= perf_rop_stalls + `PERF_CTR_BITS'(rop_exe_if.valid && ~rop_exe_if.ready);
        end
    end
    assign gpu_perf_if.rop_stalls = perf_rop_stalls;
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
    assign gpu_perf_if.imadd_stalls = perf_imadd_stalls;
`endif
    reg [`PERF_CTR_BITS-1:0] perf_wctl_stalls;
    always @(posedge clk) begin
        if (reset) begin
            perf_wctl_stalls <= '0;
        end else begin
            perf_wctl_stalls <= perf_wctl_stalls + `PERF_CTR_BITS'(wctl_req_valid && ~wctl_req_ready);
        end
    end
    assign gpu_perf_if.wctl_stalls = perf_wctl_stalls;
`endif

endmodule
