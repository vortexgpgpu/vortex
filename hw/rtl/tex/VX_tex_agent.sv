`include "VX_tex_define.vh"

module VX_tex_agent #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_gpu_exe_if.slave     gpu_exe_if,
    VX_gpu_csr_if.slave     tex_csr_if,
            
    // Outputs
    VX_tex_bus_if.master    tex_bus_if,
    VX_commit_if.master     commit_if
);
    `UNUSED_PARAM (CORE_ID)

    localparam UUID_WIDTH = `UP(`UUID_BITS);
    localparam NW_WIDTH   = `UP(`NW_BITS);
    localparam REQ_QUEUE_BITS = `LOG2UP(`TEX_REQ_QUEUE_SIZE);

    // CSRs access

    tex_csrs_t tex_csrs;

    VX_tex_csr #(
        .CORE_ID    (CORE_ID)
    ) tex_csr (
        .clk        (clk),
        .reset      (reset),

        // inputs
        .tex_csr_if (tex_csr_if),

        // outputs
        .tex_csrs   (tex_csrs)
    );

    `UNUSED_VAR (tex_csrs)

    // Store request info

    wire [1:0][`NUM_THREADS-1:0][31:0] gpu_exe_coords;
    wire [`NUM_THREADS-1:0][`VX_TEX_LOD_BITS-1:0] gpu_exe_lod;
    wire [`VX_TEX_STAGE_BITS-1:0] gpu_exe_stage;

    wire [UUID_WIDTH-1:0]   rsp_uuid;
    wire [NW_WIDTH-1:0]     rsp_wid;
    wire [`NUM_THREADS-1:0] rsp_tmask;
    wire [`XLEN-1:0]        rsp_PC;
    wire [`NR_BITS-1:0]     rsp_rd;
 
    wire [REQ_QUEUE_BITS-1:0] mdata_waddr, mdata_raddr;    
    wire mdata_full;

    assign gpu_exe_stage = gpu_exe_if.op_mod[`VX_TEX_STAGE_BITS-1:0];
    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign gpu_exe_coords[0][i] = gpu_exe_if.rs1_data[i][31:0];
        assign gpu_exe_coords[1][i] = gpu_exe_if.rs2_data[i][31:0];
        assign gpu_exe_lod[i]       = gpu_exe_if.rs3_data[i][0 +: `VX_TEX_LOD_BITS];        
    end

    wire mdata_push = gpu_exe_if.valid && gpu_exe_if.ready;
    wire mdata_pop  = tex_bus_if.rsp_valid && tex_bus_if.rsp_ready;

    VX_index_buffer #(
        .DATAW (NW_WIDTH + `NUM_THREADS + `XLEN + `NR_BITS),
        .SIZE  (`TEX_REQ_QUEUE_SIZE)
    ) tag_store (
        .clk          (clk),
        .reset        (reset),
        .acquire_slot (mdata_push),       
        .write_addr   (mdata_waddr),                
        .read_addr    (mdata_raddr),
        .release_addr (mdata_raddr),        
        .write_data   ({gpu_exe_if.wid, gpu_exe_if.tmask, gpu_exe_if.PC, gpu_exe_if.rd}),
        .read_data    ({rsp_wid,        rsp_tmask,        rsp_PC,        rsp_rd}),
        .release_slot (mdata_pop),     
        .full         (mdata_full),
        `UNUSED_PIN (empty)
    );

    // submit texture request

    wire valid_in, ready_in;    
    assign valid_in = gpu_exe_if.valid && ~mdata_full;
    assign gpu_exe_if.ready = ready_in && ~mdata_full;    

    wire [`TEX_REQ_TAG_WIDTH-1:0] req_tag = {gpu_exe_if.uuid, mdata_waddr};

    VX_skid_buffer #(
        .DATAW   (`NUM_THREADS * (1 + 2 * 32 + `VX_TEX_LOD_BITS) + `VX_TEX_STAGE_BITS + `TEX_REQ_TAG_WIDTH),
        .OUT_REG (1)
    ) req_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (valid_in),
        .ready_in  (ready_in),
        .data_in   ({gpu_exe_if.tmask,    gpu_exe_coords,        gpu_exe_lod,        gpu_exe_stage,        req_tag}),
        .data_out  ({tex_bus_if.req_mask, tex_bus_if.req_coords, tex_bus_if.req_lod, tex_bus_if.req_stage, tex_bus_if.req_tag}),
        .valid_out (tex_bus_if.req_valid),
        .ready_out (tex_bus_if.req_ready)
    );

    // handle texture response

    assign mdata_raddr = tex_bus_if.rsp_tag[0 +: REQ_QUEUE_BITS];
    assign rsp_uuid    = tex_bus_if.rsp_tag[REQ_QUEUE_BITS +: UUID_WIDTH];

    wire [`NUM_THREADS-1:0][31:0] commit_data;

    VX_skid_buffer #(
        .DATAW (UUID_WIDTH + NW_WIDTH + `NUM_THREADS + `XLEN + `NR_BITS + (`NUM_THREADS * 32))
    ) rsp_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (tex_bus_if.rsp_valid),
        .ready_in  (tex_bus_if.rsp_ready),
        .data_in   ({rsp_uuid,       rsp_wid,       rsp_tmask,       rsp_PC,       rsp_rd,       tex_bus_if.rsp_texels}),
        .data_out  ({commit_if.uuid, commit_if.wid, commit_if.tmask, commit_if.PC, commit_if.rd, commit_data}),
        .valid_out (commit_if.valid),
        .ready_out (commit_if.ready)
    );

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign commit_if.data[i] = `XLEN'(commit_data[i]);
    end

    assign commit_if.wb  = 1'b1;
    assign commit_if.eop = 1'b1;

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (gpu_exe_if.valid && gpu_exe_if.ready) begin
            `TRACE(1, ("%d: core%0d-tex-req: wid=%0d, PC=0x%0h, tmask=%b, u=", $time, CORE_ID, gpu_exe_if.wid, gpu_exe_if.PC, gpu_exe_if.tmask));
            `TRACE_ARRAY1D(1, gpu_exe_if.coords[0], `NUM_THREADS);
            `TRACE(1, (", v="));
            `TRACE_ARRAY1D(1, gpu_exe_if.coords[1], `NUM_THREADS);
            `TRACE(1, (", lod="));
            `TRACE_ARRAY1D(1, gpu_exe_if.lod, `NUM_THREADS);
            `TRACE(1, (", stage=%0d, tag=0x%0h (#%0d)\n", gpu_exe_if.stage, req_tag, gpu_exe_if.uuid));
        end
        if (commit_if.valid && commit_if.ready) begin
            `TRACE(1, ("%d: core%0d-tex-rsp: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d, texels=", $time, CORE_ID, commit_if.wid, commit_if.PC, commit_if.tmask, commit_if.rd));
            `TRACE_ARRAY1D(1, commit_if.data, `NUM_THREADS);
            `TRACE(1, (" (#%0d)\n", commit_if.uuid));
        end
    end
`endif

endmodule
