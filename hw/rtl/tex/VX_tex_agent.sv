`include "VX_tex_define.vh"

module VX_tex_agent #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_gpu_csr_if.slave     tex_csr_if,
    VX_tex_agent_if.slave   tex_agent_if,
        
    // Outputs
    VX_tex_bus_if.master    tex_bus_if,
    VX_commit_if.master     tex_commit_if
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

    wire [UUID_WIDTH-1:0]   rsp_uuid;
    wire [NW_WIDTH-1:0]     rsp_wid;
    wire [`NUM_THREADS-1:0] rsp_tmask;
    wire [`XLEN-1:0]        rsp_PC;
    wire [`NR_BITS-1:0]     rsp_rd;
 
    wire [REQ_QUEUE_BITS-1:0] mdata_waddr, mdata_raddr;
    
    wire mdata_full;

    wire mdata_push = tex_agent_if.valid && tex_agent_if.ready;
    wire mdata_pop  = tex_bus_if.rsp_valid && tex_bus_if.rsp_ready;

    VX_index_buffer #(
        .DATAW (NW_WIDTH + `NUM_THREADS + 32 + `NR_BITS),
        .SIZE  (`TEX_REQ_QUEUE_SIZE)
    ) tag_store (
        .clk          (clk),
        .reset        (reset),
        .acquire_slot (mdata_push),       
        .write_addr   (mdata_waddr),                
        .read_addr    (mdata_raddr),
        .release_addr (mdata_raddr),        
        .write_data   ({tex_agent_if.wid, tex_agent_if.tmask, tex_agent_if.PC, tex_agent_if.rd}),                    
        .read_data    ({rsp_wid,          rsp_tmask,          rsp_PC,          rsp_rd}), 
        .release_slot (mdata_pop),     
        .full         (mdata_full),
        `UNUSED_PIN (empty)
    );

    // submit texture request

    wire valid_in, ready_in;    
    assign valid_in = tex_agent_if.valid && ~mdata_full;
    assign tex_agent_if.ready = ready_in && ~mdata_full;    

    wire [`TEX_REQ_TAG_WIDTH-1:0] req_tag = {tex_agent_if.uuid, mdata_waddr};

    VX_skid_buffer #(
        .DATAW   (`NUM_THREADS * (1 + 2 * 32 + `TEX_LOD_BITS) + `TEX_STAGE_BITS + `TEX_REQ_TAG_WIDTH),
        .OUT_REG (1)
    ) req_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (valid_in),
        .ready_in  (ready_in),
        .data_in   ({tex_agent_if.tmask,  tex_agent_if.coords,   tex_agent_if.lod,   tex_agent_if.stage,   req_tag}),
        .data_out  ({tex_bus_if.req_mask, tex_bus_if.req_coords, tex_bus_if.req_lod, tex_bus_if.req_stage, tex_bus_if.req_tag}),
        .valid_out (tex_bus_if.req_valid),
        .ready_out (tex_bus_if.req_ready)
    );

    // handle texture response

    assign mdata_raddr = tex_bus_if.rsp_tag[0 +: REQ_QUEUE_BITS];
    assign rsp_uuid    = tex_bus_if.rsp_tag[REQ_QUEUE_BITS +: UUID_WIDTH];

    VX_skid_buffer #(
        .DATAW (UUID_WIDTH + NW_WIDTH + `NUM_THREADS + 32 + `NR_BITS + (`NUM_THREADS * 32))
    ) rsp_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (tex_bus_if.rsp_valid),
        .ready_in  (tex_bus_if.rsp_ready),
        .data_in   ({rsp_uuid,           rsp_wid,           rsp_tmask,           rsp_PC,           rsp_rd,           tex_bus_if.rsp_texels}),
        .data_out  ({tex_commit_if.uuid, tex_commit_if.wid, tex_commit_if.tmask, tex_commit_if.PC, tex_commit_if.rd, tex_commit_if.data}),
        .valid_out (tex_commit_if.valid),
        .ready_out (tex_commit_if.ready)
    );

    assign tex_commit_if.wb  = 1'b1;
    assign tex_commit_if.eop = 1'b1;

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (tex_agent_if.valid && tex_agent_if.ready) begin
            `TRACE(1, ("%d: core%0d-tex-req: wid=%0d, PC=0x%0h, tmask=%b, u=", $time, CORE_ID, tex_agent_if.wid, tex_agent_if.PC, tex_agent_if.tmask));
            `TRACE_ARRAY1D(1, tex_agent_if.coords[0], `NUM_THREADS);
            `TRACE(1, (", v="));
            `TRACE_ARRAY1D(1, tex_agent_if.coords[1], `NUM_THREADS);
            `TRACE(1, (", lod="));
            `TRACE_ARRAY1D(1, tex_agent_if.lod, `NUM_THREADS);
            `TRACE(1, (", stage=%0d, tag=0x%0h (#%0d)\n", tex_agent_if.stage, req_tag, tex_agent_if.uuid));
        end
        if (tex_commit_if.valid && tex_commit_if.ready) begin
            `TRACE(1, ("%d: core%0d-tex-rsp: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d, texels=", $time, CORE_ID, tex_commit_if.wid, tex_commit_if.PC, tex_commit_if.tmask, tex_commit_if.rd));
            `TRACE_ARRAY1D(1, tex_commit_if.data, `NUM_THREADS);
            `TRACE(1, (" (#%0d)\n", tex_commit_if.uuid));
        end
    end
`endif

endmodule
