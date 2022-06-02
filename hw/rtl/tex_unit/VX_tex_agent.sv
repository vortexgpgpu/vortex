`include "VX_tex_define.vh"

module VX_tex_agent #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_gpu_csr_if.slave     tex_csr_if,
    VX_tex_agent_if.slave   tex_agent_req_if,    
    VX_tex_rsp_if.slave     tex_rsp_if,
        
    // Outputs
    VX_tex_req_if.master    tex_req_if,
    VX_commit_if.master     tex_agent_rsp_if
);
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

    // submit texture request

    wire [`TEX_REQ_TAG_WIDTH-1:0] tex_req_tag = {
        tex_agent_req_if.uuid,
        tex_agent_req_if.wid,
        tex_agent_req_if.PC,
        tex_agent_req_if.rd
    };

    VX_skid_buffer #(
        .DATAW (`NUM_THREADS * (1 + 2 * 32 + `TEX_LOD_BITS) + `TEX_STAGE_BITS + `TEX_REQ_TAG_WIDTH)
    ) req_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (tex_agent_req_if.valid),
        .ready_in  (tex_agent_req_if.ready),
        .data_in   ({tex_agent_req_if.tmask, tex_agent_req_if.coords, tex_agent_req_if.lod, tex_agent_req_if.stage, tex_req_tag}),
        .data_out  ({tex_req_if.mask,        tex_req_if.coords,       tex_req_if.lod,       tex_req_if.stage,       tex_req_if.tag}),
        .valid_out (tex_req_if.valid),
        .ready_out (tex_req_if.ready)
    );

    // handle texture response

    wire [`UUID_BITS-1:0] tex_rsp_uuid;
    wire [`NW_BITS-1:0]   tex_rsp_wid;
    wire [31:0]           tex_rsp_PC;
    wire [`NR_BITS-1:0]   tex_rsp_rd;

    assign {tex_rsp_uuid, tex_rsp_wid, tex_rsp_PC, tex_rsp_rd} = tex_rsp_if.tag;

    VX_skid_buffer #(
        .DATAW (`UUID_BITS + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + (`NUM_THREADS * 32))
    ) rsp_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (tex_rsp_if.valid),
        .ready_in  (tex_rsp_if.ready),
        .data_in   ({tex_rsp_uuid,          tex_rsp_wid,          tex_rsp_if.mask,        tex_rsp_PC,          tex_rsp_rd,          tex_rsp_if.texels}),
        .data_out  ({tex_agent_rsp_if.uuid, tex_agent_rsp_if.wid, tex_agent_rsp_if.tmask, tex_agent_rsp_if.PC, tex_agent_rsp_if.rd, tex_agent_rsp_if.data}),
        .valid_out (tex_agent_rsp_if.valid),
        .ready_out (tex_agent_rsp_if.ready)
    );

    assign tex_agent_rsp_if.wb   = 1'b1;
    assign tex_agent_rsp_if.eop  = 1'b1;

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (tex_agent_req_if.valid && tex_agent_req_if.ready) begin
            `TRACE(1, ("%d: core%0d-tex-req: wid=%0d, PC=0x%0h, tmask=%b, u=", $time, CORE_ID, tex_agent_req_if.wid, tex_agent_req_if.PC, tex_agent_req_if.tmask));
            `TRACE_ARRAY1D(1, tex_agent_req_if.coords[0], `NUM_THREADS);
            `TRACE(1, (", v="));
            `TRACE_ARRAY1D(1, tex_agent_req_if.coords[1], `NUM_THREADS);
            `TRACE(1, (", lod="));
            `TRACE_ARRAY1D(1, tex_agent_req_if.lod, `NUM_THREADS);
            `TRACE(1, (", stage=%0d, tag=0x%0h (#%0d)\n", tex_req_tag, tex_agent_req_if.stage, tex_agent_req_if.uuid));
        end
        if (tex_agent_rsp_if.valid && tex_agent_rsp_if.ready) begin
            `TRACE(1, ("%d: core%0d-tex-rsp: wid=%0d, PC=0x%0h, tmask=%b, rd=%0d, texels=", $time, CORE_ID, tex_agent_rsp_if.wid, tex_agent_rsp_if.PC, tex_agent_rsp_if.tmask, tex_agent_rsp_if.rd));
            `TRACE_ARRAY1D(1, tex_agent_rsp_if.data, `NUM_THREADS);
            `TRACE(1, (" (#%0d)\n", tex_agent_rsp_if.uuid));
        end
    end
`endif

endmodule
