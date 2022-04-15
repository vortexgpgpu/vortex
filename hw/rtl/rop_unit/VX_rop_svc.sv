`include "VX_rop_define.vh"

module VX_rop_svc #(
    parameter CORE_ID = 0
) (
    input wire clk,    
    input wire reset,

    // Inputs    
    VX_rop_svc_if.slave     rop_svc_req_if,    
    VX_gpu_csr_if.slave     rop_csr_if,  

    // Outputs    
    VX_commit_if.master     rop_svc_rsp_if,
    VX_rop_req_if.master    rop_req_if
);
    // CSRs access

    rop_csrs_t rop_csrs;

    VX_rop_csr #(
        .CORE_ID    (CORE_ID)
    ) rop_csr (
        .clk        (clk),
        .reset      (reset),

        // inputs
        .rop_csr_if (rop_csr_if),

        // outputs
        .rop_csrs   (rop_csrs)
    );

    `UNUSED_VAR (rop_csrs)

    wire stall_out;

    // it is possible to have ready = f(valid) when using arbiters, 
    // because of that we need to decouple rop_svc_req_if and rop_svc_rsp_if handshake with a pipe register

    assign rop_req_if.valid    = rop_svc_req_if.valid & ~stall_out;
    assign rop_req_if.tmask    = rop_svc_req_if.tmask;
    assign rop_req_if.pos_x    = rop_svc_req_if.pos_x;
    assign rop_req_if.pos_y    = rop_svc_req_if.pos_y;
    assign rop_req_if.color    = rop_svc_req_if.color;
    assign rop_req_if.depth    = rop_svc_req_if.depth;
    assign rop_req_if.backface = rop_svc_req_if.backface;

    assign rop_svc_req_if.ready = rop_req_if.ready & ~stall_out;

    wire response_valid = rop_svc_req_if.valid & rop_req_if.ready;

    assign stall_out = ~rop_svc_rsp_if.ready && rop_svc_rsp_if.valid;

    VX_pipe_register #(
        .DATAW  (1 + `UUID_BITS + `NW_BITS + `NUM_THREADS + 32),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall_out),
        .data_in  ({response_valid,       rop_svc_req_if.uuid, rop_svc_req_if.wid, rop_svc_req_if.tmask, rop_svc_req_if.PC}),
        .data_out ({rop_svc_rsp_if.valid, rop_svc_rsp_if.uuid, rop_svc_rsp_if.wid, rop_svc_rsp_if.tmask, rop_svc_rsp_if.PC})
    );

    assign rop_svc_rsp_if.data = 'x;
    assign rop_svc_rsp_if.rd   = 'x;
    assign rop_svc_rsp_if.wb   = 0;
    assign rop_svc_rsp_if.eop  = 1'b1;

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (rop_svc_req_if.valid && rop_svc_req_if.ready) begin
            dpi_trace(1, "%d: core%0d-rop-req: wid=%0d, PC=0x%0h, tmask=%b, x=", $time, CORE_ID, rop_svc_req_if.wid, rop_svc_req_if.PC, rop_svc_req_if.tmask);
            `TRACE_ARRAY1D(1, rop_svc_req_if.pos_x, `NUM_THREADS);
            dpi_trace(1, ", y=");
            `TRACE_ARRAY1D(1, rop_svc_req_if.pos_y, `NUM_THREADS);
            dpi_trace(1, ", backface=");
            `TRACE_ARRAY1D(1, rop_svc_req_if.backface, `NUM_THREADS);
            dpi_trace(1, ", color=");
            `TRACE_ARRAY1D(1, rop_svc_req_if.color, `NUM_THREADS);
            dpi_trace(1, ", depth=");
            `TRACE_ARRAY1D(1, rop_svc_req_if.depth, `NUM_THREADS);
            dpi_trace(1, " (#%0d)\n", tex_req_if.uuid);
        end
    end
`endif

endmodule
