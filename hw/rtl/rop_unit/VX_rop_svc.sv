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

    wire rop_rsp_valid_out, rop_rsp_ready_out;

    VX_skid_buffer #(
        .DATAW (`NUM_THREADS + `UUID_BITS + `NW_BITS + 32 + `NUM_THREADS * (2 * `ROP_DIM_BITS + 32 + `ROP_DEPTH_BITS + 1))
    ) rsp_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rop_svc_req_if.valid),
        .ready_in  (rop_svc_req_if.ready),
        .data_in   ({rop_svc_req_if.tmask, rop_svc_req_if.uuid, rop_svc_req_if.wid, rop_svc_req_if.PC, rop_svc_req_if.pos_x, rop_svc_req_if.pos_y, rop_svc_req_if.color, rop_svc_req_if.depth, rop_svc_req_if.backface}),
        .data_out  ({rop_svc_rsp_if.tmask, rop_svc_rsp_if.uuid, rop_svc_rsp_if.wid, rop_svc_rsp_if.PC, rop_req_if.pos_x,     rop_req_if.pos_y,     rop_req_if.color,     rop_req_if.depth,     rop_req_if.backface}),
        .valid_out (rop_rsp_valid_out),
        .ready_out (rop_rsp_ready_out)
    );

    assign rop_rsp_ready_out    = rop_req_if.ready  && rop_svc_rsp_if.ready;
	assign rop_svc_rsp_if.valid = rop_rsp_valid_out && rop_req_if.ready;
	assign rop_req_if.valid     = rop_rsp_valid_out && rop_svc_rsp_if.ready;
    assign rop_req_if.tmask     = rop_svc_rsp_if.tmask;    

    assign rop_svc_rsp_if.data = 'x;
    assign rop_svc_rsp_if.rd   = 'x;
    assign rop_svc_rsp_if.wb   = 0;
    assign rop_svc_rsp_if.eop  = 1'b1;

`ifdef DBG_TRACE_ROP
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
            dpi_trace(1, " (#%0d)\n", rop_svc_req_if.uuid);
        end
    end
`endif

endmodule
