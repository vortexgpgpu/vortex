`include "VX_rop_define.vh"

module VX_rop_agent #(
    parameter CORE_ID = 0
) (
    input wire clk,    
    input wire reset,

    // Inputs    
    VX_rop_agent_if.slave rop_agent_if,    
    VX_gpu_csr_if.slave   rop_csr_if,  

    // Outputs    
    VX_commit_if.master   rop_commit_if,
    VX_rop_req_if.master  rop_req_if
);
    `UNUSED_PARAM (CORE_ID)
    
    localparam UUID_WIDTH = `UP(`UUID_BITS);
    localparam NW_WIDTH   = `UP(`NW_BITS);

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

    wire rop_req_valid, rop_req_ready;
    wire rop_rsp_valid, rop_rsp_ready;

    // it is possible to have ready = f(valid) when using arbiters, 
    // because of that we need to decouple rop_agent_if and rop_commit_if handshake with a pipe register

    VX_skid_buffer #(
        .DATAW   (UUID_WIDTH + `NUM_THREADS * (1 + 2 * `ROP_DIM_BITS + 32 + `ROP_DEPTH_BITS + 1)),
        .OUT_REG (1)
    ) req_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rop_req_valid),
        .ready_in  (rop_req_ready),
        .data_in   ({rop_agent_if.uuid, rop_agent_if.tmask, rop_agent_if.pos_x, rop_agent_if.pos_y, rop_agent_if.color, rop_agent_if.depth, rop_agent_if.face}),
        .data_out  ({rop_req_if.uuid,   rop_req_if.mask,    rop_req_if.pos_x,   rop_req_if.pos_y,   rop_req_if.color,   rop_req_if.depth,   rop_req_if.face}),
        .valid_out (rop_req_if.valid),
        .ready_out (rop_req_if.ready)
    );

    assign rop_req_valid = rop_agent_if.valid && rop_rsp_ready;
    assign rop_agent_if.ready = rop_req_ready && rop_rsp_ready;
    assign rop_rsp_valid = rop_agent_if.valid && rop_req_ready;

    VX_skid_buffer #(
        .DATAW (UUID_WIDTH + NW_WIDTH + `NUM_THREADS + 32)
    ) rsp_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rop_rsp_valid),
        .ready_in  (rop_rsp_ready),
        .data_in   ({rop_agent_if.uuid,  rop_agent_if.wid,  rop_agent_if.tmask,  rop_agent_if.PC}),
        .data_out  ({rop_commit_if.uuid, rop_commit_if.wid, rop_commit_if.tmask, rop_commit_if.PC}),
        .valid_out (rop_commit_if.valid),
        .ready_out (rop_commit_if.ready)
    );

    assign rop_commit_if.data = '0;
    assign rop_commit_if.rd   = '0;
    assign rop_commit_if.wb   = 0;
    assign rop_commit_if.eop  = 1;

`ifdef DBG_TRACE_ROP
    always @(posedge clk) begin
        if (rop_agent_if.valid && rop_agent_if.ready) begin
            `TRACE(1, ("%d: core%0d-rop-req: wid=%0d, PC=0x%0h, tmask=%b, x=", $time, CORE_ID, rop_agent_if.wid, rop_agent_if.PC, rop_agent_if.tmask));
            `TRACE_ARRAY1D(1, rop_agent_if.pos_x, `NUM_THREADS);
            `TRACE(1, (", y="));
            `TRACE_ARRAY1D(1, rop_agent_if.pos_y, `NUM_THREADS);
            `TRACE(1, (", face="));
            `TRACE_ARRAY1D(1, rop_agent_if.face, `NUM_THREADS);
            `TRACE(1, (", color="));
            `TRACE_ARRAY1D(1, rop_agent_if.color, `NUM_THREADS);
            `TRACE(1, (", depth="));
            `TRACE_ARRAY1D(1, rop_agent_if.depth, `NUM_THREADS);
            `TRACE(1, (", face=%b (#%0d)\n", rop_agent_if.face, rop_agent_if.uuid));
        end
    end
`endif

endmodule
