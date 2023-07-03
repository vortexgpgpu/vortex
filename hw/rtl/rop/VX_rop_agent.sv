`include "VX_rop_define.vh"

module VX_rop_agent #(
    parameter CORE_ID = 0
) (
    input wire clk,    
    input wire reset,

    // Inputs    
    VX_gpu_exe_if.slave     gpu_exe_if,
    VX_gpu_csr_if.slave     rop_csr_if,  

    // Outputs    
    VX_rop_bus_if.master    rop_bus_if,
    VX_commit_if.master     commit_if
);
    `UNUSED_PARAM (CORE_ID)
    
    localparam UUID_WIDTH = `UP(`UUID_BITS);
    localparam NW_WIDTH   = `UP(`NW_BITS);

    wire [`NUM_THREADS-1:0][`VX_ROP_DIM_BITS-1:0] gpu_exe_pos_x;
    wire [`NUM_THREADS-1:0][`VX_ROP_DIM_BITS-1:0] gpu_exe_pos_y;
    wire [`NUM_THREADS-1:0]                       gpu_exe_face;
    wire [`NUM_THREADS-1:0][31:0]                 gpu_exe_color;
    wire [`NUM_THREADS-1:0][`VX_ROP_DEPTH_BITS-1:0] gpu_exe_depth;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign gpu_exe_face[i]  = gpu_exe_if.rs1_data[i][0];
        assign gpu_exe_pos_x[i] = gpu_exe_if.rs1_data[i][1 +: `VX_ROP_DIM_BITS];
        assign gpu_exe_pos_y[i] = gpu_exe_if.rs1_data[i][16 +: `VX_ROP_DIM_BITS];
        assign gpu_exe_color[i] = gpu_exe_if.rs2_data[i][31:0];
        assign gpu_exe_depth[i] = gpu_exe_if.rs3_data[i][`VX_ROP_DEPTH_BITS-1:0];
    end

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
    // because of that we need to decouple gpu_exe_if and commit_if handshake with a pipe register

    VX_skid_buffer #(
        .DATAW   (UUID_WIDTH + `NUM_THREADS * (1 + 2 * `VX_ROP_DIM_BITS + 32 + `VX_ROP_DEPTH_BITS + 1)),
        .OUT_REG (1)
    ) req_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rop_req_valid),
        .ready_in  (rop_req_ready),
        .data_in   ({gpu_exe_if.uuid,     gpu_exe_if.tmask,    gpu_exe_pos_x,        gpu_exe_pos_y,        gpu_exe_color,        gpu_exe_depth,        gpu_exe_face}),
        .data_out  ({rop_bus_if.req_uuid, rop_bus_if.req_mask, rop_bus_if.req_pos_x, rop_bus_if.req_pos_y, rop_bus_if.req_color, rop_bus_if.req_depth, rop_bus_if.req_face}),
        .valid_out (rop_bus_if.req_valid),
        .ready_out (rop_bus_if.req_ready)
    );

    assign rop_req_valid = gpu_exe_if.valid && rop_rsp_ready;
    assign gpu_exe_if.ready = rop_req_ready && rop_rsp_ready;
    assign rop_rsp_valid = gpu_exe_if.valid && rop_req_ready;

    VX_skid_buffer #(
        .DATAW (UUID_WIDTH + NW_WIDTH + `NUM_THREADS + `XLEN)
    ) rsp_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rop_rsp_valid),
        .ready_in  (rop_rsp_ready),
        .data_in   ({gpu_exe_if.uuid, gpu_exe_if.wid, gpu_exe_if.tmask, gpu_exe_if.PC}),
        .data_out  ({commit_if.uuid,  commit_if.wid,  commit_if.tmask,  commit_if.PC}),
        .valid_out (commit_if.valid),
        .ready_out (commit_if.ready)
    );

    assign commit_if.data = '0;
    assign commit_if.rd   = '0;
    assign commit_if.wb   = 0;
    assign commit_if.eop  = 1;

`ifdef DBG_TRACE_ROP
    always @(posedge clk) begin
        if (gpu_exe_if.valid && gpu_exe_if.ready) begin
            `TRACE(1, ("%d: core%0d-rop-req: wid=%0d, PC=0x%0h, tmask=%b, x=", $time, CORE_ID, gpu_exe_if.wid, gpu_exe_if.PC, gpu_exe_if.tmask));
            `TRACE_ARRAY1D(1, gpu_exe_if.pos_x, `NUM_THREADS);
            `TRACE(1, (", y="));
            `TRACE_ARRAY1D(1, gpu_exe_if.pos_y, `NUM_THREADS);
            `TRACE(1, (", face="));
            `TRACE_ARRAY1D(1, gpu_exe_if.face, `NUM_THREADS);
            `TRACE(1, (", color="));
            `TRACE_ARRAY1D(1, gpu_exe_if.color, `NUM_THREADS);
            `TRACE(1, (", depth="));
            `TRACE_ARRAY1D(1, gpu_exe_if.depth, `NUM_THREADS);
            `TRACE(1, (", face=%b (#%0d)\n", gpu_exe_if.face, gpu_exe_if.uuid));
        end
    end
`endif

endmodule
