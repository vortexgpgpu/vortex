`include "VX_raster_define.vh"

module VX_raster_svc #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

    // Inputs    
    VX_raster_svc_if.slave raster_svc_req_if,    
    VX_raster_req_if.slave raster_req_if,
        
    // Outputs
    VX_commit_if.master     raster_svc_rsp_if,
    VX_gpu_csr_if.slave     raster_csr_if    
);
    wire stall_out;

    // CSRs access

    wire csr_write_enable = raster_req_if.valid & raster_svc_req_if.valid & ~stall_out;

    VX_raster_csr #(
        .CORE_ID (CORE_ID)
    ) raster_csr (
        .clk            (clk),
        .reset          (reset),
        // inputs
        .write_enable   (csr_write_enable),    
        .write_uuid     (raster_svc_req_if.uuid),
        .write_wid      (raster_svc_req_if.wid),
        .write_tmask    (raster_svc_req_if.tmask),
        .write_data     (raster_req_if.stamps),
        // outputs
        .raster_csr_if  (raster_csr_if)
    );

    // it is possible to have ready = f(valid) when using arbiters, 
    // because of that we need to decouple raster_svc_req_if and raster_svc_rsp_if handshake with a pipe register

    assign raster_svc_req_if.ready = raster_req_if.valid & ~stall_out;

    assign raster_req_if.ready = raster_svc_req_if.valid & ~stall_out;

    wire response_valid = raster_svc_req_if.valid & raster_req_if.valid;

    wire [`NUM_THREADS-1:0][31:0] response_data;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign response_data[i] = {31'(raster_req_if.stamps[i].pid), !raster_req_if.empty};
    end

    assign stall_out = ~raster_svc_rsp_if.ready && raster_svc_rsp_if.valid;

    VX_pipe_register #(
        .DATAW  (1 + `UUID_BITS + `NW_BITS + `NUM_THREADS + 32 + `NR_BITS + 1 + (`NUM_THREADS * 32)),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (!stall_out),
        .data_in  ({response_valid,          raster_svc_req_if.uuid, raster_svc_req_if.wid, raster_svc_req_if.tmask, raster_svc_req_if.PC, raster_svc_req_if.rd, raster_svc_req_if.wb, response_data}),
        .data_out ({raster_svc_rsp_if.valid, raster_svc_rsp_if.uuid, raster_svc_rsp_if.wid, raster_svc_rsp_if.tmask, raster_svc_rsp_if.PC, raster_svc_rsp_if.rd, raster_svc_rsp_if.wb, raster_svc_rsp_if.data})
    );

    assign raster_svc_rsp_if.eop  = 1'b1;


`ifdef DBG_TRACE_CORE_PIPELINE
    always @(posedge clk) begin
        if (csr_write_enable) begin
            for (integer i = 0; i < `NUM_THREADS; ++i) begin
                dpi_trace(1, "raster-quad-csr-write: %d: empty=%b, x_loc = %0d, y_loc = %0d, pid = %0d, mask = %0d bcoords = %0d %0d %0d %0d, %0d %0d %0d %0d, %0d %0d %0d %0d\n", 
                    $time, raster_req_if.empty,
                    raster_req_if.stamps[i].pos_x,  raster_req_if.stamps[i].pos_y, raster_req_if.stamps[i].pid, raster_req_if.stamps[i].mask,
                    raster_req_if.stamps[i].bcoord_x[0], raster_req_if.stamps[i].bcoord_x[1], raster_req_if.stamps[i].bcoord_x[2], raster_req_if.stamps[i].bcoord_x[3],
                    raster_req_if.stamps[i].bcoord_y[0], raster_req_if.stamps[i].bcoord_y[1], raster_req_if.stamps[i].bcoord_y[2], raster_req_if.stamps[i].bcoord_y[3],
                    raster_req_if.stamps[i].bcoord_z[0], raster_req_if.stamps[i].bcoord_z[1], raster_req_if.stamps[i].bcoord_z[2], raster_req_if.stamps[i].bcoord_z[3]
                );
            end
        end
    end
`endif

endmodule