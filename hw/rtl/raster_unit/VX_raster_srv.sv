`include "VX_raster_define.vh"

module VX_raster_srv #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

    // Inputs    
    VX_raster_srv_if.slave  raster_srv_if,
    VX_gpu_csr_if.slave     raster_csr_if,
    
    // Outputs
    output wire [`NUM_THREADS-1:0][15:0] fragment_x,
    output wire [`NUM_THREADS-1:0][15:0] fragment_y,
    VX_commit_if.master     raster_rsp_if
);
    `UNUSED_VAR (raster_srv_if.valid)
    `UNUSED_VAR (raster_srv_if.uuid)
    `UNUSED_VAR (raster_srv_if.wid)
    `UNUSED_VAR (raster_srv_if.tmask)
    `UNUSED_VAR (raster_srv_if.PC)
    `UNUSED_VAR (raster_srv_if.rd)
    `UNUSED_VAR (raster_srv_if.wb)
    assign raster_srv_if.ready = 0;

    assign raster_rsp_if.valid = 0;
    
    assign fragment_x = '0;
    assign fragment_y = '0;
    
    // TODO
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

endmodule