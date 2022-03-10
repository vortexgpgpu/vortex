`include "VX_raster_define.vh"

module VX_raster_dcr #(  
    parameter CORE_ID = 0
    // TODO
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_raster_dcr_if.slave raster_dcr_if,
    // TODO: Not used
    //VX_raster_req_if.slave raster_req_if,

    // Output
    output raster_dcrs_t raster_dcrs
);

    // DCR registers
    raster_dcrs_t reg_dcrs;

    // DCR read
    always @(posedge clk) begin
        if (reset) begin
            reg_dcrs.pidx_addr      <= 0;
            reg_dcrs.pidx_size      <= 0;
            reg_dcrs.pbuf_stride    <= 0;
            reg_dcrs.tile_left      <= 0;
            reg_dcrs.tile_top       <= 0;
            reg_dcrs.tile_width     <= 0;
            reg_dcrs.tile_height    <= 0;
        end else if (raster_dcr_if.write_enable) begin
            case (raster_dcr_if.write_addr)
                `DCR_RASTER_PIDX_ADDR: begin 
                    reg_dcrs.pidx_addr <= raster_dcr_if.write_data[`RASTER_DCR_DATA_BITS-1:0];
                end
                `DCR_RASTER_PIDX_SIZE: begin 
                    reg_dcrs.pidx_size <= raster_dcr_if.write_data[`RASTER_DCR_DATA_BITS-1:0];
                end
                `DCR_RASTER_PBUF_ADDR: begin 
                    reg_dcrs.pbuf_addr <= raster_dcr_if.write_data[`RASTER_DCR_DATA_BITS-1:0];
                end
                `DCR_RASTER_PBUF_STRIDE: begin 
                    reg_dcrs.pbuf_stride <= raster_dcr_if.write_data[`RASTER_DCR_DATA_BITS:0];
                end
                `DCR_RASTER_TILE_XY: begin 
                    reg_dcrs.tile_left <= raster_dcr_if.write_data[`RASTER_TILE_DATA_BITS-1:0];
                    reg_dcrs.tile_top  <= raster_dcr_if.write_data[`RASTER_DCR_DATA_BITS-1:`RASTER_TILE_DATA_BITS];
                end
                `DCR_RASTER_TILE_WH: begin 
                    reg_dcrs.tile_width  <= raster_dcr_if.write_data[`RASTER_DCR_DATA_BITS-1:0];
                    reg_dcrs.tile_height <= raster_dcr_if.write_data[`RASTER_DCR_DATA_BITS-1:RASTER_DCR_DATA_BITS];
                end
            endcase
        end
    end

    // Data write to output
    assign raster_dcrs = reg_dcrs;

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (raster_dcr_if.write_enable) begin
            dpi_trace("%d: core%0d-raster-dcr: state=", $time, CORE_ID);
            trace_raster_state(raster_dcr_if.write_addr);
            dpi_trace(", data=0x%0h (#%0d)\n", raster_dcr_if.write_data, raster_dcr_if.write_uuid);
        end
    end
`endif

endmodule