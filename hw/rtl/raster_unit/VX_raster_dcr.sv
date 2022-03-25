`include "VX_raster_define.vh"

module VX_raster_dcr #(  
    parameter CORE_ID = 0
    // TODO
) (
    input wire clk,
    input wire reset,

    // Inputs
    input  wire                             dcr_wr_valid,
    input  wire [`VX_DCR_ADDR_WIDTH-1:0]    dcr_wr_addr,
    input  wire [`VX_DCR_DATA_WIDTH-1:0]    dcr_wr_data,

    // Output
    VX_raster_dcr_if.master     raster_dcr_if
);

    // DCR registers
    raster_dcrs_t dcrs;

    // DCRs write
    always @(posedge clk) begin
        if (reset) begin
            dcrs.tbuf_addr      <= 0;
            dcrs.tile_count     <= 0;
            dcrs.pbuf_addr      <= 0;
            dcrs.pbuf_stride    <= 0;
        end else if (dcr_wr_valid) begin
            case (dcr_wr_addr)
                `DCR_RASTER_TBUF_ADDR: begin 
                    dcrs.tbuf_addr <= dcr_wr_data[`RASTER_DCR_DATA_BITS-1:0];
                end
                `DCR_RASTER_TILE_COUNT: begin 
                    dcrs.tile_count <= dcr_wr_data[`RASTER_DCR_DATA_BITS-1:0];
                end
                `DCR_RASTER_PBUF_ADDR: begin 
                    dcrs.pbuf_addr <= dcr_wr_data[`RASTER_DCR_DATA_BITS-1:0];
                end
                `DCR_RASTER_PBUF_STRIDE: begin 
                    dcrs.pbuf_stride <= dcr_wr_data[`RASTER_DCR_DATA_BITS-1:0];
                end
                `DCR_RASTER_DST_SIZE: begin 
                    dcrs.dst_width  <= dcr_wr_data[0 +: `RASTER_DIM_BITS];
                    dcrs.dst_height <= dcr_wr_data[16 +: `RASTER_DIM_BITS];
                end
            endcase
        end
    end

    // DCRs read
    assign raster_dcr_if.data = dcrs;

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (dcr_wr_valid) begin
            dpi_trace("%d: raster-dcr: state=", $time);
            trace_raster_state(dcr_wr_addr);
            dpi_trace(", data=0x%0h\n", dcr_wr_data);
        end
    end
`endif

endmodule
