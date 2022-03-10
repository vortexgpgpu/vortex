`include "VX_raster_define.vh"

module VX_raster_csr #(  
    parameter CORE_ID = 0
    // TODO
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_raster_csr_if.slave raster_csr_if,
    // TODO: Not used
    //VX_raster_req_if.slave raster_req_if,

    // Output
    output raster_csrs_t raster_csrs
);

    // CSR registers
    raster_csrs_t reg_csrs;

    // CSR read
    always @(posedge clk) begin
        if (reset) begin
            reg_csrs.pidx_addr      <= 0;
            reg_csrs.pidx_size      <= 0;
            reg_csrs.pbuf_stride    <= 0;
            reg_csrs.tile_left      <= 0;
            reg_csrs.tile_top       <= 0;
            reg_csrs.tile_width     <= 0;
            reg_csrs.tile_height    <= 0;
        end else if (raster_csr_if.write_enable) begin
            case (raster_csr_if.write_addr)
                `CSR_RASTER_PIDX_ADDR: begin 
                    reg_csrs.pidx_addr <= raster_csr_if.write_data[`RASTER_CSR_DATA_BITS-1:0];
                end
                `CSR_RASTER_PIDX_SIZE: begin 
                    reg_csrs.pidx_size <= raster_csr_if.write_data[`RASTER_CSR_DATA_BITS-1:0];
                end
                `CSR_RASTER_PBUF_ADDR: begin 
                    reg_csrs.pbuf_addr <= raster_csr_if.write_data[`RASTER_CSR_DATA_BITS-1:0];
                end
                `CSR_RASTER_PBUF_STRIDE: begin 
                    reg_csrs.pbuf_stride <= raster_csr_if.write_data[`RASTER_CSR_DATA_BITS:0];
                end
                `CSR_RASTER_TILE_XY: begin 
                    reg_csrs.tile_left <= raster_csr_if.write_data[`RASTER_TILE_DATA_BITS-1:0];
                    reg_csrs.tile_top  <= raster_csr_if.write_data[`RASTER_CSR_DATA_BITS-1:`RASTER_TILE_DATA_BITS];
                end
                `CSR_RASTER_TILE_WH: begin 
                    reg_csrs.tile_width  <= raster_csr_if.write_data[`RASTER_CSR_DATA_BITS-1:0];
                    reg_csrs.tile_height <= raster_csr_if.write_data[`RASTER_CSR_DATA_BITS-1:RASTER_CSR_DATA_BITS];
                end
            endcase
        end
    end

    // Data write to output
    assign raster_csrs = reg_csrs;

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (raster_csr_if.write_enable) begin
            dpi_trace("%d: core%0d-raster-csr: state=", $time, CORE_ID);
            trace_raster_state(raster_csr_if.write_addr);
            dpi_trace(", data=0x%0h (#%0d)\n", raster_csr_if.write_data, raster_csr_if.write_uuid);
        end
    end
`endif

endmodule