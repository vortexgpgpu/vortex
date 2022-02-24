`include "VX_raster_define.vh"

module VX_raster_csr (
    input wire clk,
    input wire reset,

   // Inputs
    input  wire                             csr_wr_valid,
    input  wire [`VX_CSR_ADDR_WIDTH-1:0]    csr_wr_addr,
    input  wire [`VX_CSR_DATA_WIDTH-1:0]    csr_wr_data,

    // Output
    VX_raster_csr_if.master raster_csr_if
);

    raster_csrs_t csrs;

    // CSRs write

    always @(posedge clk) begin
        if (reset) begin
            csrs <= 0;
        end else if (csr_wr_valid) begin
            case (csr_wr_addr)
                `CSR_RASTER_PIDX_ADDR: begin 
                    csrs.pidx_addr <= csr_wr_data[31:0];
                end
                `CSR_RASTER_PIDX_SIZE: begin 
                    csrs.pidx_size <= csr_wr_data[31:0];
                end
                `CSR_RASTER_PBUF_ADDR: begin 
                    csrs.pbuf_addr <= csr_wr_data[31:0];
                end
                `CSR_RASTER_PBUF_STRIDE: begin 
                    csrs.pbuf_stride <= csr_wr_data[31:0];
                end
                `CSR_RASTER_TILE_XY: begin 
                    csrs.tile_left <= csr_wr_data[15:0];
                    csrs.tile_top  <= csr_wr_data[31:16];
                end
                `CSR_RASTER_TILE_WH: begin 
                    csrs.tile_width  <= csr_wr_data[15:0];
                    csrs.tile_height <= csr_wr_data[31:16];
                end
            endcase
        end
    end

    // CSRs read
    assign raster_csr_if.data = csrs;

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (csr_wr_valid) begin
            dpi_trace("%d: raster-csr: state=", $time);
            trace_raster_state(csr_wr_addr);
            dpi_trace(", data=0x%0h\n", csr_wr_data);
        end
    end
`endif

endmodule