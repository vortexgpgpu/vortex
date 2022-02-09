`include "VX_raster_define.vh"

module VX_raster_csr #(  
    parameter CORE_ID = 0
    // TODO
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_raster_csr_if.slave raster_csr_if,
    VX_raster_req_if.slave raster_req_if,

    // Output
    output raster_csrs_t raster_csrs
);

    raster_csrs_t reg_csrs;

    // CSRs write

    always @(posedge clk) begin
        if (reset) begin
            reg_csrs <= 0;
        end else if (rop_csr_if.write_enable) begin
            case (rop_csr_if.write_addr)
                `CSR_RASTER_PIDX_ADDR: begin 
                    reg_csrs.pidx_addr <= rop_csr_if.write_data[31:0];
                end
                `CSR_RASTER_PIDX_SIZE: begin 
                    reg_csrs.pidx_size <= rop_csr_if.write_data[31:0];
                end
                `CSR_RASTER_PBUF_ADDR: begin 
                    reg_csrs.pbuf_addr <= rop_csr_if.write_data[31:0];
                end
                `CSR_RASTER_PBUF_STRIDE: begin 
                    reg_csrs.pbuf_stride <= rop_csr_if.write_data[31:0];
                end
                `CSR_RASTER_TILE_XY: begin 
                    reg_csrs.tile_left <= rop_csr_if.write_data[15:0];
                    reg_csrs.tile_top  <= rop_csr_if.write_data[31:16];
                end
                `CSR_RASTER_TILE_WH: begin 
                    reg_csrs.tile_width  <= rop_csr_if.write_data[15:0];
                    reg_csrs.tile_height <= rop_csr_if.write_data[31:16];
                end
            endcase
        end
    end

    // CSRs read
    assign raster_csrs = reg_csrs;

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (raster_csr_if.write_enable) begin
            dpi_trace("%d: core%0d-raster-csr: state=", $time, CORE_ID);
            trace_raster_state(raster_csr_if.write_addr);
            dpi_trace(", data=%0h (#%0d)\n", raster_csr_if.write_data, raster_csr_if.write_uuid);
        end
    end
`endif

endmodule