`include "VX_raster_define.vh"

module VX_raster_csr #( 
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

    // Inputs    
    input wire                              write_enable,
    input wire [`UUID_BITS-1:0]             write_uuid,
    input wire [`NW_BITS-1:0]               write_wid,
    input wire [`NUM_THREADS-1:0]           write_tmask,
    input raster_stamp_t [`NUM_THREADS-1:0] write_data,

    // Output
    VX_gpu_csr_if.slave raster_csr_if
);
    `UNUSED_VAR (reset)

    raster_csrs_t [`NUM_THREADS-1:0] wdata;
    raster_csrs_t [`NUM_THREADS-1:0] rdata;
    wire [`NUM_THREADS-1:0]          wren;
    wire [`NW_BITS-1:0]              waddr;
    wire [`NW_BITS-1:0]              raddr;

    // CSR registers
    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        VX_dp_ram #(
            .DATAW       ($bits(raster_csrs_t)),
            .SIZE        (`NUM_WARPS),
            .INIT_ENABLE (1),
            .INIT_VALUE  (0)
        ) dp_ram (
            .clk   (clk),
            .wren  (wren[i]),
            .waddr (waddr),
            .wdata (wdata[i]),
            .raddr (raddr),
            .rdata (rdata[i])
        );
    end

    // CSRs write

    assign wren  = {`NUM_THREADS{write_enable}} & write_tmask;
    assign waddr = write_wid;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign wdata[i].pos_mask = {write_data[i].pos_y, write_data[i].pos_x, write_data[i].mask}; 
        assign wdata[i].bcoord_x = write_data[i].bcoord_x;
        assign wdata[i].bcoord_y = write_data[i].bcoord_y;
        assign wdata[i].bcoord_z = write_data[i].bcoord_z;    
    end
    
    // CSRs read

    assign raddr = raster_csr_if.read_wid;

    reg [`NUM_THREADS-1:0][31:0] read_data_r;
    always @(*) begin
        for (integer i = 0; i < `NUM_THREADS; ++i) begin
            read_data_r[i] = 'x;
            case (raster_csr_if.read_addr)
                `CSR_RASTER_POS_MASK:  read_data_r[i] = rdata[i].pos_mask;
                `CSR_RASTER_BCOORD_X0: read_data_r[i] = rdata[i].bcoord_x[0];
                `CSR_RASTER_BCOORD_X1: read_data_r[i] = rdata[i].bcoord_x[1];
                `CSR_RASTER_BCOORD_X2: read_data_r[i] = rdata[i].bcoord_x[2];
                `CSR_RASTER_BCOORD_X3: read_data_r[i] = rdata[i].bcoord_x[3];
                `CSR_RASTER_BCOORD_Y0: read_data_r[i] = rdata[i].bcoord_y[0];
                `CSR_RASTER_BCOORD_Y1: read_data_r[i] = rdata[i].bcoord_y[1];
                `CSR_RASTER_BCOORD_Y2: read_data_r[i] = rdata[i].bcoord_y[2];
                `CSR_RASTER_BCOORD_Y3: read_data_r[i] = rdata[i].bcoord_y[3];
                `CSR_RASTER_BCOORD_Z0: read_data_r[i] = rdata[i].bcoord_z[0];
                `CSR_RASTER_BCOORD_Z1: read_data_r[i] = rdata[i].bcoord_z[1];
                `CSR_RASTER_BCOORD_Z2: read_data_r[i] = rdata[i].bcoord_z[2];
                `CSR_RASTER_BCOORD_Z3: read_data_r[i] = rdata[i].bcoord_z[3];
                default:;
            endcase
        end
    end

    assign raster_csr_if.read_data = read_data_r;

    `UNUSED_VAR (write_uuid)

    `UNUSED_VAR (raster_csr_if.read_enable)
    `UNUSED_VAR (raster_csr_if.read_uuid)
    `UNUSED_VAR (raster_csr_if.read_tmask)

    `UNUSED_VAR (raster_csr_if.write_enable)
    `UNUSED_VAR (raster_csr_if.write_addr)
    `UNUSED_VAR (raster_csr_if.write_data)
    `UNUSED_VAR (raster_csr_if.write_uuid)
    `UNUSED_VAR (raster_csr_if.write_wid)
    `UNUSED_VAR (raster_csr_if.write_tmask)

`ifdef DBG_TRACE_TEX
    logic [`NUM_THREADS-1:0][`RASTER_DIM_BITS-2:0] pos_x;
    logic [`NUM_THREADS-1:0][`RASTER_DIM_BITS-2:0] pos_y;
    logic [`NUM_THREADS-1:0][3:0]                  mask;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign pos_x[i] = write_data[i].pos_x;
        assign pos_y[i] = write_data[i].pos_y;
        assign mask[i]  = write_data[i].mask;
    end

    always @(posedge clk) begin
        if (raster_csr_if.read_enable) begin
            dpi_trace(1, "%d: core%0d-raster-csr-read: wid=%0d, tmask=%b, state=", $time, CORE_ID, raster_csr_if.read_wid, raster_csr_if.read_tmask);
            trace_raster_csr(1, raster_csr_if.read_addr);
            dpi_trace(1, ", data=");
            `TRACE_ARRAY1D(1, raster_csr_if.read_data, `NUM_THREADS);
            dpi_trace(1, " (#%0d)\n", raster_csr_if.read_uuid);
        end
        if (write_enable) begin
            dpi_trace(1, "%d: core%0d-raster-fetch: wid=%0d, tmask=%b, pos_x=", $time, CORE_ID, write_wid, write_tmask);
            `TRACE_ARRAY1D(1, pos_x, `NUM_THREADS);
            dpi_trace(1, ", pos_y=");
            `TRACE_ARRAY1D(1, pos_y, `NUM_THREADS);
            dpi_trace(1, ", mask=");
            `TRACE_ARRAY1D(1, mask, `NUM_THREADS);
            dpi_trace(1, " (#%0d)\n", write_uuid);
        end
    end
`endif

endmodule
