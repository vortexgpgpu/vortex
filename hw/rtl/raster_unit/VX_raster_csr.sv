`include "VX_raster_define.vh"

module VX_raster_csr #( 
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

    // Inputs    
    input wire                              write_enable,
    input wire [`UP(`UUID_BITS)-1:0]        write_uuid,
    input wire [`UP(`NW_BITS)-1:0]          write_wid,
    input wire [`NUM_THREADS-1:0]           write_tmask,
    input raster_stamp_t [`NUM_THREADS-1:0] write_data,

    // Output
    VX_gpu_csr_if.slave raster_csr_if
);
    `UNUSED_PARAM (CORE_ID)

    `UNUSED_VAR (reset)

    localparam NW_WIDTH      = `UP(`NW_BITS);
    localparam NUM_CSRS_BITS = `CLOG2(`CSR_RASTER_COUNT);

    raster_csrs_t [`NUM_THREADS-1:0] wdata;
    raster_csrs_t [`NUM_THREADS-1:0] rdata;
    wire [`NUM_THREADS-1:0]          write;
    wire [NW_WIDTH-1:0]              waddr;
    wire [NW_WIDTH-1:0]              raddr;

    // CSR registers
    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        VX_dp_ram #(
            .DATAW  ($bits(raster_csrs_t)),
            .SIZE   (`NUM_WARPS),
            .LUTRAM (1)
        ) stamp_store (
            .clk   (clk),
            .write  (write[i]),            
            `UNUSED_PIN (wren),               
            .waddr (waddr),
            .wdata (wdata[i]),
            .raddr (raddr),
            .rdata (rdata[i])
        );
    end

    // CSRs write
    
    assign waddr = write_wid;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign write[i]          = write_enable && write_tmask[i];
        assign wdata[i].pos_mask = {write_data[i].pos_y, write_data[i].pos_x, write_data[i].mask};
        assign wdata[i].bcoords  = write_data[i].bcoords;
    end
    
    // CSRs read

    assign raddr = raster_csr_if.read_wid;

    wire [NUM_CSRS_BITS-1:0] csr_addr = raster_csr_if.read_addr[NUM_CSRS_BITS-1:0];

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        wire [`CSR_RASTER_COUNT-1:0][31:0] indexable_rdata = rdata[i];
        assign raster_csr_if.read_data[i] = indexable_rdata[csr_addr];
    end

    `UNUSED_VAR (write_uuid)

    `UNUSED_VAR (raster_csr_if.read_enable)
    `UNUSED_VAR (raster_csr_if.read_addr)
    `UNUSED_VAR (raster_csr_if.read_uuid)
    `UNUSED_VAR (raster_csr_if.read_tmask)

    `UNUSED_VAR (raster_csr_if.write_enable)
    `UNUSED_VAR (raster_csr_if.write_addr)
    `UNUSED_VAR (raster_csr_if.write_data)
    `UNUSED_VAR (raster_csr_if.write_uuid)
    `UNUSED_VAR (raster_csr_if.write_wid)
    `UNUSED_VAR (raster_csr_if.write_tmask)

`ifdef DBG_TRACE_RASTER
    wire [`NUM_THREADS-1:0][`RASTER_DIM_BITS-2:0] pos_x;
    wire [`NUM_THREADS-1:0][`RASTER_DIM_BITS-2:0] pos_y;
    wire [`NUM_THREADS-1:0][3:0]                  mask;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign pos_x[i] = write_data[i].pos_x;
        assign pos_y[i] = write_data[i].pos_y;
        assign mask[i]  = write_data[i].mask;
    end

    always @(posedge clk) begin
        if (raster_csr_if.read_enable) begin
            `TRACE(1, ("%d: core%0d-raster-csr-read: wid=%0d, tmask=%b, state=", $time, CORE_ID, raster_csr_if.read_wid, raster_csr_if.read_tmask));
            `TRACE_RASTER_CSR(1, raster_csr_if.read_addr);
            `TRACE(1, (", data="));
            `TRACE_ARRAY1D(1, raster_csr_if.read_data, `NUM_THREADS);
            `TRACE(1, (" (#%0d)\n", raster_csr_if.read_uuid));
        end
        if (write_enable) begin
            `TRACE(1, ("%d: core%0d-raster-fetch: wid=%0d, tmask=%b, pos_x=", $time, CORE_ID, write_wid, write_tmask));
            `TRACE_ARRAY1D(1, pos_x, `NUM_THREADS);
            `TRACE(1, (", pos_y="));
            `TRACE_ARRAY1D(1, pos_y, `NUM_THREADS);
            `TRACE(1, (", mask="));
            `TRACE_ARRAY1D(1, mask, `NUM_THREADS);
            `TRACE(1, (" (#%0d)\n", write_uuid));
        end
    end
`endif

endmodule
