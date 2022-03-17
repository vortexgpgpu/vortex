`include "VX_raster_define.vh"

module VX_raster_csr #( 
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_gpu_csr_if.slave raster_csr_if,
    VX_raster_req_if.slave raster_req_if,

    // Output
    VX_raster_to_rop_if.slave raster_to_rop_if
);
    `UNUSED_VAR (reset)

    raster_csrs_t [`NUM_THREADS-1:0] wdata = 0;
    raster_csrs_t [`NUM_THREADS-1:0] rdata;
    wire [`NUM_THREADS-1:0] wren = 0;
    wire [`NW_BITS-1:0] waddr = 0;
    wire [`NW_BITS-1:0] raddr = 0;

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

    `UNUSED_VAR (rdata)

    `UNUSED_VAR (raster_req_if.valid)
    `UNUSED_VAR (raster_req_if.tmask)
    `UNUSED_VAR (raster_req_if.stamp)
    `UNUSED_VAR (raster_req_if.empty)
    `UNUSED_VAR (raster_req_if.ready)

    `UNUSED_VAR (raster_to_rop_if.valid)
    `UNUSED_VAR (raster_to_rop_if.wid)
    assign raster_to_rop_if.pos_x = 0;
    assign raster_to_rop_if.pos_y = 0;
    assign raster_to_rop_if.mask  = 0;
    assign raster_to_rop_if.ready = 0;
    
    /*
    // CSRs write
    always @(posedge clk) begin
        if (reset) begin
            reg_csrs <= 0;
        end else if (raster_csr_if.write_enable) begin
            case (raster_csr_if.write_addr)
                `CSR_RASTER_FRAG:;
                `CSR_RASTER_X_Y:;
                `CSR_RASTER_MASK_PID:;
                `CSR_RASTER_BCOORD_X:;
                `CSR_RASTER_BCOORD_Y:;
                `CSR_RASTER_BCOORD_Z:;
                `CSR_RASTER_GRAD_X:;
                `CSR_RASTER_GRAD_Y:;
                default:;
            endcase
        end
    end

    // CSRs read

    reg [31:0] read_data_r;
    always @(*) begin
        read_data_r = 'x;
        case (raster_csr_if.read_addr)
            `CSR_RASTER_FRAG:;
            `CSR_RASTER_X_Y:;
            `CSR_RASTER_MASK_PID:;
            `CSR_RASTER_BCOORD_X:;
            `CSR_RASTER_BCOORD_Y:;
            `CSR_RASTER_BCOORD_Z:;
            `CSR_RASTER_GRAD_X:;
            `CSR_RASTER_GRAD_Y:;
            default:;
        endcase
    end

    assign raster_csr_if.read_data = {`NUM_THREADS{read_data_r}};

    assign raster_csrs = reg_csrs;

    `UNUSED_VAR (raster_csr_if.read_enable)
    `UNUSED_VAR (raster_csr_if.read_uuid)
    `UNUSED_VAR (raster_csr_if.read_wid)
    `UNUSED_VAR (raster_csr_if.read_tmask)
    `UNUSED_VAR (raster_csr_if.write_uuid)
    `UNUSED_VAR (raster_csr_if.write_wid)
    `UNUSED_VAR (raster_csr_if.write_tmask)*/

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (raster_csr_if.read_enable) begin
            dpi_trace("%d: core%0d-raster-csr-read: wid=%0d, tmask=%b, state=", $time, CORE_ID, raster_csr_if.read_wid, raster_csr_if.read_tmask);
            trace_raster_csr(raster_csr_if.read_addr);
            dpi_trace(", data=", raster_csr_if.read_data);
            `TRACE_ARRAY1D(raster_csr_if.read_data, `NUM_THREADS);
            dpi_trace(" (#%0d)\n", raster_csr_if.read_uuid);
        end
        if (raster_csr_if.write_enable) begin
            dpi_trace("%d: core%0d-raster-csr-write: wid=%0d, tmask=%b, state=", $time, CORE_ID, raster_csr_if.write_wid, raster_csr_if.write_tmask);
            trace_raster_csr(raster_csr_if.write_addr);
            dpi_trace(", data=", raster_csr_if.write_data);
            `TRACE_ARRAY1D(raster_csr_if.write_data, `NUM_THREADS);
            dpi_trace(" (#%0d)\n", raster_csr_if.write_uuid);
        end
    end
`endif

endmodule