`include "VX_define.vh"

module VX_csr_global (
    input wire clk,
    input wire reset,

`ifdef EXT_TEX_ENABLE
    VX_tex_csr_if.master        tex_csr_if,
`endif
`ifdef EXT_RASTER_ENABLE
    VX_raster_csr_if.master     raster_csr_if,
`endif
`ifdef EXT_ROP_ENABLE
    VX_rop_csr_if.master        rop_csr_if,
`endif

    input  wire                             csr_wr_valid,
    input  wire [`VX_CSR_ADDR_WIDTH-1:0]    csr_wr_addr,
    input  wire [`VX_CSR_DATA_WIDTH-1:0]    csr_wr_data,
    output wire                             csr_wr_ready
);
`ifdef EXT_TEX_ENABLE
    wire is_tex_csr = (csr_wr_addr >= `CSR_TEX_STATE_BEGIN && csr_wr_addr < `CSR_TEX_STATE_END);
`endif
`ifdef EXT_RASTER_ENABLE
    wire is_raster_csr = (csr_wr_addr >= `CSR_RASTER_STATE_BEGIN && csr_wr_addr < `CSR_RASTER_STATE_END);
`endif
`ifdef EXT_ROP_ENABLE
    wire is_rop_csr = (csr_wr_addr >= `CSR_ROP_STATE_BEGIN && csr_wr_addr < `CSR_ROP_STATE_END);
`endif

    always @(posedge clk) begin
        reg write_addr_valid;
        if (reset) begin
            //--
        end else if (csr_wr_valid) begin
            write_addr_valid = 0;
            `ifdef EXT_TEX_ENABLE
                if (is_tex_csr) begin
                    write_addr_valid = 1;
                end
            `endif
            `ifdef EXT_RASTER_ENABLE
                if (is_raster_csr) begin
                    write_addr_valid = 1;
                end
            `endif
            `ifdef EXT_ROP_ENABLE
                if (is_rop_csr) begin
                    write_addr_valid = 1;
                end
            `endif
            `ASSERT(write_addr_valid, ("%t: *** invalid global CSR write address: 0x%0h, data=0x%0h", $time, csr_wr_addr, csr_wr_data));
        end
    end

    assign csr_wr_ready = 1; // no handshaking needed

`ifdef EXT_TEX_ENABLE
    VX_tex_csr #(
        .NUM_STAGES (`TEX_STAGE_COUNT)
    ) tex_csr (
        .clk        (clk),
        .reset      (reset),

        .csr_wr_valid (csr_wr_valid && is_tex_csr),
        .csr_wr_addr  (csr_wr_addr),
        .csr_wr_data  (csr_wr_data),

        .tex_csr_if (tex_csr_if)
    );
`endif

`ifdef EXT_RASTER_ENABLE
    VX_raster_csr raster_csr (
        .clk        (clk),
        .reset      (reset),

        .csr_wr_valid (csr_wr_valid && is_raster_csr),
        .csr_wr_addr  (csr_wr_addr),
        .csr_wr_data  (csr_wr_data),

        .raster_csr_if (raster_csr_if)
    );
`endif

`ifdef EXT_ROP_ENABLE
    VX_rop_csr rop_csr (
        .clk        (clk),
        .reset      (reset),

        .csr_wr_valid (csr_wr_valid && is_rop_csr),
        .csr_wr_addr  (csr_wr_addr),
        .csr_wr_data  (csr_wr_data),

        .rop_csr_if (rop_csr_if)
    );
`endif

endmodule