`include "VX_define.vh"

module VX_dcr_data (
    input wire clk,
    input wire reset,

`ifdef EXT_TEX_ENABLE
    VX_tex_dcr_if.slave         tex_dcr_if,
`endif
`ifdef EXT_RASTER_ENABLE
    VX_raster_dcr_if.slave     raster_dcr_if,
`endif
`ifdef EXT_ROP_ENABLE
    VX_rop_dcr_if.slave        rop_dcr_if,
`endif

    input  wire                             dcr_wr_valid,
    input  wire [`VX_DCR_ADDR_WIDTH-1:0]    dcr_wr_addr,
    input  wire [`VX_DCR_DATA_WIDTH-1:0]    dcr_wr_data,
    output wire                             dcr_wr_ready
);
`ifdef EXT_TEX_ENABLE
    wire is_tex_dcr = (dcr_wr_addr >= `DCR_TEX_STATE_BEGIN && dcr_wr_addr < `DCR_TEX_STATE_END);
`endif
`ifdef EXT_RASTER_ENABLE
    wire is_raster_dcr = (dcr_wr_addr >= `DCR_RASTER_STATE_BEGIN && dcr_wr_addr < `DCR_RASTER_STATE_END);
`endif
`ifdef EXT_ROP_ENABLE
    wire is_rop_dcr = (dcr_wr_addr >= `DCR_ROP_STATE_BEGIN && dcr_wr_addr < `DCR_ROP_STATE_END);
`endif

    always @(posedge clk) begin
        reg write_addr_valid;
        if (reset) begin
            //--
        end else if (dcr_wr_valid) begin
            write_addr_valid = 0;
            `ifdef EXT_TEX_ENABLE
                if (is_tex_dcr) begin
                    write_addr_valid = 1;
                end
            `endif
            `ifdef EXT_RASTER_ENABLE
                if (is_raster_dcr) begin
                    write_addr_valid = 1;
                end
            `endif
            `ifdef EXT_ROP_ENABLE
                if (is_rop_dcr) begin
                    write_addr_valid = 1;
                end
            `endif
            `ASSERT(write_addr_valid, ("%t: *** invalid device configuration register write address: 0x%0h, data=0x%0h", $time, dcr_wr_addr, dcr_wr_data));
        end
    end

    assign dcr_wr_ready = 1; // no handshaking needed

`ifdef EXT_TEX_ENABLE
    VX_tex_dcr #(
        .NUM_STAGES (`TEX_STAGE_COUNT)
    ) tex_dcr (
        .clk        (clk),
        .reset      (reset),

        .dcr_wr_valid (dcr_wr_valid && is_tex_dcr),
        .dcr_wr_addr  (dcr_wr_addr),
        .dcr_wr_data  (dcr_wr_data),

        .tex_dcr_if (tex_dcr_if)
    );
`endif

`ifdef EXT_RASTER_ENABLE
    VX_raster_dcr raster_dcr (
        .clk        (clk),
        .reset      (reset),

        .dcr_wr_valid (dcr_wr_valid && is_raster_dcr),
        .dcr_wr_addr  (dcr_wr_addr),
        .dcr_wr_data  (dcr_wr_data),

        .raster_dcr_if (raster_dcr_if)
    );
`endif

`ifdef EXT_ROP_ENABLE
    VX_rop_dcr rop_dcr (
        .clk        (clk),
        .reset      (reset),

        .dcr_wr_valid (dcr_wr_valid && is_rop_dcr),
        .dcr_wr_addr  (dcr_wr_addr),
        .dcr_wr_data  (dcr_wr_data),

        .rop_dcr_if (rop_dcr_if)
    );
`endif

endmodule
