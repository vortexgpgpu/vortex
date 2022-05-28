`include "VX_define.vh"
`include "VX_gpu_types.vh"
`include "VX_trace_info.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
`IGNORE_WARNINGS_END


module VX_dcr_data (
    input wire clk,
    input wire reset,

    VX_dcr_base_if.master   dcr_base_if,

`ifdef EXT_TEX_ENABLE
    VX_tex_dcr_if.master    tex_dcr_if,
`endif
`ifdef EXT_RASTER_ENABLE
    VX_raster_dcr_if.master raster_dcr_if,
`endif
`ifdef EXT_ROP_ENABLE
    VX_rop_dcr_if.master    rop_dcr_if,
`endif

    input  wire                             dcr_wr_valid,
    input  wire [`VX_DCR_ADDR_WIDTH-1:0]    dcr_wr_addr,
    input  wire [`VX_DCR_DATA_WIDTH-1:0]    dcr_wr_data,
    output wire                             dcr_wr_ready
);
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

    wire is_base_dcr = (dcr_wr_addr >= `DCR_BASE_STATE_BEGIN && dcr_wr_addr < `DCR_BASE_STATE_END);
    
`ifdef EXT_TEX_ENABLE
    wire is_tex_dcr = (dcr_wr_addr >= `DCR_TEX_STATE_BEGIN && dcr_wr_addr < `DCR_TEX_STATE_END);
`endif
`ifdef EXT_RASTER_ENABLE
    wire is_raster_dcr = (dcr_wr_addr >= `DCR_RASTER_STATE_BEGIN && dcr_wr_addr < `DCR_RASTER_STATE_END);
`endif
`ifdef EXT_ROP_ENABLE
    wire is_rop_dcr = (dcr_wr_addr >= `DCR_ROP_STATE_BEGIN && dcr_wr_addr < `DCR_ROP_STATE_END);
`endif

    reg dcr_addr_valid;
    always @(*) begin
        dcr_addr_valid = is_base_dcr;
    `ifdef EXT_TEX_ENABLE
        if (is_tex_dcr) begin
            dcr_addr_valid = 1;
        end
    `endif
    `ifdef EXT_RASTER_ENABLE
        if (is_raster_dcr) begin
            dcr_addr_valid = 1;
        end
    `endif
    `ifdef EXT_ROP_ENABLE
        if (is_rop_dcr) begin
            dcr_addr_valid = 1;
        end
    `endif
    end

    `RUNTIME_ASSERT(~dcr_wr_valid || dcr_addr_valid, ("%t: *** invalid device configuration register write address: 0x%0h, data=0x%0h", $time, dcr_wr_addr, dcr_wr_data));

    assign dcr_wr_ready = 1; // no handshaking needed

    ///////////////////////////////////////////////////////////////////////////

    base_dcrs_t base_dcrs;

    always @(posedge clk) begin
        if (reset) begin
            base_dcrs <= '0;
            base_dcrs.startup_addr <= `STARTUP_ADDR;
        end else if (dcr_wr_valid) begin
            case (dcr_wr_addr)
            `DCR_STARTUP_ADDR : base_dcrs.startup_addr <= dcr_wr_data[`XLEN-1:0];
            `DCR_MPM_CLASS    : base_dcrs.mpm_class    <= dcr_wr_data[7:0];
            default:;
            endcase
        end
    end

    assign dcr_base_if.data = base_dcrs;

    ///////////////////////////////////////////////////////////////////////////

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

`ifdef DBG_TRACE_CORE_PIPELINE
    always @(posedge clk) begin
        if (dcr_wr_valid && is_base_dcr) begin
            `TRACE(1, ("%d: base-dcr: state=", $time));
            trace_base_dcr(1, dcr_wr_addr);
            `TRACE(1, (", data=0x%0h\n", dcr_wr_data));
        end
    end
`endif

endmodule
