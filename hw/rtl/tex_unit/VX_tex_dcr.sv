`include "VX_tex_define.vh"

import VX_tex_types::*;

module VX_tex_dcr #(
    parameter NUM_STAGES = 1
) (
    input wire clk,
    input wire reset,

    // Inputs
    input  wire                             dcr_wr_valid,
    input  wire [`VX_DCR_ADDR_WIDTH-1:0]    dcr_wr_addr,
    input  wire [`VX_DCR_DATA_WIDTH-1:0]    dcr_wr_data,

    // Output
    VX_tex_dcr_if.master    tex_dcr_if
);

    `UNUSED_VAR (reset)

    // DCR registers

    reg [$clog2(NUM_STAGES)-1:0] tex_stage;
    tex_dcrs_t tex_dcrs [NUM_STAGES-1:0];

    // DCRs write

    always @(posedge clk) begin
        if (reset) begin
            tex_stage <= 0;
            for (integer  i = 0; i < NUM_STAGES; ++i) begin
                tex_dcrs[i].mipoff  <= 0;
                tex_dcrs[i].logdims <= 0;
                tex_dcrs[i].wraps   <= 0;
                tex_dcrs[i].baddr   <= 0;
                tex_dcrs[i].format  <= 0;
                tex_dcrs[i].filter  <= 0;
            end
        end else if (dcr_wr_valid) begin
            case (dcr_wr_addr)
                `DCR_TEX_STAGE: begin 
                    tex_stage <= dcr_wr_data[$clog2(NUM_STAGES)-1:0];
                end
                `DCR_TEX_ADDR: begin 
                    tex_dcrs[tex_stage].baddr <= dcr_wr_data[`TEX_ADDR_BITS-1:0];
                end
                `DCR_TEX_FORMAT: begin 
                    tex_dcrs[tex_stage].format <= dcr_wr_data[`TEX_FORMAT_BITS-1:0];
                end
                `DCR_TEX_FILTER: begin 
                    tex_dcrs[tex_stage].filter <= dcr_wr_data[`TEX_FILTER_BITS-1:0];
                end
                `DCR_TEX_WRAP: begin
                    tex_dcrs[tex_stage].wraps[0] <= dcr_wr_data[0  +: `TEX_WRAP_BITS];
                    tex_dcrs[tex_stage].wraps[1] <= dcr_wr_data[16 +: `TEX_WRAP_BITS];
                end
                `DCR_TEX_LOGDIM: begin 
                    tex_dcrs[tex_stage].logdims[0] <= dcr_wr_data[0  +: `TEX_LOD_BITS];
                    tex_dcrs[tex_stage].logdims[1] <= dcr_wr_data[16 +: `TEX_LOD_BITS];
                end
                default: begin
                    for (integer j = 0; j <= `TEX_LOD_MAX; ++j) begin
                    `IGNORE_WARNINGS_BEGIN
                        if (dcr_wr_addr == `DCR_TEX_MIPOFF(j)) begin
                    `IGNORE_WARNINGS_END
                            tex_dcrs[tex_stage].mipoff[j] <= dcr_wr_data[`TEX_MIPOFF_BITS-1:0];
                        end
                    end
                end
            endcase
        end
    end

    // DCRs read
    assign tex_dcr_if.data = tex_dcrs;

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (dcr_wr_valid) begin
            dpi_trace(1, "%d: tex-dcr: stage=%0d, state=", $time, tex_stage);
            trace_tex_dcr(1, dcr_wr_addr);
            dpi_trace(1, ", data=0x%0h\n", dcr_wr_data);
        end
    end
`endif

endmodule
