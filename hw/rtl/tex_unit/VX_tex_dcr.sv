`include "VX_tex_define.vh"

module VX_tex_dcr #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_STAGES = 1
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_dcr_write_if.slave               dcr_write_if,

    // Output
    input wire [`TEX_STAGE_BITS-1:0]    stage,
    output tex_dcrs_t                   tex_dcrs
);
    `UNUSED_SPARAM (INSTANCE_ID)
    
    `UNUSED_VAR (reset)

    // DCR registers

    reg [$clog2(NUM_STAGES)-1:0] dcr_stage;
    tex_dcrs_t dcrs [NUM_STAGES-1:0];

    // DCRs write

    always @(posedge clk) begin
        if (dcr_write_if.valid) begin
            case (dcr_write_if.addr)
                `DCR_TEX_STAGE: begin 
                    dcr_stage <= dcr_write_if.data[$clog2(NUM_STAGES)-1:0];
                end
                `DCR_TEX_ADDR: begin 
                    dcrs[dcr_stage].baseaddr <= dcr_write_if.data[`TEX_ADDR_BITS-1:0];
                end
                `DCR_TEX_FORMAT: begin 
                    dcrs[dcr_stage].format <= dcr_write_if.data[`TEX_FORMAT_BITS-1:0];
                end
                `DCR_TEX_FILTER: begin 
                    dcrs[dcr_stage].filter <= dcr_write_if.data[`TEX_FILTER_BITS-1:0];
                end
                `DCR_TEX_WRAP: begin
                    dcrs[dcr_stage].wraps[0] <= dcr_write_if.data[0  +: `TEX_WRAP_BITS];
                    dcrs[dcr_stage].wraps[1] <= dcr_write_if.data[16 +: `TEX_WRAP_BITS];
                end
                `DCR_TEX_LOGDIM: begin 
                    dcrs[dcr_stage].logdims[0] <= dcr_write_if.data[0  +: `TEX_LOD_BITS];
                    dcrs[dcr_stage].logdims[1] <= dcr_write_if.data[16 +: `TEX_LOD_BITS];
                end
                default: begin
                    for (integer j = 0; j <= `TEX_LOD_MAX; ++j) begin
                    `IGNORE_WARNINGS_BEGIN
                        if (dcr_write_if.addr == `DCR_TEX_MIPOFF(j)) begin
                    `IGNORE_WARNINGS_END
                            dcrs[dcr_stage].mipoff[j] <= dcr_write_if.data[`TEX_MIPOFF_BITS-1:0];
                        end
                    end
                end
            endcase
        end
    end

    // DCRs read
    assign tex_dcrs = dcrs[stage];

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (dcr_write_if.valid) begin
            `TRACE(1, ("%d: %s-tex-dcr: stage=%0d, state=", $time, INSTANCE_ID, dcr_stage));
            `TRACE_TEX_DCR(1, dcr_write_if.addr);
            `TRACE(1, (", data=0x%0h\n", dcr_write_if.data));
        end
    end
`endif

endmodule
