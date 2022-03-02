`include "VX_tex_define.vh"

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
    VX_tex_dcr_if.master tex_dcr_if
);

    `UNUSED_VAR (reset)

    // DCR registers

    reg [$clog2(NUM_STAGES)-1:0] dcr_tex_stage;
    reg [(`TEX_LOD_MAX+1)-1:0][`TEX_MIPOFF_BITS-1:0] tex_mipoff [NUM_STAGES-1:0];
    reg [1:0][`TEX_LOD_BITS-1:0]  tex_logdims [NUM_STAGES-1:0];
    reg [1:0][`TEX_WRAP_BITS-1:0] tex_wraps  [NUM_STAGES-1:0];
    reg [`TEX_ADDR_BITS-1:0]      tex_baddr  [NUM_STAGES-1:0];     
    reg [`TEX_FORMAT_BITS-1:0]    tex_format [NUM_STAGES-1:0];
    reg [`TEX_FILTER_BITS-1:0]    tex_filter [NUM_STAGES-1:0];

    // DCRs write

    always @(posedge clk) begin
        if (reset) begin
            dcr_tex_stage <= 0;
            for (integer  i = 0; i < NUM_STAGES; ++i) begin
                tex_mipoff[i]  <= 0;
                tex_logdims[i] <= 0;
                tex_wraps[i]   <= 0;
                tex_baddr[i]   <= 0;
                tex_format[i]  <= 0;
                tex_filter[i]  <= 0;
            end
        end else if (dcr_wr_valid) begin
            case (dcr_wr_addr)
                `DCR_TEX_STAGE: begin 
                    dcr_tex_stage <= dcr_wr_data[$clog2(NUM_STAGES)-1:0];
                end
                `DCR_TEX_ADDR: begin 
                    tex_baddr[dcr_tex_stage] <= dcr_wr_data[`TEX_ADDR_BITS-1:0];
                end
                `DCR_TEX_FORMAT: begin 
                    tex_format[dcr_tex_stage] <= dcr_wr_data[`TEX_FORMAT_BITS-1:0];
                end
                `DCR_TEX_FILTER: begin 
                    tex_filter[dcr_tex_stage] <= dcr_wr_data[`TEX_FILTER_BITS-1:0];
                end
                `DCR_TEX_WRAP: begin
                    tex_wraps[dcr_tex_stage][0] <= dcr_wr_data[15:0][`TEX_WRAP_BITS-1:0];
                    tex_wraps[dcr_tex_stage][1] <= dcr_wr_data[31:16][`TEX_WRAP_BITS-1:0];
                end
                `DCR_TEX_LOGDIM: begin 
                    tex_logdims[dcr_tex_stage][0] <= dcr_wr_data[15:0][`TEX_LOD_BITS-1:0];
                    tex_logdims[dcr_tex_stage][1] <= dcr_wr_data[31:16][`TEX_LOD_BITS-1:0];
                end
                default: begin
                    for (integer j = 0; j <= `TEX_LOD_MAX; ++j) begin
                    `IGNORE_WARNINGS_BEGIN
                        if (dcr_wr_addr == `DCR_TEX_MIPOFF(j)) begin
                    `IGNORE_WARNINGS_END
                            tex_mipoff[dcr_tex_stage][j] <= dcr_wr_data[`TEX_MIPOFF_BITS-1:0];
                        end
                    end
                end
            endcase
        end
    end

    // DCRs read
    assign tex_dcr_if.data.mipoff  = tex_mipoff[tex_dcr_if.stage];
    assign tex_dcr_if.data.logdims = tex_logdims[tex_dcr_if.stage];
    assign tex_dcr_if.data.wraps   = tex_wraps[tex_dcr_if.stage];
    assign tex_dcr_if.data.baddr   = tex_baddr[tex_dcr_if.stage];
    assign tex_dcr_if.data.format  = tex_format[tex_dcr_if.stage];
    assign tex_dcr_if.data.filter  = tex_filter[tex_dcr_if.stage];

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (dcr_wr_valid) begin
            dpi_trace("%d: tex-dcr: stage=%0d, state=", $time, dcr_tex_stage);
            trace_tex_state(dcr_wr_addr);
            dpi_trace(", data=0x%0h\n", dcr_wr_data);
        end
    end
`endif

endmodule