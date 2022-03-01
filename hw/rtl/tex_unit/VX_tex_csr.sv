`include "VX_tex_define.vh"

module VX_tex_csr #(
    parameter NUM_STAGES = 1
) (
    input wire clk,
    input wire reset,

    // Inputs
    input  wire                             csr_wr_valid,
    input  wire [`VX_CSR_ADDR_WIDTH-1:0]    csr_wr_addr,
    input  wire [`VX_CSR_DATA_WIDTH-1:0]    csr_wr_data,

    // Output
    VX_tex_csr_if.master tex_csr_if
);

    `UNUSED_VAR (reset)

    // CSR registers

    reg [$clog2(NUM_STAGES)-1:0] csr_tex_stage;
    reg [(`TEX_LOD_MAX+1)-1:0][`TEX_MIPOFF_BITS-1:0] tex_mipoff [NUM_STAGES-1:0];
    reg [1:0][`TEX_LOD_BITS-1:0]  tex_logdims [NUM_STAGES-1:0];
    reg [1:0][`TEX_WRAP_BITS-1:0] tex_wraps  [NUM_STAGES-1:0];
    reg [`TEX_ADDR_BITS-1:0]      tex_baddr  [NUM_STAGES-1:0];     
    reg [`TEX_FORMAT_BITS-1:0]    tex_format [NUM_STAGES-1:0];
    reg [`TEX_FILTER_BITS-1:0]    tex_filter [NUM_STAGES-1:0];

    // CSRs write

    always @(posedge clk) begin
        if (reset) begin
            csr_tex_stage <= 0;
            for (integer  i = 0; i < NUM_STAGES; ++i) begin
                tex_mipoff[i]  <= 0;
                tex_logdims[i] <= 0;
                tex_wraps[i]   <= 0;
                tex_baddr[i]   <= 0;
                tex_format[i]  <= 0;
                tex_filter[i]  <= 0;
            end
        end else if (csr_wr_valid) begin
            case (csr_wr_addr)
                `CSR_TEX_STAGE: begin 
                    csr_tex_stage <= csr_wr_data[$clog2(NUM_STAGES)-1:0];
                end
                `CSR_TEX_ADDR: begin 
                    tex_baddr[csr_tex_stage] <= csr_wr_data[`TEX_ADDR_BITS-1:0];
                end
                `CSR_TEX_FORMAT: begin 
                    tex_format[csr_tex_stage] <= csr_wr_data[`TEX_FORMAT_BITS-1:0];
                end
                `CSR_TEX_FILTER: begin 
                    tex_filter[csr_tex_stage] <= csr_wr_data[`TEX_FILTER_BITS-1:0];
                end
                `CSR_TEX_WRAP: begin
                    tex_wraps[csr_tex_stage][0] <= csr_wr_data[15:0][`TEX_WRAP_BITS-1:0];
                    tex_wraps[csr_tex_stage][1] <= csr_wr_data[31:16][`TEX_WRAP_BITS-1:0];
                end
                `CSR_TEX_LOGDIM: begin 
                    tex_logdims[csr_tex_stage][0] <= csr_wr_data[15:0][`TEX_LOD_BITS-1:0];
                    tex_logdims[csr_tex_stage][1] <= csr_wr_data[31:16][`TEX_LOD_BITS-1:0];
                end
                default: begin
                    for (integer j = 0; j <= `TEX_LOD_MAX; ++j) begin
                    `IGNORE_WARNINGS_BEGIN
                        if (csr_wr_addr == `CSR_TEX_MIPOFF(j)) begin
                    `IGNORE_WARNINGS_END
                            tex_mipoff[csr_tex_stage][j] <= csr_wr_data[`TEX_MIPOFF_BITS-1:0];
                        end
                    end
                end
            endcase
        end
    end

    // CSRs read
    assign tex_csr_if.data.mipoff  = tex_mipoff[tex_csr_if.stage];
    assign tex_csr_if.data.logdims = tex_logdims[tex_csr_if.stage];
    assign tex_csr_if.data.wraps   = tex_wraps[tex_csr_if.stage];
    assign tex_csr_if.data.baddr   = tex_baddr[tex_csr_if.stage];
    assign tex_csr_if.data.format  = tex_format[tex_csr_if.stage];
    assign tex_csr_if.data.filter  = tex_filter[tex_csr_if.stage];

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (csr_wr_valid) begin
            dpi_trace("%d: tex-csr: stage=%0d, state=", $time, csr_tex_stage);
            trace_tex_state(csr_wr_addr);
            dpi_trace(", data=0x%0h\n", csr_wr_data);
        end
    end
`endif

endmodule