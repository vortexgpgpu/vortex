`include "VX_tex_define.vh"

module VX_tex_csr #(  
    parameter CORE_ID = 0,
    parameter NUM_STAGES = 1
) (
    input wire clk,
    input wire reset,

    // Inputs
    VX_tex_csr_if.slave tex_csr_if,
    VX_tex_req_if.slave tex_req_if,

    // Output
    output tex_csrs_t tex_csrs
);

    `UNUSED_VAR (reset)

    // CSR registers

    reg [$clog2(NUM_STAGES)-1:0] csr_tex_unit;
    reg [(`TEX_LOD_MAX+1)-1:0][`TEX_MIPOFF_BITS-1:0] tex_mipoff [NUM_STAGES-1:0];
    reg [1:0][`TEX_LOD_BITS-1:0]  tex_logdims [NUM_STAGES-1:0];
    reg [1:0][`TEX_WRAP_BITS-1:0] tex_wraps  [NUM_STAGES-1:0];
    reg [`TEX_ADDR_BITS-1:0]      tex_baddr  [NUM_STAGES-1:0];     
    reg [`TEX_FORMAT_BITS-1:0]    tex_format [NUM_STAGES-1:0];
    reg [`TEX_FILTER_BITS-1:0]    tex_filter [NUM_STAGES-1:0];

    // CSRs write

    always @(posedge clk) begin
        if (reset) begin
            csr_tex_unit <= 0;
            for (integer  i = 0; i < NUM_STAGES; ++i) begin
                tex_mipoff[i]  <= 0;
                tex_logdims[i] <= 0;
                tex_wraps[i]   <= 0;
                tex_baddr[i]   <= 0;
                tex_format[i]  <= 0;
                tex_filter[i]  <= 0;
            end
        end else if (tex_csr_if.write_enable) begin
            case (tex_csr_if.write_addr)
                `CSR_TEX_UNIT: begin 
                    csr_tex_unit <= tex_csr_if.write_data[$clog2(NUM_STAGES)-1:0];
                end
                `CSR_TEX_ADDR: begin 
                    tex_baddr[csr_tex_unit] <= tex_csr_if.write_data[`TEX_ADDR_BITS-1:0];
                end
                `CSR_TEX_FORMAT: begin 
                    tex_format[csr_tex_unit] <= tex_csr_if.write_data[`TEX_FORMAT_BITS-1:0];
                end
                `CSR_TEX_WRAPU: begin
                    tex_wraps[csr_tex_unit][0] <= tex_csr_if.write_data[`TEX_WRAP_BITS-1:0];
                end
                `CSR_TEX_WRAPV: begin
                    tex_wraps[csr_tex_unit][1] <= tex_csr_if.write_data[`TEX_WRAP_BITS-1:0];
                end
                `CSR_TEX_FILTER: begin 
                    tex_filter[csr_tex_unit] <= tex_csr_if.write_data[`TEX_FILTER_BITS-1:0];
                end
                `CSR_TEX_WIDTH: begin 
                    tex_logdims[csr_tex_unit][0] <= tex_csr_if.write_data[`TEX_LOD_BITS-1:0];
                end
                `CSR_TEX_HEIGHT: begin 
                    tex_logdims[csr_tex_unit][1] <= tex_csr_if.write_data[`TEX_LOD_BITS-1:0];
                end
                default: begin
                    for (integer j = 0; j <= `TEX_LOD_MAX; ++j) begin
                    `IGNORE_WARNINGS_BEGIN
                        if (tex_csr_if.write_addr == `CSR_TEX_MIPOFF(j)) begin
                    `IGNORE_WARNINGS_END
                            tex_mipoff[csr_tex_unit][j] <= tex_csr_if.write_data[`TEX_MIPOFF_BITS-1:0];
                        end
                    end
                end
            endcase
        end
    end

    // CSRs read
    assign tex_csrs.mipoff  = tex_mipoff[tex_req_if.unit];
    assign tex_csrs.logdims = tex_logdims[tex_req_if.unit];
    assign tex_csrs.wraps   = tex_wraps[tex_req_if.unit];
    assign tex_csrs.baddr   = tex_baddr[tex_req_if.unit];
    assign tex_csrs.format  = tex_format[tex_req_if.unit];
    assign tex_csrs.filter  = tex_filter[tex_req_if.unit];

endmodule