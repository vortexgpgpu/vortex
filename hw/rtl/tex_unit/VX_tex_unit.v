`include "VX_platform.vh"
`include "VX_define.vh"

module VX_tex_unit #(  
    parameter CORE_ID = 0
) (
    input wire  clk,
    input wire  reset,    

    // Texture unit <-> Memory Unit
    VX_dcache_core_req_if dcache_req_if,
    VX_dcache_core_rsp_if dcache_rsp_if,

    // Inputs
    VX_tex_req_if   tex_req_if,
    VX_tex_csr_if   tex_csr_if,

    // Outputs
    VX_tex_rsp_if   tex_rsp_if
);

    localparam REQ_TAG_WIDTH = `TEX_FORMAT_BITS + `NW_BITS + 32 + `NR_BITS + 1;

    `UNUSED_PARAM (CORE_ID)
    `UNUSED_VAR (reset)

    reg [`TEX_ADDR_BITS-1:0]   tex_addr   [`NUM_TEX_UNITS-1: 0]; 
    reg [`TEX_FORMAT_BITS-1:0] tex_format [`NUM_TEX_UNITS-1: 0];
    reg [`TEX_WIDTH_BITS-1:0]  tex_width  [`NUM_TEX_UNITS-1: 0];
    reg [`TEX_HEIGHT_BITS-1:0] tex_height [`NUM_TEX_UNITS-1: 0];
    reg [`TEX_STRIDE_BITS-1:0] tex_stride [`NUM_TEX_UNITS-1: 0];
    reg [`TEX_WRAP_BITS-1:0]   tex_wrap_u [`NUM_TEX_UNITS-1: 0];
    reg [`TEX_WRAP_BITS-1:0]   tex_wrap_v [`NUM_TEX_UNITS-1: 0];
    reg [`TEX_FILTER_BITS-1:0] tex_filter [`NUM_TEX_UNITS-1: 0];

    // CSRs programming

    for (genvar i = 0; i < `NUM_TEX_UNITS; ++i) begin
        always @(posedge clk ) begin        
            if (reset) begin
                tex_addr[i]   <= 0;
                tex_format[i] <= 0;
                tex_width[i]  <= 0;
                tex_height[i] <= 0;
                tex_stride[i] <= 0;
                tex_wrap_u[i] <= 0;
                tex_wrap_v[i] <= 0;
                tex_filter[i] <= 0;
            end begin
                if (tex_csr_if.write_enable) begin            
                    case (tex_csr_if.write_addr)
                        `CSR_TEX_ADDR(i)   : tex_addr[i]   <= tex_csr_if.write_data[`TEX_ADDR_BITS-1:0];
                        `CSR_TEX_FORMAT(i) : tex_format[i] <= tex_csr_if.write_data[`TEX_FORMAT_BITS-1:0];
                        `CSR_TEX_WIDTH(i)  : tex_width[i]  <= tex_csr_if.write_data[`TEX_WIDTH_BITS-1:0];
                        `CSR_TEX_HEIGHT(i) : tex_height[i] <= tex_csr_if.write_data[`TEX_HEIGHT_BITS-1:0];
                        `CSR_TEX_STRIDE(i) : tex_stride[i] <= tex_csr_if.write_data[`TEX_STRIDE_BITS-1:0];
                        `CSR_TEX_WRAP_U(i) : tex_wrap_u[i] <= tex_csr_if.write_data[`TEX_WRAP_BITS-1:0];
                        `CSR_TEX_WRAP_V(i) : tex_wrap_v[i] <= tex_csr_if.write_data[`TEX_WRAP_BITS-1:0];
                        `CSR_TEX_FILTER(i) : tex_filter[i] <= tex_csr_if.write_data[`TEX_FILTER_BITS-1:0];
                        default:
                            assert(tex_csr_if.write_addr >= `CSR_TEX_BEGIN(0) 
                                && tex_csr_if.write_addr < `CSR_TEX_BEGIN(`CSR_TEX_STATES));
                    endcase
                end
            end
        end
    end

    // address generation

    wire mem_req_valid;
    wire [`NUM_THREADS-1:0] mem_req_tmask;
    wire [`TEX_FILTER_BITS-1:0] mem_req_filter;
    wire [`NUM_THREADS-1:0][3:0][31:0] mem_req_addr;
    wire [REQ_TAG_WIDTH-1:0] mem_req_tag;
    wire mem_req_ready;

    wire mem_rsp_valid;
    wire [`NUM_THREADS-1:0] mem_rsp_tmask;
    wire [`TEX_FILTER_BITS-1:0] mem_rsp_filter;
    wire [`NUM_THREADS-1:0][3:0][31:0] mem_rsp_data;
    wire [REQ_TAG_WIDTH-1:0] mem_rsp_tag;
    wire mem_rsp_ready;
                
    VX_tex_addr_gen #(
        .FRAC_BITS     (20),
        .REQ_TAG_WIDTH (REQ_TAG_WIDTH)
    ) tex_addr_gen (
        .clk            (clk),
        .reset          (reset),

        .valid_in       (tex_req_if.valid),
        .ready_in       (tex_req_if.ready),   

        .filter         (tex_filter[tex_req_if.unit]),
        .wrap_u         (tex_wrap_u[tex_req_if.unit]),
        .wrap_v         (tex_wrap_v[tex_req_if.unit]),
        .req_tmask      (tex_req_if.tmask),
        .req_tag        ({tex_format[tex_req_if.unit], tex_req_if.wid, tex_req_if.PC, tex_req_if.rd, tex_req_if.wb}),

        .base_addr      (tex_addr[tex_req_if.unit]),
        .log2_stride    (tex_stride[tex_req_if.unit]),
        .log2_width     (tex_width[tex_req_if.unit]),
        .log2_height    (tex_height[tex_req_if.unit]),
        
        .coord_u        (tex_req_if.u),
        .coord_v        (tex_req_if.v),
        .lod            (tex_req_if.lod),

        .mem_req_valid  (mem_req_valid),   
        .mem_req_tmask  (mem_req_tmask), 
        .mem_req_filter (mem_req_filter), 
        .mem_req_tag    (mem_req_tag),
        .mem_req_addr   (mem_req_addr),
        .mem_req_ready  (mem_req_ready)
    );

    // retrieve texel values from memory
    
    VX_tex_memory #(
        .CORE_ID       (CORE_ID),
        .REQ_TAG_WIDTH (REQ_TAG_WIDTH)
    ) tex_memory (
        .clk           (clk),
        .reset         (reset),

        // memory interface
        .dcache_req_if (dcache_req_if),
        .dcache_rsp_if (dcache_rsp_if),

        // inputs
        .req_valid (mem_req_valid),
        .req_tmask (mem_req_tmask), 
        .req_filter(mem_req_filter), 
        .req_addr  (mem_req_addr),
        .req_tag   (mem_req_tag),
        .req_ready (mem_req_ready),

        // outputs
        .rsp_valid (mem_rsp_valid),
        .rsp_tmask (mem_rsp_tmask), 
        .rsp_filter(mem_rsp_filter), 
        .rsp_data  (mem_rsp_data),
        .rsp_tag   (mem_rsp_tag),
        .rsp_ready (mem_rsp_ready)
    );

    // apply sampler

    wire [`TEX_FORMAT_BITS-1:0] rsp_format;
    wire [`NW_BITS-1:0]         rsp_wid;
    wire [31:0]                 rsp_PC;
    wire [`NR_BITS-1:0]         rsp_rd;   
    wire                        rsp_wb;
    
    assign {rsp_format, rsp_wid, rsp_PC, rsp_rd, rsp_wb} = mem_rsp_tag;

     VX_tex_sampler #(
        .CORE_ID (CORE_ID)
     ) tex_sampler (
        .clk        (clk),
        .reset      (reset),

        // inputs
        .req_valid  (mem_rsp_valid),  
        .req_tmask  (mem_rsp_tmask),         
        .req_texels (mem_rsp_data),     
        .req_filter (mem_rsp_filter),     
        .req_format (rsp_format),  
        .req_wid    (rsp_wid),        
        .req_PC     (rsp_PC),
        .req_rd     (rsp_rd),
        .req_wb     (rsp_wb), 
        .req_ready  (mem_rsp_ready),

        // outputs
        .rsp_valid  (tex_rsp_if.valid),
        .rsp_wid    (tex_rsp_if.wid),
        .rsp_tmask  (tex_rsp_if.tmask),
        .rsp_PC     (tex_rsp_if.PC),
        .rsp_rd     (tex_rsp_if.rd),
        .rsp_wb     (tex_rsp_if.wb),
        .rsp_data   (tex_rsp_if.data),
        .rsp_ready  (tex_rsp_if.ready)
    );    

`ifdef DBG_PRINT_TEX
    always @(posedge clk) begin
        if (tex_csr_if.write_enable 
         && (tex_csr_if.write_addr >= `CSR_TEX_BEGIN(0) 
          && tex_csr_if.write_addr < `CSR_TEX_BEGIN(`CSR_TEX_STATES))) begin
            $display("%t: core%0d-tex_csr: csr_tex0_addr, csr_data=%0h", $time, CORE_ID, tex_addr[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_format, csr_data=%0h", $time, CORE_ID, tex_format[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_width, csr_data=%0h", $time, CORE_ID, tex_width[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_height, csr_data=%0h", $time, CORE_ID, tex_height[0]);
            $display("%t: core%0d-tex_csr: CSR_TEX0_PITCH, csr_data=%0h", $time, CORE_ID, tex_stride[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_wrap_u, csr_data=%0h", $time, CORE_ID, tex_wrap_u[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_wrap_v, csr_data=%0h", $time, CORE_ID, tex_wrap_v[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_min_filter, csr_data=%0h", $time, CORE_ID, tex_min_filter[0]);
            $display("%t: core%0d-tex_csr: csr_tex0_max_filter, csr_data=%0h", $time, CORE_ID, tex_max_filter[0]);
        end
    end
`endif

endmodule