`include "VX_tex_define.vh"

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

    localparam REQ_INFO_WIDTH_A = `TEX_FORMAT_BITS + `NR_BITS + 1;
    localparam REQ_INFO_WIDTH_M = (2 * `NUM_THREADS * `BLEND_FRAC) + REQ_INFO_WIDTH_A;

    `UNUSED_PARAM (CORE_ID)
    `UNUSED_VAR (reset)

    reg [`TEX_MIPOFF_BITS-1:0] tex_mipoff [`NUM_TEX_UNITS-1:0][(1 << `TEX_LOD_BITS)-1:0];
    reg [`TEX_WIDTH_BITS-1:0]  tex_width  [`NUM_TEX_UNITS-1:0][(1 << `TEX_LOD_BITS)-1:0];
    reg [`TEX_HEIGHT_BITS-1:0] tex_height [`NUM_TEX_UNITS-1:0][(1 << `TEX_LOD_BITS)-1:0];

    reg [`TEX_ADDR_BITS-1:0]   tex_baddr  [`NUM_TEX_UNITS-1:0];     
    reg [`TEX_FORMAT_BITS-1:0] tex_format [`NUM_TEX_UNITS-1:0];
    reg [`TEX_WRAP_BITS-1:0]   tex_wrap_u [`NUM_TEX_UNITS-1:0];
    reg [`TEX_WRAP_BITS-1:0]   tex_wrap_v [`NUM_TEX_UNITS-1:0];
    reg [`TEX_FILTER_BITS-1:0] tex_filter [`NUM_TEX_UNITS-1:0];

    // CSRs programming    

    for (genvar i = 0; i < `NUM_TEX_UNITS; ++i) begin
        wire [`TEX_LOD_BITS-1:0] mip_level = tex_csr_if.write_data[28 +: `TEX_LOD_BITS];   
        always @(posedge clk) begin                    
            if (tex_csr_if.write_enable) begin            
                case (tex_csr_if.write_addr)
                    `CSR_TEX_ADDR(i) : begin 
                        tex_baddr[i]  <= tex_csr_if.write_data[`TEX_ADDR_BITS-1:0];
                    end
                    `CSR_TEX_FORMAT(i) : begin 
                        tex_format[i] <= tex_csr_if.write_data[`TEX_FORMAT_BITS-1:0];
                    end
                    `CSR_TEX_WRAP(i) : begin
                        tex_wrap_u[i] <= tex_csr_if.write_data[0 +: `TEX_WRAP_BITS];
                        tex_wrap_v[i] <= tex_csr_if.write_data[`TEX_WRAP_BITS +: `TEX_WRAP_BITS];
                    end
                    `CSR_TEX_FILTER(i) : begin 
                        tex_filter[i] <= tex_csr_if.write_data[`TEX_FILTER_BITS-1:0];                        
                    end
                    `CSR_TEX_MIPOFF(i) : begin 
                        tex_mipoff[i][mip_level] <= tex_csr_if.write_data[`TEX_MIPOFF_BITS-1:0];
                    end
                    `CSR_TEX_WIDTH(i) : begin 
                        tex_width[i][mip_level]  <= tex_csr_if.write_data[`TEX_WIDTH_BITS-1:0];
                    end
                    `CSR_TEX_HEIGHT(i) : begin 
                        tex_height[i][mip_level] <= tex_csr_if.write_data[`TEX_HEIGHT_BITS-1:0];
                    end
                    default:
                        assert(tex_csr_if.write_addr >= `CSR_TEX_BEGIN(0) 
                            && tex_csr_if.write_addr < `CSR_TEX_BEGIN(`CSR_TEX_STATES));
                endcase
            end
        end
    end

    // mipmap attributes

    wire [`NUM_THREADS-1:0][`TEX_MIPOFF_BITS-1:0] tex_mipoffs;    
    wire [`NUM_THREADS-1:0][`TEX_WIDTH_BITS-1:0]  tex_widths;
    wire [`NUM_THREADS-1:0][`TEX_HEIGHT_BITS-1:0] tex_heights;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        wire [`NTEX_BITS-1:0] unit = tex_req_if.unit[`NTEX_BITS-1:0];
        wire [`TEX_LOD_BITS-1:0] mip_level = tex_req_if.lod[i][20+:`TEX_LOD_BITS];        
        assign tex_mipoffs[i] = tex_mipoff[unit][mip_level];
        assign tex_widths[i]  = tex_width[unit][mip_level];
        assign tex_heights[i] = tex_height[unit][mip_level];
    end

    // address generation

    wire mem_req_valid;
    wire [`NW_BITS-1:0]     mem_req_wid;
    wire [`NUM_THREADS-1:0] mem_req_tmask;
    wire [31:0]             mem_req_PC;
    wire [`TEX_FILTER_BITS-1:0] mem_req_filter;
    wire [`TEX_STRIDE_BITS-1:0] mem_req_stride;
    wire [`NUM_THREADS-1:0][`BLEND_FRAC-1:0] mem_req_blend_u;
    wire [`NUM_THREADS-1:0][`BLEND_FRAC-1:0] mem_req_blend_v;
    wire [`NUM_THREADS-1:0][3:0][31:0] mem_req_addr;
    wire [REQ_INFO_WIDTH_A-1:0] mem_req_info;
    wire mem_req_ready;

    wire mem_rsp_valid;
    wire [`NW_BITS-1:0]     mem_rsp_wid;
    wire [`NUM_THREADS-1:0] mem_rsp_tmask;
    wire [31:0]             mem_rsp_PC;
    wire [`TEX_FILTER_BITS-1:0] mem_rsp_filter;    
    wire [`NUM_THREADS-1:0][3:0][31:0] mem_rsp_data;
    wire [REQ_INFO_WIDTH_M-1:0] mem_rsp_info;
    wire mem_rsp_ready;        
                
    VX_tex_addr #(
        .REQ_INFO_WIDTH (REQ_INFO_WIDTH_A)
    ) tex_addr (
        .clk        (clk),
        .reset      (reset),

        .valid_in   (tex_req_if.valid),
        .ready_in   (tex_req_if.ready),   

        .req_wid    (tex_req_if.wid), 
        .req_tmask  (tex_req_if.tmask),
        .req_PC     (tex_req_if.PC), 
        .req_info   ({tex_format[tex_req_if.unit], tex_req_if.rd, tex_req_if.wb}),

        .format     (tex_format[tex_req_if.unit]),
        .filter     (tex_filter[tex_req_if.unit]),
        .wrap_u     (tex_wrap_u[tex_req_if.unit]),
        .wrap_v     (tex_wrap_v[tex_req_if.unit]),        
        
        .base_addr  (tex_baddr[tex_req_if.unit]),    
        .mip_offsets(tex_mipoffs),
        .log_widths (tex_widths),
        .log_heights(tex_heights),
        
        .coord_u    (tex_req_if.u),
        .coord_v    (tex_req_if.v),

        .rsp_valid  (mem_req_valid), 
        .rsp_wid    (mem_req_wid),   
        .rsp_tmask  (mem_req_tmask),         
        .rsp_PC     (mem_req_PC), 
        .rsp_filter (mem_req_filter), 
        .rsp_stride (mem_req_stride),
        .rsp_addr   (mem_req_addr),
        .rsp_blend_u(mem_req_blend_u),
        .rsp_blend_v(mem_req_blend_v),
        .rsp_info   (mem_req_info),
        .rsp_ready  (mem_req_ready)
    );

    // retrieve texel values from memory
    VX_tex_memory #(
        .CORE_ID        (CORE_ID),
        .REQ_INFO_WIDTH (REQ_INFO_WIDTH_M)
    ) tex_memory (
        .clk           (clk),
        .reset         (reset),

        // memory interface
        .dcache_req_if (dcache_req_if),
        .dcache_rsp_if (dcache_rsp_if),

        // inputs
        .req_valid (mem_req_valid),
        .req_wid   (mem_req_wid), 
        .req_tmask (mem_req_tmask), 
        .req_PC    (mem_req_PC), 
        .req_filter(mem_req_filter), 
        .req_stride(mem_req_stride),
        .req_addr  (mem_req_addr),
        .req_info  ({mem_req_blend_u, mem_req_blend_v, mem_req_info}),
        .req_ready (mem_req_ready),

        // outputs
        .rsp_valid (mem_rsp_valid),
        .rsp_wid   (mem_rsp_wid), 
        .rsp_tmask (mem_rsp_tmask), 
        .rsp_PC    (mem_rsp_PC),         
        .rsp_filter(mem_rsp_filter), 
        .rsp_data  (mem_rsp_data),
        .rsp_info  (mem_rsp_info),
        .rsp_ready (mem_rsp_ready)
    );

    // apply sampler

    wire [`NUM_THREADS-1:0][`BLEND_FRAC-1:0] rsp_blend_u, rsp_blend_v;
    wire [`TEX_FORMAT_BITS-1:0] rsp_format;
    wire [`NR_BITS-1:0]         rsp_rd;   
    wire                        rsp_wb;
    
    assign {rsp_blend_u, rsp_blend_v, rsp_format, rsp_rd, rsp_wb} = mem_rsp_info;

    VX_tex_sampler #(
        .CORE_ID (CORE_ID)
    ) tex_sampler (
        .clk        (clk),
        .reset      (reset),

        // inputs
        .req_valid  (mem_rsp_valid),  
        .req_wid    (mem_rsp_wid), 
        .req_tmask  (mem_rsp_tmask),         
        .req_PC     (mem_rsp_PC),
        .req_data   (mem_rsp_data),     
        .req_filter (mem_rsp_filter),     
        .req_format (rsp_format),  
        .req_blend_u(rsp_blend_u),
        .req_blend_v(rsp_blend_v),               
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
    for (genvar i = 0; i < `NUM_TEX_UNITS; ++i) begin
        always @(posedge clk) begin
            if (tex_csr_if.write_enable 
             && (tex_csr_if.write_addr >= `CSR_TEX_BEGIN(i) 
              && tex_csr_if.write_addr < `CSR_TEX_BEGIN(i+1))) begin
                $display("%t: core%0d-tex-csr: tex%0d_addr=%0h", $time, CORE_ID, i, tex_baddr[i]);
                $display("%t: core%0d-tex-csr: tex%0d_format=%0h", $time, CORE_ID, i, tex_format[i]);
                $display("%t: core%0d-tex-csr: tex%0d_wrap_u=%0h", $time, CORE_ID, i, tex_wrap_u[i]);
                $display("%t: core%0d-tex-csr: tex%0d_wrap_v=%0h", $time, CORE_ID, i, tex_wrap_v[i]);
                $display("%t: core%0d-tex-csr: tex%0d_filter=%0h", $time, CORE_ID, i, tex_filter[i]);
                $display("%t: core%0d-tex-csr: tex%0d_mipoff[0]=%0h", $time, CORE_ID, i, tex_mipoff[i][0]);
                $display("%t: core%0d-tex-csr: tex%0d_width[0]=%0h", $time, CORE_ID, i, tex_width[i][0]);
                $display("%t: core%0d-tex-csr: tex%0d_height[0]=%0h", $time, CORE_ID, i, tex_height[i][0]);
            end
        end
    end
    always @(posedge clk) begin
        if (tex_req_if.valid && tex_req_if.ready) begin
             $display("%t: core%0d-tex-req: wid=%0d, PC=%0h, tmask=%b, unit=%0d, lod=%0h, u=", 
                    $time, CORE_ID, tex_req_if.wid, tex_req_if.PC, tex_req_if.tmask, tex_req_if.unit, tex_req_if.lod);
            `PRINT_ARRAY1D(tex_req_if.u, `NUM_THREADS);
            $write(", v=");
            `PRINT_ARRAY1D(tex_req_if.v, `NUM_THREADS);
            $write("\n");
        end
        if (tex_rsp_if.valid && tex_rsp_if.ready) begin
             $write("%t: core%0d-tex-rsp: wid=%0d, PC=%0h, tmask=%b, data=", 
                    $time, CORE_ID, tex_rsp_if.wid, tex_rsp_if.PC, tex_rsp_if.tmask);
            `PRINT_ARRAY1D(tex_rsp_if.data, `NUM_THREADS);
            $write("\n");
        end
    end
`endif

endmodule