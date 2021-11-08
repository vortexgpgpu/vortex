`include "VX_tex_define.vh"

module VX_tex_unit #(  
    parameter CORE_ID = 0
) (
    input wire  clk,
    input wire  reset,    

    // Texture unit <-> Memory Unit
    VX_dcache_req_if.master dcache_req_if,
    VX_dcache_rsp_if.slave  dcache_rsp_if,

    // Inputs
    VX_tex_req_if.slave     tex_req_if,
    VX_tex_csr_if.slave     tex_csr_if,

    // Outputs
    VX_tex_rsp_if.master    tex_rsp_if
);

    localparam REQ_INFOW_S = `NR_BITS + 1 + `NW_BITS + 32;
    localparam REQ_INFOW_A = `TEX_FORMAT_BITS + REQ_INFOW_S;
    localparam REQ_INFOW_M = (2 * `NUM_THREADS * `BLEND_FRAC) + REQ_INFOW_A;
    
    reg [`TEX_MIPOFF_BITS-1:0]    tex_mipoff [`NUM_TEX_UNITS-1:0][(1 << `TEX_LOD_BITS)-1:0];
    reg [1:0][`TEX_DIM_BITS-1:0]  tex_dims   [`NUM_TEX_UNITS-1:0][(1 << `TEX_LOD_BITS)-1:0];
    reg [`TEX_ADDR_BITS-1:0]      tex_baddr  [`NUM_TEX_UNITS-1:0];     
    reg [`TEX_FORMAT_BITS-1:0]    tex_format [`NUM_TEX_UNITS-1:0];
    reg [1:0][`TEX_WRAP_BITS-1:0] tex_wraps  [`NUM_TEX_UNITS-1:0];
    reg [`TEX_FILTER_BITS-1:0]    tex_filter [`NUM_TEX_UNITS-1:0];

    // CSRs programming    

    reg [`NUM_TEX_UNITS-1:0] csrs_dirty;
    `UNUSED_VAR (csrs_dirty)

    for (genvar i = 0; i < `NUM_TEX_UNITS; ++i) begin
        wire [`TEX_LOD_BITS-1:0] mip_level = tex_csr_if.write_data[28 +: `TEX_LOD_BITS];
        always @(posedge clk) begin  
            if (tex_csr_if.write_enable) begin
                case (tex_csr_if.write_addr)
                    `CSR_TEX_ADDR(i) : begin 
                        tex_baddr[i]  <= tex_csr_if.write_data[`TEX_ADDR_BITS-1:0];
                        csrs_dirty[i] <= 1;
                    end
                    `CSR_TEX_FORMAT(i) : begin 
                        tex_format[i] <= tex_csr_if.write_data[`TEX_FORMAT_BITS-1:0];
                        csrs_dirty[i] <= 1;
                    end
                    `CSR_TEX_WRAP(i) : begin
                        tex_wraps[i][0] <= tex_csr_if.write_data[0 +: `TEX_WRAP_BITS];
                        tex_wraps[i][1] <= tex_csr_if.write_data[`TEX_WRAP_BITS +: `TEX_WRAP_BITS];
                        csrs_dirty[i] <= 1;
                    end
                    `CSR_TEX_FILTER(i) : begin 
                        tex_filter[i] <= tex_csr_if.write_data[`TEX_FILTER_BITS-1:0];                        
                        csrs_dirty[i] <= 1;
                    end
                    `CSR_TEX_MIPOFF(i) : begin 
                        tex_mipoff[i][mip_level] <= tex_csr_if.write_data[`TEX_MIPOFF_BITS-1:0];
                        csrs_dirty[i] <= 1;
                    end
                    `CSR_TEX_WIDTH(i) : begin 
                        tex_dims[i][mip_level][0] <= tex_csr_if.write_data[`TEX_DIM_BITS-1:0];
                        csrs_dirty[i] <= 1;
                    end
                    `CSR_TEX_HEIGHT(i) : begin 
                        tex_dims[i][mip_level][1] <= tex_csr_if.write_data[`TEX_DIM_BITS-1:0];
                        csrs_dirty[i] <= 1;
                    end
                endcase
            end
            if (reset || (tex_req_if.valid && tex_req_if.ready)) begin                
                csrs_dirty[i] <= '0;
            end
        end
    end

    // mipmap attributes

    wire [`NUM_THREADS-1:0][`TEX_MIPOFF_BITS-1:0] sel_mipoff;    
    wire [`NUM_THREADS-1:0][1:0][`TEX_DIM_BITS-1:0] sel_dims;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        wire [`NTEX_BITS-1:0] unit = tex_req_if.unit[`NTEX_BITS-1:0];
        wire [`TEX_LOD_BITS-1:0] mip_level = tex_req_if.lod[i][20+:`TEX_LOD_BITS];        
        assign sel_mipoff[i] = tex_mipoff[unit][mip_level];
        assign sel_dims[i]   = tex_dims[unit][mip_level];
    end

    // address generation

    wire mem_req_valid;
    wire [`NUM_THREADS-1:0] mem_req_tmask;
    wire [`TEX_FILTER_BITS-1:0] mem_req_filter;
    wire [`TEX_STRIDE_BITS-1:0] mem_req_stride;
    wire [`NUM_THREADS-1:0][1:0][`BLEND_FRAC-1:0] mem_req_blends;
    wire [`NUM_THREADS-1:0][3:0][31:0] mem_req_addr;
    wire [REQ_INFOW_A-1:0] mem_req_info;
    wire mem_req_ready;
                
    VX_tex_addr #(
        .CORE_ID   (CORE_ID),
        .REQ_INFOW (REQ_INFOW_A),
        .NUM_REQS  (`NUM_THREADS)
    ) tex_addr (
        .clk        (clk),
        .reset      (reset),

        .req_valid  (tex_req_if.valid),
        .req_tmask  (tex_req_if.tmask),
        .req_coords (tex_req_if.coords),
        .req_format (tex_format[tex_req_if.unit]),
        .req_filter (tex_filter[tex_req_if.unit]),
        .req_wraps  (tex_wraps[tex_req_if.unit]),
        .req_baseaddr (tex_baddr[tex_req_if.unit]),    
        .req_mipoff (sel_mipoff),
        .req_logdims (sel_dims),
        .req_info   ({tex_format[tex_req_if.unit], tex_req_if.rd, tex_req_if.wb, tex_req_if.wid, tex_req_if.PC}),
        .req_ready  (tex_req_if.ready),

        .rsp_valid  (mem_req_valid), 
        .rsp_tmask  (mem_req_tmask),
        .rsp_filter (mem_req_filter), 
        .rsp_stride (mem_req_stride),
        .rsp_addr   (mem_req_addr),
        .rsp_blends (mem_req_blends),
        .rsp_info   (mem_req_info),
        .rsp_ready  (mem_req_ready)
    );

    // retrieve texel values from memory  

    wire mem_rsp_valid;
    wire [`NUM_THREADS-1:0] mem_rsp_tmask;
    wire [`NUM_THREADS-1:0][3:0][31:0] mem_rsp_data;
    wire [REQ_INFOW_M-1:0] mem_rsp_info;
    wire mem_rsp_ready;        

    VX_tex_mem #(
        .CORE_ID   (CORE_ID),
        .REQ_INFOW (REQ_INFOW_M),
        .NUM_REQS  (`NUM_THREADS)
    ) tex_mem (
        .clk           (clk),
        .reset         (reset),

        // memory interface
        .dcache_req_if (dcache_req_if),
        .dcache_rsp_if (dcache_rsp_if),

        // inputs
        .req_valid (mem_req_valid),
        .req_tmask (mem_req_tmask),
        .req_filter(mem_req_filter), 
        .req_stride(mem_req_stride),
        .req_addr  (mem_req_addr),
        .req_info  ({mem_req_blends, mem_req_info}),
        .req_ready (mem_req_ready),

        // outputs
        .rsp_valid (mem_rsp_valid),
        .rsp_tmask (mem_rsp_tmask),
        .rsp_data  (mem_rsp_data),
        .rsp_info  (mem_rsp_info),
        .rsp_ready (mem_rsp_ready)
    );

    // apply sampler

    wire [`NUM_THREADS-1:0][1:0][`BLEND_FRAC-1:0] rsp_blends;
    wire [`TEX_FORMAT_BITS-1:0] rsp_format;
    wire [REQ_INFOW_S-1:0] rsp_info;
    
    assign {rsp_blends, rsp_format, rsp_info} = mem_rsp_info;

    VX_tex_sampler #(
        .CORE_ID   (CORE_ID),
        .REQ_INFOW (REQ_INFOW_S),
        .NUM_REQS  (`NUM_THREADS)
    ) tex_sampler (
        .clk        (clk),
        .reset      (reset),

        // inputs
        .req_valid  (mem_rsp_valid),  
        .req_tmask  (mem_rsp_tmask),
        .req_data   (mem_rsp_data), 
        .req_format (rsp_format),  
        .req_blends (rsp_blends),
        .req_info   (rsp_info), 
        .req_ready  (mem_rsp_ready),

        // outputs
        .rsp_valid  (tex_rsp_if.valid),
        .rsp_tmask  (tex_rsp_if.tmask),
        .rsp_data   (tex_rsp_if.data),
        .rsp_info   ({tex_rsp_if.rd, tex_rsp_if.wb, tex_rsp_if.wid, tex_rsp_if.PC}),
        .rsp_ready  (tex_rsp_if.ready)
    );    

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (tex_req_if.valid && tex_req_if.ready) begin
            for (integer i = 0; i < `NUM_TEX_UNITS; ++i) begin
                if (csrs_dirty[i]) begin
                    dpi_trace("%d: core%0d-tex-csr: tex%0d_addr=%0h\n", $time, CORE_ID, i, tex_baddr[i]);
                    dpi_trace("%d: core%0d-tex-csr: tex%0d_format=%0h\n", $time, CORE_ID, i, tex_format[i]);
                    dpi_trace("%d: core%0d-tex-csr: tex%0d_wrap_u=%0h\n", $time, CORE_ID, i, tex_wraps[i][0]);
                    dpi_trace("%d: core%0d-tex-csr: tex%0d_wrap_v=%0h\n", $time, CORE_ID, i, tex_wraps[i][1]);
                    dpi_trace("%d: core%0d-tex-csr: tex%0d_filter=%0h\n", $time, CORE_ID, i, tex_filter[i]);
                    dpi_trace("%d: core%0d-tex-csr: tex%0d_mipoff[0]=%0h\n", $time, CORE_ID, i, tex_mipoff[i][0]);
                    dpi_trace("%d: core%0d-tex-csr: tex%0d_width[0]=%0h\n", $time, CORE_ID, i, tex_dims[i][0][0]);
                    dpi_trace("%d: core%0d-tex-csr: tex%0d_height[0]=%0h\n", $time, CORE_ID, i, tex_dims[i][0][1]);
                end
            end
            
            dpi_trace("%d: core%0d-tex-req: wid=%0d, PC=%0h, tmask=%b, unit=%0d, lod=%0h, u=", 
                    $time, CORE_ID, tex_req_if.wid, tex_req_if.PC, tex_req_if.tmask, tex_req_if.unit, tex_req_if.lod);
            `TRACE_ARRAY1D(tex_req_if.coords[0], `NUM_THREADS);
            dpi_trace(", v=");
            `TRACE_ARRAY1D(tex_req_if.coords[1], `NUM_THREADS);
            dpi_trace("\n");
        end
        if (tex_rsp_if.valid && tex_rsp_if.ready) begin
             dpi_trace("%d: core%0d-tex-rsp: wid=%0d, PC=%0h, tmask=%b, data=", 
                    $time, CORE_ID, tex_rsp_if.wid, tex_rsp_if.PC, tex_rsp_if.tmask);
            `TRACE_ARRAY1D(tex_rsp_if.data, `NUM_THREADS);
            dpi_trace("\n");
        end
    end
`endif

endmodule