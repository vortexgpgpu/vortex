`include "VX_tex_define.vh"

module VX_tex_unit #(  
    parameter CORE_ID = 0
) (
    input wire  clk,
    input wire  reset,    

    // PERF
`ifdef PERF_ENABLE
    VX_perf_tex_if.master perf_tex_if,
`endif

    // Texture unit <-> Memory Unit
    VX_dcache_req_if.master dcache_req_if,
    VX_dcache_rsp_if.slave  dcache_rsp_if,

    // Inputs
    VX_tex_req_if.slave     tex_req_if,
    VX_tex_csr_if.slave     tex_csr_if,

    // Outputs
    VX_tex_rsp_if.master    tex_rsp_if
);

    localparam REQ_INFO_W = `NR_BITS + 1 + `NW_BITS + 32 + `UUID_BITS;
    localparam BLEND_FRAC_W = (2 * `NUM_THREADS * `TEX_BLEND_FRAC);
    
    reg [$clog2(`NUM_TEX_UNITS)-1:0] csr_tex_unit;
    reg [`TEX_MIPOFF_BITS-1:0]    tex_mipoff [`NUM_TEX_UNITS-1:0][(`TEX_LOD_MAX+1)-1:0];
    reg [1:0][`TEX_LOD_BITS-1:0]  tex_logdims [`NUM_TEX_UNITS-1:0];
    reg [1:0][`TEX_WRAP_BITS-1:0] tex_wraps  [`NUM_TEX_UNITS-1:0];
    reg [`TEX_ADDR_BITS-1:0]      tex_baddr  [`NUM_TEX_UNITS-1:0];     
    reg [`TEX_FORMAT_BITS-1:0]    tex_format [`NUM_TEX_UNITS-1:0];
    reg [`TEX_FILTER_BITS-1:0]    tex_filter [`NUM_TEX_UNITS-1:0];

    // CSRs programming

    always @(posedge clk) begin
        if (tex_csr_if.write_enable) begin
            case (tex_csr_if.write_addr)
                `CSR_TEX_UNIT: begin 
                    csr_tex_unit <= tex_csr_if.write_data[$clog2(`NUM_TEX_UNITS)-1:0];
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
    wire [`UUID_BITS-1:0] write_uuid = tex_csr_if.write_uuid;
    `UNUSED_VAR (write_uuid);

    // mipmap attributes

    wire [`NUM_THREADS-1:0][`TEX_LOD_BITS-1:0]      mip_level;
    wire [`NUM_THREADS-1:0][`TEX_MIPOFF_BITS-1:0]   sel_mipoff;
    wire [`NUM_THREADS-1:0][1:0][`TEX_LOD_BITS-1:0] sel_logdims;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        wire [`NTEX_BITS-1:0] unit = tex_req_if.unit[`NTEX_BITS-1:0];
        assign mip_level[i]      = tex_req_if.lod[i][`TEX_LOD_BITS-1:0];        
        assign sel_mipoff[i]     = tex_mipoff[unit][mip_level[i]];
        assign sel_logdims[i][0] = tex_logdims[unit][0];
        assign sel_logdims[i][1] = tex_logdims[unit][1];
    end

    // address generation

    wire mem_req_valid;
    wire [`NUM_THREADS-1:0] mem_req_tmask;
    wire [`TEX_FILTER_BITS-1:0] mem_req_filter;
    wire [`TEX_LGSTRIDE_BITS-1:0] mem_req_lgstride;
    wire [`NUM_THREADS-1:0][1:0][`TEX_BLEND_FRAC-1:0] mem_req_blends;
    wire [`NUM_THREADS-1:0][3:0][31:0] mem_req_addr;
    wire [`NUM_THREADS-1:0][31:0] mem_req_baseaddr;
    wire [(`TEX_FORMAT_BITS + REQ_INFO_W)-1:0] mem_req_info;
    wire mem_req_ready;
                
    VX_tex_addr #(
        .CORE_ID   (CORE_ID),
        .REQ_INFOW (`TEX_FORMAT_BITS + REQ_INFO_W),
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
        .req_baseaddr(tex_baddr[tex_req_if.unit]),    
        .mip_level  (mip_level),
        .req_mipoff (sel_mipoff),
        .req_logdims(sel_logdims),
        .req_info   ({tex_format[tex_req_if.unit], tex_req_if.rd, tex_req_if.wb, tex_req_if.wid, tex_req_if.PC, tex_req_if.uuid}),
        .req_ready  (tex_req_if.ready),

        .rsp_valid  (mem_req_valid), 
        .rsp_tmask  (mem_req_tmask),
        .rsp_filter (mem_req_filter), 
        .rsp_lgstride(mem_req_lgstride),
        .rsp_baseaddr(mem_req_baseaddr),
        .rsp_addr   (mem_req_addr),
        .rsp_blends (mem_req_blends),
        .rsp_info   (mem_req_info),
        .rsp_ready  (mem_req_ready)
    );

    // retrieve texel values from memory  

    wire mem_rsp_valid;
    wire [`NUM_THREADS-1:0] mem_rsp_tmask;
    wire [`NUM_THREADS-1:0][3:0][31:0] mem_rsp_data;
    wire [(BLEND_FRAC_W + `TEX_FORMAT_BITS + REQ_INFO_W)-1:0] mem_rsp_info;
    wire mem_rsp_ready;        

    VX_tex_mem #(
        .CORE_ID   (CORE_ID),
        .REQ_INFOW (BLEND_FRAC_W + `TEX_FORMAT_BITS + REQ_INFO_W),
        .NUM_REQS  (`NUM_THREADS)
    ) tex_mem (
        .clk       (clk),
        .reset     (reset),

        // memory interface
        .dcache_req_if (dcache_req_if),
        .dcache_rsp_if (dcache_rsp_if),

        // inputs
        .req_valid (mem_req_valid),
        .req_tmask (mem_req_tmask),
        .req_filter(mem_req_filter), 
        .req_lgstride(mem_req_lgstride),
        .req_baseaddr(mem_req_baseaddr),    
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

    VX_tex_sampler #(
        .CORE_ID   (CORE_ID),
        .REQ_INFOW (REQ_INFO_W),
        .NUM_REQS  (`NUM_THREADS)
    ) tex_sampler (
        .clk        (clk),
        .reset      (reset),

        // inputs
        .req_valid  (mem_rsp_valid),  
        .req_tmask  (mem_rsp_tmask),
        .req_data   (mem_rsp_data), 
        .req_blends (mem_rsp_info[(REQ_INFO_W+`TEX_FORMAT_BITS) +: BLEND_FRAC_W]),
        .req_format (mem_rsp_info[REQ_INFO_W +: `TEX_FORMAT_BITS]),
        .req_info   (mem_rsp_info[0 +: REQ_INFO_W]),
        .req_ready  (mem_rsp_ready),

        // outputs
        .rsp_valid  (tex_rsp_if.valid),
        .rsp_tmask  (tex_rsp_if.tmask),
        .rsp_data   (tex_rsp_if.data),
        .rsp_info   ({tex_rsp_if.rd, tex_rsp_if.wb, tex_rsp_if.wid, tex_rsp_if.PC, tex_rsp_if.uuid}),
        .rsp_ready  (tex_rsp_if.ready)
    );

`ifdef PERF_ENABLE
    wire [$clog2(`NUM_THREADS+1)-1:0] perf_mem_req_per_cycle;
    wire [$clog2(`NUM_THREADS+1)-1:0] perf_mem_rsp_per_cycle;

    wire [`NUM_THREADS-1:0] perf_mem_req_per_mask = dcache_req_if.valid & dcache_req_if.ready;
    wire [`NUM_THREADS-1:0] perf_mem_rsp_per_mask = dcache_rsp_if.tmask & {`NUM_THREADS{dcache_rsp_if.valid & dcache_rsp_if.ready}};

    `POP_COUNT(perf_mem_req_per_cycle, perf_mem_req_per_mask);    
    `POP_COUNT(perf_mem_rsp_per_cycle, perf_mem_rsp_per_mask);

    reg [`PERF_CTR_BITS-1:0] perf_pending_reads;   
    wire [$clog2(`NUM_THREADS+1)+1-1:0] perf_pending_reads_cycle = perf_mem_req_per_cycle - perf_mem_rsp_per_cycle;

    always @(posedge clk) begin
        if (reset) begin
            perf_pending_reads <= 0;
        end else begin
            perf_pending_reads <= perf_pending_reads + `PERF_CTR_BITS'($signed(perf_pending_reads_cycle));
        end
    end

    reg [`PERF_CTR_BITS-1:0] perf_mem_reads;
    reg [`PERF_CTR_BITS-1:0] perf_mem_latency;

    always @(posedge clk) begin
        if (reset) begin
            perf_mem_reads   <= 0;
            perf_mem_latency <= 0;
        end else begin
            perf_mem_reads   <= perf_mem_reads + `PERF_CTR_BITS'(perf_mem_req_per_cycle);
            perf_mem_latency <= perf_mem_latency + `PERF_CTR_BITS'(perf_pending_reads);
        end
    end

    assign perf_tex_if.mem_reads   = perf_mem_reads;
    assign perf_tex_if.mem_latency = perf_mem_latency;
`endif  

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (tex_csr_if.write_enable) begin
            dpi_trace("%d: core%0d-tex-csr: unit=%0d, state=", $time, CORE_ID, csr_tex_unit);
            trace_tex_state(tex_csr_if.write_addr);
            dpi_trace(", data=%0h (#%0d)\n", tex_csr_if.write_data, tex_csr_if.write_uuid);
        end
        if (tex_req_if.valid && tex_req_if.ready) begin
            dpi_trace("%d: core%0d-tex-req: wid=%0d, PC=%0h, tmask=%b, unit=%0d, lod=%0h, u=", 
                $time, CORE_ID, tex_req_if.wid, tex_req_if.PC, tex_req_if.tmask, tex_req_if.unit, tex_req_if.lod);
            `TRACE_ARRAY1D(tex_req_if.coords[0], `NUM_THREADS);
            dpi_trace(", v=");
            `TRACE_ARRAY1D(tex_req_if.coords[1], `NUM_THREADS);
            dpi_trace(" (#%0d)\n", tex_req_if.uuid);
        end
        if (tex_rsp_if.valid && tex_rsp_if.ready) begin
             dpi_trace("%d: core%0d-tex-rsp: wid=%0d, PC=%0h, tmask=%b, data=", 
                    $time, CORE_ID, tex_rsp_if.wid, tex_rsp_if.PC, tex_rsp_if.tmask);
            `TRACE_ARRAY1D(tex_rsp_if.data, `NUM_THREADS);
            dpi_trace(" (#%0d)\n", tex_rsp_if.uuid);
        end
    end
`endif

endmodule