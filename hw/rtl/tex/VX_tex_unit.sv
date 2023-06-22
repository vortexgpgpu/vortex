`include "VX_tex_define.vh"

module VX_tex_unit #(  
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = 1,
    parameter TAG_WIDTH = 1
) (
    input wire  clk,
    input wire  reset,    

    // PERF
`ifdef PERF_ENABLE
    VX_tex_perf_if.master   perf_tex_if,
`endif

    // Memory interface
    VX_cache_bus_if.master  cache_bus_if,

    // Inputs
    VX_dcr_write_if.slave   dcr_write_if,
    VX_tex_req_if.slave     tex_req_if,
    
    // Outputs
    VX_tex_rsp_if.master    tex_rsp_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    
    localparam BLEND_FRAC_W = (2 * NUM_LANES * `TEX_BLEND_FRAC);  

    // DCRs
    
    tex_dcrs_t tex_dcrs;

    VX_tex_dcr #(
        .INSTANCE_ID (INSTANCE_ID),
        .NUM_STAGES  (`TEX_STAGE_COUNT)
    ) tex_dcr (
        .clk        (clk),
        .reset      (reset),
        .dcr_write_if(dcr_write_if),
        .stage      (tex_req_if.stage),
        .tex_dcrs   (tex_dcrs)
    );

    // Texture stage select 

    wire                                        req_valid;
    wire [NUM_LANES-1:0]                        req_mask;          
    wire [`TEX_FILTER_BITS-1:0]                 req_filter;    
    wire [`TEX_FORMAT_BITS-1:0]                 req_format;    
    wire [1:0][`TEX_WRAP_BITS-1:0]              req_wraps;
    wire [1:0][`TEX_LOD_BITS-1:0]               req_logdims;
    wire [`TEX_ADDR_BITS-1:0]                   req_baseaddr;
    wire [1:0][NUM_LANES-1:0][31:0]             req_coords;
    wire [NUM_LANES-1:0][`TEX_LOD_BITS-1:0]     req_miplevel, sel_miplevel;
    wire [NUM_LANES-1:0][`TEX_MIPOFF_BITS-1:0]  req_mipoff, sel_mipoff;    
    wire [TAG_WIDTH-1:0]                        req_tag;
    wire                                        req_ready;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign sel_miplevel[i] = tex_req_if.lod[i][`TEX_LOD_BITS-1:0];
        assign sel_mipoff[i] = tex_dcrs.mipoff[sel_miplevel[i]];
    end

    VX_generic_buffer #(
        .DATAW   (NUM_LANES  + `TEX_FILTER_BITS + `TEX_FORMAT_BITS + 2 * `TEX_WRAP_BITS + 2 * `TEX_LOD_BITS + `TEX_ADDR_BITS + NUM_LANES * (2 * 32 + `TEX_LOD_BITS + `TEX_MIPOFF_BITS) + TAG_WIDTH),
        .OUT_REG (1)
    ) pipe_reg (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (tex_req_if.valid),
        .ready_in  (tex_req_if.ready),
        .data_in   ({tex_req_if.mask, tex_dcrs.filter, tex_dcrs.format, tex_dcrs.wraps, tex_dcrs.logdims, tex_dcrs.baseaddr, tex_req_if.coords, sel_miplevel, sel_mipoff, tex_req_if.tag}),
        .data_out  ({req_mask,        req_filter,      req_format,      req_wraps,      req_logdims,      req_baseaddr,      req_coords,        req_miplevel, req_mipoff, req_tag}),
        .valid_out (req_valid),
        .ready_out (req_ready)
    );

    // address generation

    wire mem_req_valid;
    wire [NUM_LANES-1:0] mem_req_mask;
    wire [`TEX_FILTER_BITS-1:0] mem_req_filter;
    wire [`TEX_LGSTRIDE_BITS-1:0] mem_req_lgstride;
    wire [NUM_LANES-1:0][1:0][`TEX_BLEND_FRAC-1:0] mem_req_blends;
    wire [NUM_LANES-1:0][3:0][31:0] mem_req_addr;
    wire [NUM_LANES-1:0][31:0] mem_req_baseaddr;
    wire [(TAG_WIDTH + `TEX_FORMAT_BITS)-1:0] mem_req_info;
    wire mem_req_ready;

    `RESET_RELAY (addr_reset, reset);
                
    VX_tex_addr #(
        .INSTANCE_ID (INSTANCE_ID),
        .REQ_INFOW   (TAG_WIDTH + `TEX_FORMAT_BITS),
        .NUM_LANES   (NUM_LANES)
    ) tex_addr (
        .clk        (clk),
        .reset      (addr_reset),

        // inputs
        .req_valid  (req_valid),
        .req_mask   (req_mask),
        .req_coords (req_coords),
        .req_format (req_format),
        .req_filter (req_filter),
        .req_wraps  (req_wraps),
        .req_baseaddr(req_baseaddr),    
        .req_miplevel(req_miplevel),
        .req_mipoff (req_mipoff),
        .req_logdims(req_logdims),
        .req_info   ({req_tag, req_format}),
        .req_ready  (req_ready),

        // outputs
        .rsp_valid  (mem_req_valid), 
        .rsp_mask   (mem_req_mask),
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
    wire [NUM_LANES-1:0][3:0][31:0] mem_rsp_data;
    wire [(TAG_WIDTH + `TEX_FORMAT_BITS + BLEND_FRAC_W)-1:0] mem_rsp_info;
    wire mem_rsp_ready;      

    `RESET_RELAY (mem_reset, reset);  

    VX_tex_mem #(
        .INSTANCE_ID (INSTANCE_ID),
        .REQ_INFOW   (TAG_WIDTH + `TEX_FORMAT_BITS + BLEND_FRAC_W),
        .NUM_LANES   (NUM_LANES)
    ) tex_mem (
        .clk       (clk),
        .reset     (mem_reset),

        // memory interface
        .cache_bus_if (cache_bus_if),

        // inputs
        .req_valid (mem_req_valid),
        .req_mask  (mem_req_mask),
        .req_filter(mem_req_filter), 
        .req_lgstride(mem_req_lgstride),
        .req_baseaddr(mem_req_baseaddr),    
        .req_addr  (mem_req_addr),
        .req_info  ({mem_req_info, mem_req_blends}),
        .req_ready (mem_req_ready),

        // outputs
        .rsp_valid (mem_rsp_valid),
        .rsp_data  (mem_rsp_data),
        .rsp_info  (mem_rsp_info),
        .rsp_ready (mem_rsp_ready)
    );

    // apply sampler

    wire sampler_rsp_valid;
    wire [NUM_LANES-1:0][31:0] sampler_rsp_data;
    wire [TAG_WIDTH-1:0] sampler_rsp_info;
    wire sampler_rsp_ready;

    `RESET_RELAY (sample_reset, reset);

    VX_tex_sampler #(
        .INSTANCE_ID (INSTANCE_ID),
        .REQ_INFOW   (TAG_WIDTH),
        .NUM_LANES   (NUM_LANES)
    ) tex_sampler (
        .clk        (clk),
        .reset      (sample_reset),

        // inputs
        .req_valid  (mem_rsp_valid),
        .req_data   (mem_rsp_data), 
        .req_blends (mem_rsp_info[0 +: BLEND_FRAC_W]),
        .req_format (mem_rsp_info[BLEND_FRAC_W +: `TEX_FORMAT_BITS]),
        .req_info   (mem_rsp_info[(BLEND_FRAC_W + `TEX_FORMAT_BITS) +: TAG_WIDTH]),
        .req_ready  (mem_rsp_ready),

        // outputs
        .rsp_valid  (sampler_rsp_valid),
        .rsp_data   (sampler_rsp_data),
        .rsp_info   (sampler_rsp_info),
        .rsp_ready  (sampler_rsp_ready)
    );

    VX_skid_buffer #(
        .DATAW   (NUM_LANES * 32 + TAG_WIDTH),
        .OUT_REG (1)
    ) rsp_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (sampler_rsp_valid),
        .ready_in  (sampler_rsp_ready),
        .data_in   ({sampler_rsp_data,  sampler_rsp_info}),
        .data_out  ({tex_rsp_if.texels, tex_rsp_if.tag}),
        .valid_out (tex_rsp_if.valid),
        .ready_out (tex_rsp_if.ready)
    );

`ifdef PERF_ENABLE
    wire [$clog2(TCACHE_NUM_REQS+1)-1:0] perf_mem_req_per_cycle;
    wire [$clog2(TCACHE_NUM_REQS+1)-1:0] perf_mem_rsp_per_cycle;
    wire [$clog2(TCACHE_NUM_REQS+1)+1-1:0] perf_pending_reads_cycle;

    wire [TCACHE_NUM_REQS-1:0] perf_mem_req_fire = cache_bus_if.req_valid & cache_bus_if.req_ready;
    wire [TCACHE_NUM_REQS-1:0] perf_mem_rsp_fire = cache_bus_if.rsp_valid & cache_bus_if.rsp_ready;

    `POP_COUNT(perf_mem_req_per_cycle, perf_mem_req_fire);
    `POP_COUNT(perf_mem_rsp_per_cycle, perf_mem_rsp_fire);

    reg [`PERF_CTR_BITS-1:0] perf_pending_reads;   
    assign perf_pending_reads_cycle = perf_mem_req_per_cycle - perf_mem_rsp_per_cycle;

    always @(posedge clk) begin
        if (reset) begin
            perf_pending_reads <= '0;
        end else begin
            perf_pending_reads <= $signed(perf_pending_reads) + `PERF_CTR_BITS'($signed(perf_pending_reads_cycle));
        end
    end

    wire perf_stall_cycle = tex_req_if.valid & ~tex_req_if.ready;

    reg [`PERF_CTR_BITS-1:0] perf_mem_reads;
    reg [`PERF_CTR_BITS-1:0] perf_mem_latency;
    reg [`PERF_CTR_BITS-1:0] perf_stall_cycles;

    always @(posedge clk) begin
        if (reset) begin
            perf_mem_reads    <= '0;
            perf_mem_latency  <= '0;
            perf_stall_cycles <= '0;
        end else begin
            perf_mem_reads    <= perf_mem_reads + `PERF_CTR_BITS'(perf_mem_req_per_cycle);
            perf_mem_latency  <= perf_mem_latency + `PERF_CTR_BITS'(perf_pending_reads);            
            perf_stall_cycles <= perf_stall_cycles + `PERF_CTR_BITS'(perf_stall_cycle);
        end
    end

    assign perf_tex_if.mem_reads    = perf_mem_reads;
    assign perf_tex_if.mem_latency  = perf_mem_latency;
    assign perf_tex_if.stall_cycles = perf_stall_cycles;
`endif  

`ifdef DBG_TRACE_TEX
    always @(posedge clk) begin
        if (tex_req_if.valid && tex_req_if.ready) begin
            `TRACE(1, ("%d: %s-req: mask=%b, stage=%0d, lod=0x%0h, u=", 
                    $time, INSTANCE_ID, tex_req_if.mask, tex_req_if.stage, tex_req_if.lod));
            `TRACE_ARRAY1D(1, tex_req_if.coords[0], NUM_LANES);
            `TRACE(1, (", v="));
            `TRACE_ARRAY1D(1, tex_req_if.coords[1], NUM_LANES);
            `TRACE(1, (", tag=0x%0h (#%0d)\n", tex_req_if.tag, tex_req_if.tag[TAG_WIDTH-1 -: `UUID_BITS]));
        end
        if (tex_rsp_if.valid && tex_rsp_if.ready) begin
            `TRACE(1, ("%d: %s-rsp: texels=", $time, INSTANCE_ID));
            `TRACE_ARRAY1D(1, tex_rsp_if.texels, NUM_LANES);
            `TRACE(1, (", tag=0x%0h (#%0d)\n", tex_rsp_if.tag, tex_rsp_if.tag[TAG_WIDTH-1 -: `UUID_BITS]));
        end
    end
`endif

endmodule

///////////////////////////////////////////////////////////////////////////////

module VX_tex_unit_top #(  
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = `NUM_THREADS,
    parameter TAG_WIDTH = `TEX_REQ_TAG_WIDTH
) (
    input wire                              clk,
    input wire                              reset,    

    input wire                              dcr_write_valid,
    input wire [`VX_DCR_ADDR_WIDTH-1:0]     dcr_write_addr,
    input wire [`VX_DCR_DATA_WIDTH-1:0]     dcr_write_data,

    input  wire                             tex_req_valid,
    input  wire [NUM_LANES-1:0]             tex_req_mask,
    input  wire [1:0][NUM_LANES-1:0][31:0]  tex_req_coords,
    input  wire [NUM_LANES-1:0][`TEX_LOD_BITS-1:0] tex_req_lod,
    input  wire [`TEX_STAGE_BITS-1:0]       tex_req_stage,
    input  wire [TAG_WIDTH-1:0]             tex_req_tag,  
    output wire                             tex_req_ready,

    output wire                             tex_rsp_valid,
    output wire [NUM_LANES-1:0][31:0]       tex_rsp_texels,
    output wire [TAG_WIDTH-1:0]             tex_rsp_tag, 
    input  wire                             tex_rsp_ready,

    output wire [TCACHE_NUM_REQS-1:0]       cache_req_valid,
    output wire [TCACHE_NUM_REQS-1:0]       cache_req_rw,
    output wire [TCACHE_NUM_REQS-1:0][TCACHE_WORD_SIZE-1:0] cache_req_byteen,
    output wire [TCACHE_NUM_REQS-1:0][TCACHE_ADDR_WIDTH-1:0] cache_req_addr,
    output wire [TCACHE_NUM_REQS-1:0][TCACHE_WORD_SIZE*8-1:0] cache_req_data,
    output wire [TCACHE_NUM_REQS-1:0][TCACHE_TAG_WIDTH-1:0] cache_req_tag,
    input  wire [TCACHE_NUM_REQS-1:0]       cache_req_ready,

    input wire  [TCACHE_NUM_REQS-1:0]       cache_rsp_valid,
    input wire  [TCACHE_NUM_REQS-1:0][TCACHE_WORD_SIZE*8-1:0] cache_rsp_data,
    input wire  [TCACHE_NUM_REQS-1:0][TCACHE_TAG_WIDTH-1:0] cache_rsp_tag,
    output wire [TCACHE_NUM_REQS-1:0]       cache_rsp_ready
);
    
    VX_tex_perf_if perf_tex_if();

    VX_dcr_write_if dcr_write_if();

    assign dcr_write_if.valid = dcr_write_valid;
    assign dcr_write_if.addr = dcr_write_addr;
    assign dcr_write_if.data = dcr_write_data;

    VX_tex_req_if #(
        .NUM_LANES (NUM_LANES),
        .TAG_WIDTH (TAG_WIDTH)
    ) tex_req_if();

    VX_tex_rsp_if #(
        .NUM_LANES (NUM_LANES),
        .TAG_WIDTH (TAG_WIDTH)
    ) tex_rsp_if();

    assign tex_req_if.valid = tex_req_valid;
    assign tex_req_if.mask = tex_req_mask;
    assign tex_req_if.coords = tex_req_coords;
    assign tex_req_if.lod = tex_req_lod;
    assign tex_req_if.stage = tex_req_stage;
    assign tex_req_if.tag = tex_req_tag;  
    assign tex_req_ready = tex_req_if.ready;

    assign tex_rsp_valid = tex_rsp_if.valid;
    assign tex_rsp_texels = tex_rsp_if.texels;
    assign tex_rsp_tag = tex_rsp_if.tag; 
    assign tex_rsp_if.ready = tex_rsp_ready;

    VX_cache_bus_if #(
        .NUM_REQS  (TCACHE_NUM_REQS), 
        .WORD_SIZE (TCACHE_WORD_SIZE), 
        .TAG_WIDTH (TCACHE_TAG_WIDTH)
    ) cache_bus_if();

    assign cache_req_valid = cache_bus_if.req_valid;
    assign cache_req_rw = cache_bus_if.req_rw;
    assign cache_req_byteen = cache_bus_if.req_byteen;
    assign cache_req_addr = cache_bus_if.req_addr;
    assign cache_req_data = cache_bus_if.req_data;
    assign cache_req_tag = cache_bus_if.req_tag;
    assign cache_bus_if.req_ready = cache_req_ready;

    assign cache_bus_if.rsp_valid = cache_rsp_valid;
    assign cache_bus_if.rsp_tag = cache_rsp_tag;
    assign cache_bus_if.rsp_data = cache_rsp_data;
    assign cache_rsp_ready = cache_bus_if.rsp_ready;

    VX_tex_unit #(
        .INSTANCE_ID (INSTANCE_ID),
        .NUM_LANES   (NUM_LANES),
        .TAG_WIDTH   (TAG_WIDTH)
    ) tex_unit (
        .clk          (clk),
        .reset        (reset),
    `ifdef PERF_ENABLE
        .perf_tex_if  (perf_tex_if),
    `endif
        .dcr_write_if (dcr_write_if),
        .tex_req_if   (tex_req_if),
        .tex_rsp_if   (tex_rsp_if),
        .cache_bus_if (cache_bus_if)
    );

endmodule
