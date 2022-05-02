`include "VX_define.vh"

module VX_cluster #(
    parameter CLUSTER_ID = 0
) ( 
    `SCOPE_IO_VX_cluster

    // Clock
    input  wire             clk,
    input  wire             reset,

    VX_dcr_base_if.slave    dcr_base_if,

`ifdef EXT_TEX_ENABLE
    VX_tex_dcr_if.slave     tex_dcr_if,
`endif
`ifdef EXT_RASTER_ENABLE
    VX_raster_dcr_if.slave  raster_dcr_if,
`endif
`ifdef EXT_ROP_ENABLE
    VX_rop_dcr_if.slave     rop_dcr_if,
`endif

    // Memory
    VX_mem_req_if.master    mem_req_if,
    VX_mem_rsp_if.slave     mem_rsp_if,

    // simulation helper signals
    output wire             sim_ebreak,
    output wire [`NUM_REGS-1:0][31:0] sim_last_wb_value,

    // Status
    output wire             busy
);

`ifdef EXT_RASTER_ENABLE

    VX_raster_req_if #(
        .NUM_LANES (`NUM_THREADS)
    ) per_core_raster_req_if[`NUM_CORES]();

    VX_raster_req_if #(
        .NUM_LANES (`NUM_THREADS)
    ) raster_req_if();

`ifdef PERF_ENABLE
    VX_perf_cache_if    perf_rcache_if();
    VX_raster_perf_if   raster_perf_if();
`endif

    VX_cache_req_if #(
        .NUM_REQS  (`RCACHE_NUM_REQS), 
        .WORD_SIZE (`RCACHE_WORD_SIZE), 
        .TAG_WIDTH (`RCACHE_TAG_WIDTH)
    ) rcache_req_if();

    VX_cache_rsp_if #(
        .NUM_REQS  (`RCACHE_NUM_REQS), 
        .WORD_SIZE (`RCACHE_WORD_SIZE), 
        .TAG_WIDTH (`RCACHE_TAG_WIDTH)
    ) rcache_rsp_if();

    `RESET_RELAY (raster_reset);

    VX_raster_unit #(
        .CLUSTER_ID  (CLUSTER_ID),
        .NUM_SLICES  (`RASTER_NUM_SLICES),
        .RASTER_TILE_SIZE (1 << `RASTER_TILE_LOGSIZE),
        .RASTER_BLOCK_SIZE(1 << `RASTER_BLOCK_LOGSIZE),
        .NUM_OUTPUTS (`NUM_THREADS)
    ) raster_unit (
        .clk           (clk),
        .reset         (raster_reset),
    `ifdef PERF_ENABLE
        .raster_perf_if(raster_perf_if),
    `endif
        .raster_req_if (raster_req_if),
        .raster_dcr_if (raster_dcr_if),        
        .cache_req_if  (rcache_req_if),
        .cache_rsp_if  (rcache_rsp_if)
    );

    VX_mem_req_if #(
        .DATA_WIDTH (`RCACHE_MEM_DATA_WIDTH),
        .ADDR_WIDTH (`RCACHE_MEM_ADDR_WIDTH),
        .TAG_WIDTH  (`RCACHE_MEM_TAG_WIDTH)
    ) rcache_mem_req_if();
    
    VX_mem_rsp_if #(
        .DATA_WIDTH (`RCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (`RCACHE_MEM_TAG_WIDTH)
    ) rcache_mem_rsp_if();

    VX_mem_req_if #(
        .DATA_WIDTH (`RCACHE_WORD_SIZE*8), 
        .ADDR_WIDTH (`RCACHE_ADDR_WIDTH),
        .TAG_WIDTH  (`RCACHE_TAG_WIDTH)
    ) rcache_req_qual_if[`RCACHE_NUM_REQS]();

    VX_mem_rsp_if #(
        .DATA_WIDTH (`RCACHE_WORD_SIZE*8), 
        .TAG_WIDTH (`RCACHE_TAG_WIDTH)
    ) rcache_rsp_qual_if[`RCACHE_NUM_REQS]();

    for (genvar i = 0; i < `RCACHE_NUM_REQS; ++i) begin
        `CACHE_REQ_TO_MEM(rcache_req_qual_if, rcache_req_if, i);
    end

    VX_cache #(
        .CACHE_ID           (`RCACHE_ID),
        .CACHE_SIZE         (`RCACHE_SIZE),
        .CACHE_LINE_SIZE    (`RCACHE_LINE_SIZE),
        .NUM_BANKS          (`RCACHE_NUM_BANKS),
        .NUM_WAYS           (`RCACHE_NUM_WAYS),
        .NUM_PORTS          (`RCACHE_NUM_PORTS),
        .WORD_SIZE          (`RCACHE_WORD_SIZE),
        .NUM_REQS           (`RCACHE_NUM_REQS),
        .CREQ_SIZE          (`RCACHE_CREQ_SIZE),
        .CRSQ_SIZE          (`RCACHE_CRSQ_SIZE),
        .MSHR_SIZE          (`RCACHE_MSHR_SIZE),
        .MRSQ_SIZE          (`RCACHE_MRSQ_SIZE),
        .MREQ_SIZE          (`RCACHE_MREQ_SIZE),
        .WRITE_ENABLE       (0),
        .REQ_UUID_BITS      (0),
        .CORE_TAG_WIDTH     (`RCACHE_TAG_WIDTH),
        .MEM_TAG_WIDTH      (`RCACHE_MEM_TAG_WIDTH),
        .NC_ENABLE          (0),
        .PASSTHRU           (!`RCACHE_ENABLED)
    ) rcache (
        `SCOPE_BIND_VX_cluster_rcache

    `ifdef PERF_ENABLE
        .perf_cache_if  (perf_rcache_if),
    `endif
        
        .clk            (clk),
        .reset          (raster_reset),
        .core_req_if    (rcache_req_qual_if),
        .core_rsp_if    (rcache_rsp_qual_if),
        .mem_req_if     (rcache_mem_req_if),
        .mem_rsp_if     (rcache_mem_rsp_if)
    );

    for (genvar i = 0; i < `RCACHE_NUM_REQS; ++i) begin
        `CACHE_RSP_FROM_MEM(rcache_rsp_if, rcache_rsp_qual_if, i);
    end  

    VX_raster_req_demux #(
        .NUM_REQS  (`NUM_CORES),
        .NUM_LANES (`NUM_THREADS),
        .BUFFERED  (1)
    ) raster_req_demux (
        .clk        (clk),
        .reset      (raster_reset),
        .req_in_if  (raster_req_if),
        .req_out_if (per_core_raster_req_if)
    );

`endif

`ifdef EXT_ROP_ENABLE

    VX_rop_req_if #(
        .NUM_LANES (`NUM_THREADS)
    ) per_core_rop_req_if[`NUM_CORES]();

    VX_rop_req_if #(
        .NUM_LANES (`NUM_THREADS)
    ) rop_req_if();

`ifdef PERF_ENABLE
    VX_perf_cache_if    perf_ocache_if();
    VX_rop_perf_if      rop_perf_if(); 
`endif
    
    VX_cache_req_if #(
        .NUM_REQS  (`OCACHE_NUM_REQS), 
        .WORD_SIZE (`OCACHE_WORD_SIZE), 
        .TAG_WIDTH (`OCACHE_TAG_WIDTH)
    ) ocache_req_if();

    VX_cache_rsp_if #(
        .NUM_REQS  (`OCACHE_NUM_REQS), 
        .WORD_SIZE (`OCACHE_WORD_SIZE), 
        .TAG_WIDTH (`OCACHE_TAG_WIDTH)
    ) ocache_rsp_if();

    `RESET_RELAY (rop_reset);

    VX_rop_unit #(
        .CLUSTER_ID (CLUSTER_ID),
        .NUM_SLICES (`ROP_NUM_SLICES),
        .NUM_LANES  (`NUM_THREADS)
    ) rop_unit (
        .clk           (clk),
        .reset         (rop_reset),
    `ifdef PERF_ENABLE
        .rop_perf_if   (rop_perf_if),
    `endif
        .rop_req_if    (rop_req_if),
        .rop_dcr_if    (rop_dcr_if),
        .cache_req_if  (ocache_req_if),
        .cache_rsp_if  (ocache_rsp_if)
    );

    VX_mem_req_if #(
        .DATA_WIDTH (`OCACHE_MEM_DATA_WIDTH),
        .ADDR_WIDTH (`OCACHE_MEM_ADDR_WIDTH),
        .TAG_WIDTH  (`OCACHE_MEM_TAG_WIDTH)
    ) ocache_mem_req_if();
    
    VX_mem_rsp_if #(
        .DATA_WIDTH (`OCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (`OCACHE_MEM_TAG_WIDTH)
    ) ocache_mem_rsp_if();

    VX_mem_req_if #(
        .DATA_WIDTH (`OCACHE_WORD_SIZE*8), 
        .ADDR_WIDTH (`OCACHE_ADDR_WIDTH),
        .TAG_WIDTH  (`OCACHE_TAG_WIDTH)
    ) ocache_req_qual_if[`OCACHE_NUM_REQS]();

    VX_mem_rsp_if #(
        .DATA_WIDTH (`OCACHE_WORD_SIZE*8), 
        .TAG_WIDTH (`OCACHE_TAG_WIDTH)
    ) ocache_rsp_qual_if[`OCACHE_NUM_REQS]();

    for (genvar i = 0; i < `OCACHE_NUM_REQS; ++i) begin
        `CACHE_REQ_TO_MEM(ocache_req_qual_if, ocache_req_if, i);
    end

    VX_cache #(
        .CACHE_ID           (`OCACHE_ID),
        .CACHE_SIZE         (`OCACHE_SIZE),
        .CACHE_LINE_SIZE    (`OCACHE_LINE_SIZE),
        .NUM_BANKS          (`OCACHE_NUM_BANKS),
        .NUM_WAYS           (`OCACHE_NUM_WAYS),
        .NUM_PORTS          (`OCACHE_NUM_PORTS),
        .WORD_SIZE          (`OCACHE_WORD_SIZE),
        .NUM_REQS           (`OCACHE_NUM_REQS),
        .CREQ_SIZE          (`OCACHE_CREQ_SIZE),
        .CRSQ_SIZE          (`OCACHE_CRSQ_SIZE),
        .MSHR_SIZE          (`OCACHE_MSHR_SIZE),
        .MRSQ_SIZE          (`OCACHE_MRSQ_SIZE),
        .MREQ_SIZE          (`OCACHE_MREQ_SIZE),
        .WRITE_ENABLE       (1),
        .REQ_UUID_BITS      (0),
        .CORE_TAG_WIDTH     (`OCACHE_TAG_WIDTH),
        .MEM_TAG_WIDTH      (`OCACHE_MEM_TAG_WIDTH),
        .NC_ENABLE          (0),
        .PASSTHRU           (!`OCACHE_ENABLED)
    ) ocache (
        `SCOPE_BIND_VX_cluster_ocache

    `ifdef PERF_ENABLE
        .perf_cache_if  (perf_ocache_if),
    `endif
        
        .clk            (clk),
        .reset          (rop_reset),

        .core_req_if    (ocache_req_qual_if),
        .core_rsp_if    (ocache_rsp_qual_if),
        .mem_req_if     (ocache_mem_req_if),
        .mem_rsp_if     (ocache_mem_rsp_if)
    );

    for (genvar i = 0; i < `OCACHE_NUM_REQS; ++i) begin
        `CACHE_RSP_FROM_MEM(ocache_rsp_if, ocache_rsp_qual_if, i);
    end

    VX_rop_req_mux #(
        .NUM_REQS  (`NUM_CORES),
        .NUM_LANES (`NUM_THREADS),
        .BUFFERED  (1)
    ) rop_req_mux (
        .clk        (clk),
        .reset      (rop_reset),
        .req_in_if  (per_core_rop_req_if),
        .req_out_if (rop_req_if)
    );

`endif

    wire [`NUM_CORES-1:0] per_core_sim_ebreak;
    wire [`NUM_CORES-1:0][`NUM_REGS-1:0][31:0] per_core_sim_last_wb_value;
    assign sim_ebreak = per_core_sim_ebreak[0];
    assign sim_last_wb_value = per_core_sim_last_wb_value[0];
    `UNUSED_VAR (per_core_sim_ebreak)
    `UNUSED_VAR (per_core_sim_last_wb_value)

    VX_mem_req_if #(
        .DATA_WIDTH (`DCACHE_MEM_DATA_WIDTH),
        .ADDR_WIDTH (`DCACHE_MEM_ADDR_WIDTH),
        .TAG_WIDTH  (`L1_MEM_TAG_WIDTH)
    ) per_core_mem_req_if[`NUM_CORES]();
    
    VX_mem_rsp_if #(
        .DATA_WIDTH (`DCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (`L1_MEM_TAG_WIDTH)
    ) per_core_mem_rsp_if[`NUM_CORES]();

    wire [`NUM_CORES-1:0] per_core_busy;

    for (genvar i = 0; i < `NUM_CORES; i++) begin

        `RESET_RELAY (core_reset);

        VX_core #(
            .CORE_ID ((CLUSTER_ID * `NUM_CORES) + i)
        ) core (
            `SCOPE_BIND_VX_cluster_core(i)

            .clk            (clk),
            .reset          (core_reset),

            .dcr_base_if    (dcr_base_if),

        `ifdef EXT_TEX_ENABLE
            .tex_dcr_if     (tex_dcr_if),
        `endif
        `ifdef EXT_RASTER_ENABLE        
            .raster_req_if  (per_core_raster_req_if[i]),
        `ifdef PERF_ENABLE
            .raster_perf_if (raster_perf_if),
            .perf_rcache_if (perf_rcache_if),
        `endif
        `endif
        `ifdef EXT_ROP_ENABLE        
            .rop_req_if     (per_core_rop_req_if[i]),
        `ifdef PERF_ENABLE
            .rop_perf_if    (rop_perf_if),
            .perf_ocache_if (perf_ocache_if),
        `endif
        `endif

            .mem_req_if     (per_core_mem_req_if[i]),
            .mem_rsp_if     (per_core_mem_rsp_if[i]),

            .sim_ebreak     (per_core_sim_ebreak[i]),
            .sim_last_wb_value (per_core_sim_last_wb_value[i]),

            .busy           (per_core_busy[i])
        );
    end
    
    assign busy = (| per_core_busy);

    VX_mem_req_if #(
        .DATA_WIDTH (`L2_MEM_DATA_WIDTH),
        .ADDR_WIDTH (`L2_MEM_ADDR_WIDTH),
        .TAG_WIDTH  (`L2X_MEM_TAG_WIDTH)
    ) l2_mem_req_if();
    
    VX_mem_rsp_if #(
        .DATA_WIDTH (`L2_MEM_DATA_WIDTH),
        .TAG_WIDTH  (`L2X_MEM_TAG_WIDTH)
    ) l2_mem_rsp_if();

`ifdef L2_ENABLE

`ifdef PERF_ENABLE
    VX_perf_cache_if perf_l2cache_if();
`endif

    `RESET_RELAY (l2_reset);

    VX_cache #(
        .CACHE_ID           (`L2_CACHE_ID),
        .CACHE_SIZE         (`L2_CACHE_SIZE),
        .CACHE_LINE_SIZE    (`L2_CACHE_LINE_SIZE),
        .NUM_BANKS          (`L2_NUM_BANKS),
        .NUM_WAYS           (`L2_NUM_WAYS),
        .NUM_PORTS          (`L2_NUM_PORTS),
        .WORD_SIZE          (`L2_WORD_SIZE),
        .NUM_REQS           (`L2_NUM_REQS),
        .CREQ_SIZE          (`L2_CREQ_SIZE),
        .CRSQ_SIZE          (`L2_CRSQ_SIZE),
        .MSHR_SIZE          (`L2_MSHR_SIZE),
        .MRSQ_SIZE          (`L2_MRSQ_SIZE),
        .MREQ_SIZE          (`L2_MREQ_SIZE),
        .WRITE_ENABLE       (1),       
        .REQ_UUID_BITS      (`UUID_BITS),   
        .CORE_TAG_WIDTH     (`L1_MEM_TAG_WIDTH),
        .MEM_TAG_WIDTH      (`L2X_MEM_TAG_WIDTH),
        .NC_ENABLE          (1)
    ) l2cache (
        `SCOPE_BIND_VX_cluster_l2cache
            
        .clk            (clk),
        .reset          (l2_reset),

    `ifdef PERF_ENABLE
        .perf_cache_if  (perf_l2cache_if),
    `endif  

        .core_req_if    (per_core_mem_req_if),
        .core_rsp_if    (per_core_mem_rsp_if),
        .mem_req_if     (l2_mem_req_if),
        .mem_rsp_if     (l2_mem_rsp_if)
    );

`else

    `RESET_RELAY (mem_arb_reset);

    VX_mem_mux #(
        .NUM_REQS     (`NUM_CORES),
        .DATA_WIDTH   (`DCACHE_MEM_DATA_WIDTH),
        .ADDR_WIDTH   (`DCACHE_MEM_ADDR_WIDTH),           
        .TAG_IN_WIDTH (`L1_MEM_TAG_WIDTH),
        .TAG_SEL_IDX  (1), // Skip 0 for NC flag
        .BUFFERED_REQ (1),
        .BUFFERED_RSP (1)
    ) mem_mux_core (
        .clk        (clk),
        .reset      (mem_arb_reset),
        .req_in_if  (per_core_mem_req_if),        
        .rsp_in_if  (per_core_mem_rsp_if),
        .req_out_if (l2_mem_req_if),
        .rsp_out_if (l2_mem_rsp_if)
    );

`endif

    localparam MEM_ARB_SIZE          = 1 + `EXT_RASTER_ENABLED + `EXT_ROP_ENABLED;
    localparam RCACHE_MEM_TAG_WIDTH_ = `EXT_RASTER_ENABLED ? `RCACHE_MEM_TAG_WIDTH : 0;
    localparam OCACHE_MEM_TAG_WIDTH_ = `EXT_ROP_ENABLED ? `OCACHE_MEM_TAG_WIDTH : 0;
    localparam _MEM_ARB_TAG_WIDTH    = `MAX(RCACHE_MEM_TAG_WIDTH_, OCACHE_MEM_TAG_WIDTH_);
    localparam MEM_ARB_TAG_WIDTH     = `MAX(_MEM_ARB_TAG_WIDTH, `L2X_MEM_TAG_WIDTH);

    VX_mem_req_if #(
        .DATA_WIDTH (`L2_MEM_DATA_WIDTH),
        .ADDR_WIDTH (`L2_MEM_ADDR_WIDTH),
        .TAG_WIDTH  (MEM_ARB_TAG_WIDTH)
    ) mem_req_arb_if[MEM_ARB_SIZE]();
    
    VX_mem_rsp_if #(
        .DATA_WIDTH (`L2_MEM_DATA_WIDTH),
        .TAG_WIDTH  (MEM_ARB_TAG_WIDTH)
    ) mem_rsp_arb_if[MEM_ARB_SIZE]();

    localparam RASTER_MEM_ARB_IDX = `EXT_RASTER_ENABLED;
    localparam ROP_MEM_ARB_IDX = 1 + `EXT_RASTER_ENABLED;
    `UNUSED_PARAM (RASTER_MEM_ARB_IDX)
    `UNUSED_PARAM (ROP_MEM_ARB_IDX)

    `ASSIGN_VX_MEM_REQ_IF_XTAG (mem_req_arb_if[0], l2_mem_req_if);
    assign mem_req_arb_if[0].tag = MEM_ARB_TAG_WIDTH'(l2_mem_req_if.tag);

    `ASSIGN_VX_MEM_RSP_IF_XTAG (l2_mem_rsp_if, mem_rsp_arb_if[0]);
    assign l2_mem_rsp_if.tag = `L2X_MEM_TAG_WIDTH'(mem_rsp_arb_if[0].tag);

`ifdef EXT_RASTER_ENABLE
    `ASSIGN_VX_MEM_REQ_IF_XTAG (mem_req_arb_if[RASTER_MEM_ARB_IDX], rcache_mem_req_if);
    assign mem_req_arb_if[RASTER_MEM_ARB_IDX].tag = MEM_ARB_TAG_WIDTH'(rcache_mem_req_if.tag);

    `ASSIGN_VX_MEM_RSP_IF_XTAG (rcache_mem_rsp_if, mem_rsp_arb_if[RASTER_MEM_ARB_IDX]);
    assign rcache_mem_rsp_if.tag = `RCACHE_MEM_TAG_WIDTH'(mem_rsp_arb_if[RASTER_MEM_ARB_IDX].tag);
`endif

`ifdef EXT_ROP_ENABLE
    `ASSIGN_VX_MEM_REQ_IF_XTAG (mem_req_arb_if[ROP_MEM_ARB_IDX], ocache_mem_req_if);
    assign mem_req_arb_if[ROP_MEM_ARB_IDX].tag = MEM_ARB_TAG_WIDTH'(ocache_mem_req_if.tag);

    `ASSIGN_VX_MEM_RSP_IF_XTAG (ocache_mem_rsp_if, mem_rsp_arb_if[ROP_MEM_ARB_IDX]);
    assign ocache_mem_rsp_if.tag = `OCACHE_MEM_TAG_WIDTH'(mem_rsp_arb_if[ROP_MEM_ARB_IDX].tag);
`endif

    VX_mem_mux #(
        .NUM_REQS     (MEM_ARB_SIZE),
        .DATA_WIDTH   (`L2_MEM_DATA_WIDTH),
        .ADDR_WIDTH   (`L2_MEM_ADDR_WIDTH),
        .TAG_IN_WIDTH (MEM_ARB_TAG_WIDTH),
        .TAG_SEL_IDX  (1), // Skip 0 for NC flag
        .BUFFERED_REQ (1),
        .BUFFERED_RSP (2)
    ) mem_mux_out (
        .clk        (clk),
        .reset      (reset),
        .req_in_if  (mem_req_arb_if),        
        .rsp_in_if  (mem_rsp_arb_if),
        .req_out_if (mem_req_if),
        .rsp_out_if (mem_rsp_if)
    );

endmodule
