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
    output wire [`NUM_REGS-1:0][31:0] sim_wb_value,

    // Status
    output wire             busy
);

`ifdef EXT_TEX_ENABLE

    VX_tex_req_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`TEX_REQ_TAG_WIDTH)
    ) per_core_tex_req_if[`NUM_CORES]();

    VX_tex_rsp_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`TEX_REQ_TAG_WIDTH)
    ) per_core_tex_rsp_if[`NUM_CORES]();

    VX_tex_req_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`TEX_REQX_TAG_WIDTH)
    ) tex_reqs_if[`NUM_TEX_UNITS]();

    VX_tex_rsp_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`TEX_REQX_TAG_WIDTH)
    ) tex_rsps_if[`NUM_TEX_UNITS]();

`ifdef PERF_ENABLE
    VX_perf_cache_if perf_tcache_if();
    VX_tex_perf_if tex_perfs_if[`NUM_TEX_UNITS]();
    VX_tex_perf_if tex_perf_if[`NUM_CORES]();
`endif

    VX_cache_req_if #(
        .NUM_REQS  (`TCACHE_NUM_REQS), 
        .WORD_SIZE (`TCACHE_WORD_SIZE), 
        .TAG_WIDTH (`TCACHE_TAG_WIDTH)
    ) tcache_req_if();

    VX_cache_rsp_if #(
        .NUM_REQS  (`TCACHE_NUM_REQS), 
        .WORD_SIZE (`TCACHE_WORD_SIZE), 
        .TAG_WIDTH (`TCACHE_TAG_WIDTH)
    ) tcache_rsp_if();

    VX_cache_req_if #(
        .NUM_REQS  (`TCACHE_NUM_REQS), 
        .WORD_SIZE (`TCACHE_WORD_SIZE), 
        .TAG_WIDTH (`TCACHE_TAG_ID_BITS)
    ) tcache_reqs_if[`NUM_TEX_UNITS]();

    VX_cache_rsp_if #(
        .NUM_REQS  (`TCACHE_NUM_REQS), 
        .WORD_SIZE (`TCACHE_WORD_SIZE), 
        .TAG_WIDTH (`TCACHE_TAG_ID_BITS)
    ) tcache_rsps_if[`NUM_TEX_UNITS]();

    // Generate all tex units
    for (genvar i = 0; i < `NUM_TEX_UNITS; ++i) begin
        `RESET_RELAY (tex_reset, reset);

        VX_tex_unit #(
            .INSTANCE_ID ($sformatf("cluster%0d-tex%0d", CLUSTER_ID, i)),
            .NUM_LANES   (`NUM_THREADS),
            .TAG_WIDTH   (`TEX_REQX_TAG_WIDTH)
        ) tex_unit (
            .clk           (clk),
            .reset         (tex_reset),
        `ifdef PERF_ENABLE
            .tex_perf_if   (tex_perfs_if[i]),
        `endif
            .tex_dcr_if    (tex_dcr_if),
            .tex_req_if    (tex_reqs_if[i]),
            .tex_rsp_if    (tex_rsps_if[i]),
            .cache_req_if  (tcache_reqs_if[i]),
            .cache_rsp_if  (tcache_rsps_if[i])
        );
    end

    VX_cache_arb #(
        .NUM_REQS     (`NUM_TEX_UNITS),
        .NUM_LANES    (`TCACHE_NUM_REQS),
        .DATA_SIZE    (`TCACHE_WORD_SIZE),
        .TAG_IN_WIDTH (`TCACHE_TAG_ID_BITS),
        .TAG_SEL_IDX  (0),
        .BUFFERED_REQ ((`NUM_TEX_UNITS > 1) ? 1 : 0),
        .BUFFERED_RSP ((`NUM_TEX_UNITS > 1) ? 1 : 0)
    ) tcache_arb (
        .clk        (clk),
        .reset      (reset),
        .req_in_if  (tcache_reqs_if),
        .rsp_in_if  (tcache_rsps_if),
        .req_out_if (tcache_req_if),
        .rsp_out_if (tcache_rsp_if)
    );

    VX_tex_arb #(
        .NUM_INPUTS   (`NUM_CORES),
        .NUM_LANES    (`NUM_THREADS),
        .NUM_OUTPUTS  (`NUM_TEX_UNITS),
        .TAG_WIDTH    (`TEX_REQ_TAG_WIDTH),
        .BUFFERED_REQ ((`NUM_CORES > 1 || `NUM_TEX_UNITS > 1) ? 1 : 0),
        .BUFFERED_RSP ((`NUM_CORES > 1 || `NUM_TEX_UNITS > 1) ? 1 : 0)
    ) tex_arb (
        .clk        (clk),
        .reset      (reset),
        .req_in_if  (per_core_tex_req_if),
        .rsp_in_if  (per_core_tex_rsp_if),
        .req_out_if (tex_reqs_if),
        .rsp_out_if (tex_rsps_if)
    );

    VX_mem_req_if #(
        .DATA_WIDTH (`TCACHE_MEM_DATA_WIDTH),
        .ADDR_WIDTH (`TCACHE_MEM_ADDR_WIDTH),
        .TAG_WIDTH  (`TCACHE_MEM_TAG_WIDTH)
    ) tcache_mem_req_if();
    
    VX_mem_rsp_if #(
        .DATA_WIDTH (`TCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (`TCACHE_MEM_TAG_WIDTH)
    ) tcache_mem_rsp_if();

    VX_mem_req_if #(
        .DATA_WIDTH (`TCACHE_WORD_SIZE*8), 
        .ADDR_WIDTH (`TCACHE_ADDR_WIDTH),
        .TAG_WIDTH  (`TCACHE_TAG_WIDTH)
    ) tcache_req_qual_if[`TCACHE_NUM_REQS]();

    VX_mem_rsp_if #(
        .DATA_WIDTH (`TCACHE_WORD_SIZE*8), 
        .TAG_WIDTH  (`TCACHE_TAG_WIDTH)
    ) tcache_rsp_qual_if[`TCACHE_NUM_REQS]();

    for (genvar i = 0; i < `TCACHE_NUM_REQS; ++i) begin
        `CACHE_REQ_TO_MEM(tcache_req_qual_if, tcache_req_if, i);
    end

    `RESET_RELAY (tcache_reset, reset);

    VX_cache_wrap #(
        .INSTANCE_ID    ($sformatf("cluster%0d-tcache", CLUSTER_ID)),
        .CACHE_SIZE     (`TCACHE_SIZE),
        .CACHE_LINE_SIZE(`TCACHE_LINE_SIZE),
        .NUM_BANKS      (`TCACHE_NUM_BANKS),
        .NUM_WAYS       (`TCACHE_NUM_WAYS),
        .NUM_PORTS      (`TCACHE_NUM_PORTS),
        .WORD_SIZE      (`TCACHE_WORD_SIZE),
        .NUM_REQS       (`TCACHE_NUM_REQS),
        .CREQ_SIZE      (`TCACHE_CREQ_SIZE),
        .CRSQ_SIZE      (`TCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`TCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`TCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`TCACHE_MREQ_SIZE),
        .WRITE_ENABLE   (0),
        .REQ_UUID_BITS  (0),
        .CORE_TAG_WIDTH (`TCACHE_TAG_WIDTH),
        .MEM_TAG_WIDTH  (`TCACHE_MEM_TAG_WIDTH),
        .NC_ENABLE      (0),
        .PASSTHRU       (!`TCACHE_ENABLED)
    ) tcache (
    `ifdef PERF_ENABLE
        .perf_cache_if  (perf_tcache_if),
    `endif
        
        .clk            (clk),
        .reset          (tcache_reset),
        .core_req_if    (tcache_req_qual_if),
        .core_rsp_if    (tcache_rsp_qual_if),
        .mem_req_if     (tcache_mem_req_if),
        .mem_rsp_if     (tcache_mem_rsp_if)
    );

    for (genvar i = 0; i < `TCACHE_NUM_REQS; ++i) begin
        `CACHE_RSP_FROM_MEM(tcache_rsp_if, tcache_rsp_qual_if, i);
    end
            
`endif

`ifdef EXT_RASTER_ENABLE

    VX_raster_req_if #(
        .NUM_LANES (`NUM_THREADS)
    ) per_core_raster_req_if[`NUM_CORES]();

    VX_raster_req_if #(
        .NUM_LANES (`NUM_THREADS)
    ) raster_reqs_if[`NUM_RASTER_UNITS]();

`ifdef PERF_ENABLE
    VX_perf_cache_if  perf_rcache_if();
    VX_raster_perf_if raster_perfs_if[`NUM_RASTER_UNITS]();
    VX_raster_perf_if raster_perf_if[`NUM_CORES]();
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

    VX_cache_req_if #(
        .NUM_REQS  (`RCACHE_NUM_REQS), 
        .WORD_SIZE (`RCACHE_WORD_SIZE), 
        .TAG_WIDTH (`RCACHE_TAG_ID_BITS)
    ) rcache_reqs_if[`NUM_RASTER_UNITS]();

    VX_cache_rsp_if #(
        .NUM_REQS  (`RCACHE_NUM_REQS), 
        .WORD_SIZE (`RCACHE_WORD_SIZE), 
        .TAG_WIDTH (`RCACHE_TAG_ID_BITS)
    ) rcache_rsps_if[`NUM_RASTER_UNITS]();

    // Generate all raster units
    for (genvar i = 0; i < `NUM_RASTER_UNITS; ++i) begin
        `RESET_RELAY (raster_reset, reset);

        VX_raster_unit #( 
            .INSTANCE_ID     ($sformatf("cluster%0d-raster%0d", CLUSTER_ID, i)),
            .INSTANCE_IDX    (CLUSTER_ID * `NUM_RASTER_UNITS + i),
            .NUM_INSTANCES   (`NUM_CLUSTERS * `NUM_RASTER_UNITS),
            .NUM_PES         (`RASTER_NUM_PES),
            .TILE_LOGSIZE    (`RASTER_TILE_LOGSIZE),
            .BLOCK_LOGSIZE   (`RASTER_BLOCK_LOGSIZE),
            .MEM_FIFO_DEPTH  (`RASTER_MEM_FIFO_DEPTH),
            .QUAD_FIFO_DEPTH (`RASTER_QUAD_FIFO_DEPTH),
            .OUTPUT_QUADS    (`NUM_THREADS)
        ) raster_unit (
            .clk           (clk),
            .reset         (raster_reset),
        `ifdef PERF_ENABLE
            .raster_perf_if(raster_perfs_if[i]),
        `endif
            .raster_dcr_if (raster_dcr_if),
            .raster_req_if (raster_reqs_if[i]),
            .cache_req_if  (rcache_reqs_if[i]),
            .cache_rsp_if  (rcache_rsps_if[i])
        );
    end

    VX_cache_arb #(
        .NUM_REQS     (`NUM_RASTER_UNITS),
        .NUM_LANES    (`RCACHE_NUM_REQS),
        .DATA_SIZE    (`RCACHE_WORD_SIZE),
        .TAG_IN_WIDTH (`RCACHE_TAG_ID_BITS),
        .TAG_SEL_IDX  (0),
        .BUFFERED_REQ ((`NUM_RASTER_UNITS > 1) ? 1 : 0),
        .BUFFERED_RSP ((`NUM_RASTER_UNITS > 1) ? 1 : 0)
    ) rcache_arb (
        .clk        (clk),
        .reset      (reset),
        .req_in_if  (rcache_reqs_if),
        .rsp_in_if  (rcache_rsps_if),
        .req_out_if (rcache_req_if),
        .rsp_out_if (rcache_rsp_if)
    );

    VX_raster_arb #(
        .NUM_INPUTS  (`NUM_RASTER_UNITS),
        .NUM_LANES   (`NUM_THREADS),
        .NUM_OUTPUTS (`NUM_CORES),
        .BUFFERED    ((`NUM_CORES > 1 || `NUM_RASTER_UNITS > 1) ? 1 : 0)
    ) raster_arb (
        .clk        (clk),
        .reset      (reset),
        .req_in_if  (raster_reqs_if),
        .req_out_if (per_core_raster_req_if)
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

    `RESET_RELAY (rcache_reset, reset);

    VX_cache_wrap #(
        .INSTANCE_ID    ($sformatf("cluster%0d-rcache", CLUSTER_ID)),
        .CACHE_SIZE     (`RCACHE_SIZE),
        .CACHE_LINE_SIZE(`RCACHE_LINE_SIZE),
        .NUM_BANKS      (`RCACHE_NUM_BANKS),
        .NUM_WAYS       (`RCACHE_NUM_WAYS),
        .NUM_PORTS      (`RCACHE_NUM_PORTS),
        .WORD_SIZE      (`RCACHE_WORD_SIZE),
        .NUM_REQS       (`RCACHE_NUM_REQS),
        .CREQ_SIZE      (`RCACHE_CREQ_SIZE),
        .CRSQ_SIZE      (`RCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`RCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`RCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`RCACHE_MREQ_SIZE),
        .WRITE_ENABLE   (0),
        .REQ_UUID_BITS  (0),
        .CORE_TAG_WIDTH (`RCACHE_TAG_WIDTH),
        .MEM_TAG_WIDTH  (`RCACHE_MEM_TAG_WIDTH),
        .NC_ENABLE      (0),
        .PASSTHRU       (!`RCACHE_ENABLED)
    ) rcache (
    `ifdef PERF_ENABLE
        .perf_cache_if  (perf_rcache_if),
    `endif
        
        .clk            (clk),
        .reset          (rcache_reset),
        .core_req_if    (rcache_req_qual_if),
        .core_rsp_if    (rcache_rsp_qual_if),
        .mem_req_if     (rcache_mem_req_if),
        .mem_rsp_if     (rcache_mem_rsp_if)
    );

    for (genvar i = 0; i < `RCACHE_NUM_REQS; ++i) begin
        `CACHE_RSP_FROM_MEM(rcache_rsp_if, rcache_rsp_qual_if, i);
    end

`endif

`ifdef EXT_ROP_ENABLE

    VX_rop_req_if #(
        .NUM_LANES (`NUM_THREADS)
    ) per_core_rop_req_if[`NUM_CORES]();

    VX_rop_req_if #(
        .NUM_LANES (`NUM_THREADS)
    ) rop_reqs_if[`NUM_ROP_UNITS]();

`ifdef PERF_ENABLE
    VX_perf_cache_if perf_ocache_if();
    VX_rop_perf_if   rop_perfs_if[`NUM_ROP_UNITS]();
    VX_rop_perf_if   rop_perf_if[`NUM_CORES]();
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

    VX_cache_req_if #(
        .NUM_REQS  (`OCACHE_NUM_REQS), 
        .WORD_SIZE (`OCACHE_WORD_SIZE), 
        .TAG_WIDTH (`OCACHE_TAG_ID_BITS)
    ) ocache_reqs_if[`NUM_ROP_UNITS]();

    VX_cache_rsp_if #(
        .NUM_REQS  (`OCACHE_NUM_REQS), 
        .WORD_SIZE (`OCACHE_WORD_SIZE), 
        .TAG_WIDTH (`OCACHE_TAG_ID_BITS)
    ) ocache_rsps_if[`NUM_ROP_UNITS]();

    // Generate all rop units
    for (genvar i = 0; i < `NUM_ROP_UNITS; ++i) begin
        `RESET_RELAY (rop_reset, reset);

        VX_rop_unit #(
            .INSTANCE_ID ($sformatf("cluster%0d-rop%0d", CLUSTER_ID, i)),
            .NUM_LANES   (`NUM_THREADS)
        ) rop_unit (
            .clk           (clk),
            .reset         (rop_reset),
        `ifdef PERF_ENABLE
            .rop_perf_if   (rop_perfs_if[i]),
        `endif
            .rop_req_if    (rop_reqs_if[i]),
            .rop_dcr_if    (rop_dcr_if),
            .cache_req_if  (ocache_reqs_if[i]),
            .cache_rsp_if  (ocache_rsps_if[i])
        );
    end
    
    VX_cache_arb #(
        .NUM_REQS     (`NUM_ROP_UNITS),
        .NUM_LANES    (`OCACHE_NUM_REQS),
        .DATA_SIZE    (`OCACHE_WORD_SIZE),
        .TAG_IN_WIDTH (`OCACHE_TAG_ID_BITS),
        .TAG_SEL_IDX  (0),
        .BUFFERED_REQ ((`NUM_ROP_UNITS > 1) ? 1 : 0),
        .BUFFERED_RSP ((`NUM_ROP_UNITS > 1) ? 1 : 0)
    ) ocache_arb (
        .clk        (clk),
        .reset      (reset),
        .req_in_if  (ocache_reqs_if),
        .rsp_in_if  (ocache_rsps_if),
        .req_out_if (ocache_req_if),
        .rsp_out_if (ocache_rsp_if)
    );

    VX_rop_arb #(
        .NUM_INPUTS  (`NUM_CORES),
        .NUM_LANES   (`NUM_THREADS),
        .NUM_OUTPUTS (`NUM_ROP_UNITS),
        .BUFFERED    ((`NUM_CORES > 1 || `NUM_ROP_UNITS > 1) ? 1 : 0)
    ) rop_arb (
        .clk        (clk),
        .reset      (reset),
        .req_in_if  (per_core_rop_req_if),
        .req_out_if (rop_reqs_if)
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

    `RESET_RELAY (ocache_reset, reset);

    VX_cache_wrap #(
        .INSTANCE_ID    ($sformatf("cluster%0d-ocache", CLUSTER_ID)),
        .CACHE_SIZE     (`OCACHE_SIZE),
        .CACHE_LINE_SIZE(`OCACHE_LINE_SIZE),
        .NUM_BANKS      (`OCACHE_NUM_BANKS),
        .NUM_WAYS       (`OCACHE_NUM_WAYS),
        .NUM_PORTS      (`OCACHE_NUM_PORTS),
        .WORD_SIZE      (`OCACHE_WORD_SIZE),
        .NUM_REQS       (`OCACHE_NUM_REQS),
        .CREQ_SIZE      (`OCACHE_CREQ_SIZE),
        .CRSQ_SIZE      (`OCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`OCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`OCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`OCACHE_MREQ_SIZE),
        .WRITE_ENABLE   (1),
        .REQ_UUID_BITS  (0),
        .CORE_TAG_WIDTH (`OCACHE_TAG_WIDTH),
        .MEM_TAG_WIDTH  (`OCACHE_MEM_TAG_WIDTH),
        .NC_ENABLE      (0),
        .PASSTHRU       (!`OCACHE_ENABLED)
    ) ocache (
    `ifdef PERF_ENABLE
        .perf_cache_if  (perf_ocache_if),
    `endif
        
        .clk            (clk),
        .reset          (ocache_reset),

        .core_req_if    (ocache_req_qual_if),
        .core_rsp_if    (ocache_rsp_qual_if),
        .mem_req_if     (ocache_mem_req_if),
        .mem_rsp_if     (ocache_mem_rsp_if)
    );

    for (genvar i = 0; i < `OCACHE_NUM_REQS; ++i) begin
        `CACHE_RSP_FROM_MEM(ocache_rsp_if, ocache_rsp_qual_if, i);
    end

`endif

    wire [`NUM_CORES-1:0] per_core_sim_ebreak;
    wire [`NUM_CORES-1:0][`NUM_REGS-1:0][31:0] per_core_sim_wb_value;
    assign sim_ebreak = per_core_sim_ebreak[0];
    assign sim_wb_value = per_core_sim_wb_value[0];
    `UNUSED_VAR (per_core_sim_ebreak)
    `UNUSED_VAR (per_core_sim_wb_value)

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

    // Generate all cores
    for (genvar i = 0; i < `NUM_CORES; ++i) begin

        `RESET_RELAY (core_reset, reset);

        VX_core #(
            .CORE_ID ((CLUSTER_ID * `NUM_CORES) + i)
        ) core (
            `SCOPE_BIND_VX_cluster_core(i)

            .clk            (clk),
            .reset          (core_reset),

            .dcr_base_if    (dcr_base_if),

        `ifdef EXT_TEX_ENABLE
            .tex_req_if     (per_core_tex_req_if[i]),
            .tex_rsp_if     (per_core_tex_rsp_if[i]),
        `ifdef PERF_ENABLE
            .tex_perf_if    (tex_perf_if[0]),
            .perf_tcache_if (perf_tcache_if),
        `endif
        `endif

        `ifdef EXT_RASTER_ENABLE        
            .raster_req_if  (per_core_raster_req_if[i]),
        `ifdef PERF_ENABLE
            .raster_perf_if (raster_perf_if[0]),
            .perf_rcache_if (perf_rcache_if),
        `endif
        `endif
        
        `ifdef EXT_ROP_ENABLE        
            .rop_req_if     (per_core_rop_req_if[i]),
        `ifdef PERF_ENABLE
            .rop_perf_if    (rop_perf_if[0]),
            .perf_ocache_if (perf_ocache_if),
        `endif
        `endif

            .mem_req_if     (per_core_mem_req_if[i]),
            .mem_rsp_if     (per_core_mem_rsp_if[i]),

            .sim_ebreak     (per_core_sim_ebreak[i]),
            .sim_wb_value   (per_core_sim_wb_value[i]),

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

`ifdef PERF_ENABLE
    VX_perf_cache_if perf_l2cache_if();
`endif

    `RESET_RELAY (l2_reset, reset);

    VX_cache_wrap #(
        .INSTANCE_ID    (`L2_CACHE_ID),
        .CACHE_SIZE     (`L2_CACHE_SIZE),
        .CACHE_LINE_SIZE(`L2_CACHE_LINE_SIZE),
        .NUM_BANKS      (`L2_NUM_BANKS),
        .NUM_WAYS       (`L2_NUM_WAYS),
        .NUM_PORTS      (`L2_NUM_PORTS),
        .WORD_SIZE      (`L2_WORD_SIZE),
        .NUM_REQS       (`L2_NUM_REQS),
        .CREQ_SIZE      (`L2_CREQ_SIZE),
        .CRSQ_SIZE      (`L2_CRSQ_SIZE),
        .MSHR_SIZE      (`L2_MSHR_SIZE),
        .MRSQ_SIZE      (`L2_MRSQ_SIZE),
        .MREQ_SIZE      (`L2_MREQ_SIZE),
        .WRITE_ENABLE   (1),       
        .REQ_UUID_BITS  (`UUID_BITS),   
        .CORE_TAG_WIDTH (`L1_MEM_TAG_WIDTH),
        .MEM_TAG_WIDTH  (`L2X_MEM_TAG_WIDTH),
        .NC_ENABLE      (1),
        .PASSTHRU       (!`L2_ENABLED)
    ) l2cache (            
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

    localparam MEM_ARB_SIZE          = 1 + `EXT_TEX_ENABLED + `EXT_RASTER_ENABLED + `EXT_ROP_ENABLED;
    localparam _TCACHE_MEM_TAG_WIDTH = `EXT_TEX_ENABLED ? `TCACHE_MEM_TAG_WIDTH : 0;
    localparam _RCACHE_MEM_TAG_WIDTH = `EXT_RASTER_ENABLED ? `RCACHE_MEM_TAG_WIDTH : 0;
    localparam _OCACHE_MEM_TAG_WIDTH = `EXT_ROP_ENABLED ? `OCACHE_MEM_TAG_WIDTH : 0;
    localparam _MEM_ARB_TAG_WIDTH    = `MAX(_RCACHE_MEM_TAG_WIDTH, _OCACHE_MEM_TAG_WIDTH);
    localparam __MEM_ARB_TAG_WIDTH   = `MAX(_MEM_ARB_TAG_WIDTH, _TCACHE_MEM_TAG_WIDTH);
    localparam MEM_ARB_TAG_WIDTH     = `MAX(__MEM_ARB_TAG_WIDTH, `L2X_MEM_TAG_WIDTH);

    VX_mem_req_if #(
        .DATA_WIDTH (`L2_MEM_DATA_WIDTH),
        .ADDR_WIDTH (`L2_MEM_ADDR_WIDTH),
        .TAG_WIDTH  (MEM_ARB_TAG_WIDTH)
    ) mem_req_arb_if[MEM_ARB_SIZE]();
    
    VX_mem_rsp_if #(
        .DATA_WIDTH (`L2_MEM_DATA_WIDTH),
        .TAG_WIDTH  (MEM_ARB_TAG_WIDTH)
    ) mem_rsp_arb_if[MEM_ARB_SIZE]();

    localparam TEX_MEM_ARB_IDX    = 1;
    localparam RASTER_MEM_ARB_IDX = TEX_MEM_ARB_IDX + `EXT_TEX_ENABLED;
    localparam ROP_MEM_ARB_IDX    = RASTER_MEM_ARB_IDX + `EXT_RASTER_ENABLED;
    `UNUSED_PARAM (TEX_MEM_ARB_IDX)
    `UNUSED_PARAM (RASTER_MEM_ARB_IDX)
    `UNUSED_PARAM (ROP_MEM_ARB_IDX)

    `ASSIGN_VX_MEM_REQ_IF_XTAG (mem_req_arb_if[0], l2_mem_req_if);
    assign mem_req_arb_if[0].tag = MEM_ARB_TAG_WIDTH'(l2_mem_req_if.tag);

    `ASSIGN_VX_MEM_RSP_IF_XTAG (l2_mem_rsp_if, mem_rsp_arb_if[0]);
    assign l2_mem_rsp_if.tag = `L2X_MEM_TAG_WIDTH'(mem_rsp_arb_if[0].tag);

`ifdef EXT_TEX_ENABLE
    `ASSIGN_VX_MEM_REQ_IF_XTAG (mem_req_arb_if[TEX_MEM_ARB_IDX], tcache_mem_req_if);
    assign mem_req_arb_if[TEX_MEM_ARB_IDX].tag = MEM_ARB_TAG_WIDTH'(tcache_mem_req_if.tag);

    `ASSIGN_VX_MEM_RSP_IF_XTAG (tcache_mem_rsp_if, mem_rsp_arb_if[TEX_MEM_ARB_IDX]);
    assign tcache_mem_rsp_if.tag = `TCACHE_MEM_TAG_WIDTH'(mem_rsp_arb_if[TEX_MEM_ARB_IDX].tag);
`endif

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

    VX_mem_arb #(
        .NUM_REQS     (MEM_ARB_SIZE),
        .DATA_WIDTH   (`L2_MEM_DATA_WIDTH),
        .ADDR_WIDTH   (`L2_MEM_ADDR_WIDTH),
        .TAG_IN_WIDTH (MEM_ARB_TAG_WIDTH),
        .TAG_SEL_IDX  (1), // Skip 0 for NC flag
        .BUFFERED_REQ ((MEM_ARB_SIZE > 1) ? 1 : 0),
        .BUFFERED_RSP ((MEM_ARB_SIZE > 1) ? 2 : 0)
    ) mem_arb_out (
        .clk        (clk),
        .reset      (reset),
        .req_in_if  (mem_req_arb_if),        
        .rsp_in_if  (mem_rsp_arb_if),
        .req_out_if (mem_req_if),
        .rsp_out_if (mem_rsp_if)
    );

endmodule
