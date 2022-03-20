`include "VX_define.vh"

module VX_cluster #(
    parameter CLUSTER_ID = 0
) ( 
    `SCOPE_IO_VX_cluster

    // Clock
    input  wire             clk,
    input  wire             reset,

`ifdef EXT_TEX_ENABLE
    VX_tex_dcr_if.master    tex_dcr_if,
`endif
`ifdef EXT_RASTER_ENABLE
    VX_raster_dcr_if.master raster_dcr_if,
`endif
`ifdef EXT_ROP_ENABLE
    VX_rop_dcr_if.master    rop_dcr_if,
`endif

    // Memory
    VX_mem_req_if.master    mem_req_if,
    VX_mem_rsp_if.slave     mem_rsp_if,

    // Status
    output wire             busy
); 
    `STATIC_ASSERT((`L2_ENABLE == 0 || `NUM_CORES > 1), ("invalid parameter"))

`ifdef EXT_RASTER_ENABLE

    VX_raster_req_if    per_core_raster_req_if[`NUM_CORES-1:0]();
    VX_raster_req_if    raster_req_if();

`ifdef PERF_ENABLE
    VX_raster_perf_if   raster_perf_if();
`endif

    VX_dcache_req_if    raster_cache_req_if();
    VX_dcache_rsp_if    raster_cache_rsp_if();

`ifdef PERF_ENABLE
    // TODO: remove
    `UNUSED_VAR (raster_perf_if.mem_reads)
    `UNUSED_VAR (raster_perf_if.mem_latency)
`endif

    // TODO: remove
    `UNUSED_VAR (raster_cache_req_if.valid)
    `UNUSED_VAR (raster_cache_req_if.rw)
    `UNUSED_VAR (raster_cache_req_if.byteen)
    `UNUSED_VAR (raster_cache_req_if.addr)
    `UNUSED_VAR (raster_cache_req_if.data)  
    `UNUSED_VAR (raster_cache_req_if.tag)
    assign raster_cache_req_if.ready = 0;

    // TODO: remove
    assign raster_cache_rsp_if.valid = 0;
    assign raster_cache_rsp_if.tmask = 0;
    assign raster_cache_rsp_if.data = 0;     
    assign raster_cache_rsp_if.tag = 0;
    `UNUSED_VAR (raster_cache_rsp_if.ready)

    `RESET_RELAY (raster_reset);

    VX_raster_unit #(
        .CLUSTER_ID  (CLUSTER_ID),
        .NUM_SLICES  (1),
        .NUM_OUTPUTS (`NUM_THREADS)
    ) raster_unit (
        .clk           (clk),
        .reset         (raster_reset),
    `ifdef PERF_ENABLE
        .raster_perf_if(raster_perf_if),
    `endif
        .raster_req_if (raster_req_if),
        .raster_dcr_if (raster_dcr_if),        
        .cache_req_if  (raster_cache_req_if),
        .cache_rsp_if  (raster_cache_rsp_if)
    );

    VX_raster_req_arb #(
        .NUM_REQS (`NUM_CORES)
    ) raster_req_arb (
        .clk     (clk),
        .reset   (raster_reset),
        .req_in  (per_core_raster_req_if),
        .req_out (raster_req_if)
    );

`endif

`ifdef EXT_ROP_ENABLE

    VX_rop_req_if       per_core_rop_req_if[`NUM_CORES-1:0]();
    VX_rop_req_if       rop_req_if();

`ifdef PERF_ENABLE
    VX_rop_perf_if      rop_perf_if(); 
`endif
    
    VX_dcache_req_if    rop_cache_req_if();
    VX_dcache_rsp_if    rop_cache_rsp_if();

`ifdef PERF_ENABLE
    // TODO: remove
    `UNUSED_VAR (rop_perf_if.mem_reads)
    `UNUSED_VAR (rop_perf_if.mem_writes)
    `UNUSED_VAR (rop_perf_if.mem_latency)
`endif 

    // TODO: remove
    `UNUSED_VAR (rop_cache_req_if.valid)
    `UNUSED_VAR (rop_cache_req_if.rw)
    `UNUSED_VAR (rop_cache_req_if.byteen)
    `UNUSED_VAR (rop_cache_req_if.addr)
    `UNUSED_VAR (rop_cache_req_if.data) 
    `UNUSED_VAR (rop_cache_req_if.tag)
    assign rop_cache_req_if.ready = 0;

    // TODO: remove
    assign rop_cache_rsp_if.valid = 0;
    assign rop_cache_rsp_if.tmask = 0;
    assign rop_cache_rsp_if.data = 0;     
    assign rop_cache_rsp_if.tag = 0;
    `UNUSED_VAR (rop_cache_rsp_if.ready)

    `RESET_RELAY (rop_reset);

    VX_rop_unit #(
        .CLUSTER_ID (CLUSTER_ID),
        .NUM_SLICES (`NUM_THREADS)
    ) rop_unit (
        .clk           (clk),
        .reset         (rop_reset),
    `ifdef PERF_ENABLE
        .rop_perf_if   (rop_perf_if),
    `endif
        .rop_req_if    (rop_req_if),
        .rop_dcr_if    (rop_dcr_if),
        .cache_req_if  (rop_cache_req_if),
        .cache_rsp_if  (rop_cache_rsp_if)
    );

    VX_rop_req_arb #(
        .NUM_REQS (`NUM_CORES)
    ) rop_req_arb (
        .clk     (clk),
        .reset   (rop_reset),
        .req_in  (per_core_rop_req_if),
        .req_out (rop_req_if)
    );

`endif

    VX_mem_req_if #(
        .DATA_WIDTH (`DCACHE_MEM_DATA_WIDTH),
        .ADDR_WIDTH (`DCACHE_MEM_ADDR_WIDTH),
        .TAG_WIDTH  (`L1_MEM_TAG_WIDTH)
    ) per_core_mem_req_if[`NUM_CORES-1:0]();
    
    VX_mem_rsp_if #(
        .DATA_WIDTH (`DCACHE_MEM_DATA_WIDTH),
        .TAG_WIDTH  (`L1_MEM_TAG_WIDTH)
    ) per_core_mem_rsp_if[`NUM_CORES-1:0]();

    wire [`NUM_CORES-1:0] per_core_busy;

    for (genvar i = 0; i < `NUM_CORES; i++) begin

        `RESET_RELAY (core_reset);

        VX_core #(
            .CORE_ID(i + (CLUSTER_ID * `NUM_CORES))
        ) core (
            `SCOPE_BIND_VX_cluster_core(i)

            .clk            (clk),
            .reset          (core_reset),

        `ifdef EXT_TEX_ENABLE
            .tex_dcr_if     (tex_dcr_if),
        `endif
        `ifdef EXT_RASTER_ENABLE        
            .raster_req_if  (per_core_raster_req_if[i]),
        `endif
        `ifdef EXT_RASTER_ENABLE        
            .rop_req_if     (per_core_rop_req_if[i]),
        `endif

            .mem_req_if     (per_core_mem_req_if[i]),
            .mem_rsp_if     (per_core_mem_rsp_if[i]),

            .busy           (per_core_busy[i])
        );
    end
    
    assign busy = (| per_core_busy);

    if (`L2_ENABLE) begin
    `ifdef PERF_ENABLE
        VX_perf_cache_if perf_l2cache_if();
    `endif

        `RESET_RELAY (l2_reset);

        VX_cache #(
            .CACHE_ID           (`L2_CACHE_ID),
            .CACHE_SIZE         (`L2_CACHE_SIZE),
            .CACHE_LINE_SIZE    (`L2_CACHE_LINE_SIZE),
            .NUM_BANKS          (`L2_NUM_BANKS),
            .NUM_PORTS          (`L2_NUM_PORTS),
            .WORD_SIZE          (`L2_WORD_SIZE),
            .NUM_REQS           (`L2_NUM_REQS),
            .CREQ_SIZE          (`L2_CREQ_SIZE),
            .CRSQ_SIZE          (`L2_CRSQ_SIZE),
            .MSHR_SIZE          (`L2_MSHR_SIZE),
            .MRSQ_SIZE          (`L2_MRSQ_SIZE),
            .MREQ_SIZE          (`L2_MREQ_SIZE),
            .WRITE_ENABLE       (1),          
            .CORE_TAG_WIDTH     (`L1_MEM_TAG_WIDTH),
            .CORE_TAG_ID_BITS   (0),
            .MEM_TAG_WIDTH      (`L2_MEM_TAG_WIDTH),
            .NC_ENABLE          (1)
        ) l2cache (
            `SCOPE_BIND_VX_cluster_l2cache
              
            .clk                (clk),
            .reset              (l2_reset),

        `ifdef PERF_ENABLE
            .perf_cache_if      (perf_l2cache_if),
        `endif

            // Core request
            .core_req_valid     (per_core_mem_req_if.valid),
            .core_req_rw        (per_core_mem_req_if.rw),
            .core_req_byteen    (per_core_mem_req_if.byteen),
            .core_req_addr      (per_core_mem_req_if.addr),
            .core_req_data      (per_core_mem_req_if.data),  
            .core_req_tag       (per_core_mem_req_if.tag),  
            .core_req_ready     (per_core_mem_req_if.ready),

            // Core response
            .core_rsp_valid     (per_core_mem_rsp_if.valid),
            .core_rsp_data      (per_core_mem_rsp_if.data),
            .core_rsp_tag       (per_core_mem_rsp_if.tag),
            .core_rsp_ready     (per_core_mem_rsp_if.ready),
            `UNUSED_PIN (core_rsp_tmask),

            // Memory request
            .mem_req_valid      (mem_req_if.valid),
            .mem_req_rw         (mem_req_if.rw),        
            .mem_req_byteen     (mem_req_if.byteen),
            .mem_req_addr       (mem_req_if.addr),
            .mem_req_data       (mem_req_if.data),
            .mem_req_tag        (mem_req_if.tag),
            .mem_req_ready      (mem_req_if.ready),
            
            // Memory response
            .mem_rsp_valid      (mem_rsp_if.valid),
            .mem_rsp_tag        (mem_rsp_if.tag),
            .mem_rsp_data       (mem_rsp_if.data),
            .mem_rsp_ready      (mem_rsp_if.ready)
        );

    end else begin

        `RESET_RELAY (mem_arb_reset);

        VX_mem_arb #(
            .NUM_REQS     (`NUM_CORES),
            .DATA_WIDTH   (`DCACHE_MEM_DATA_WIDTH),
            .ADDR_WIDTH   (`DCACHE_MEM_ADDR_WIDTH),           
            .TAG_IN_WIDTH (`L1_MEM_TAG_WIDTH),            
            .TYPE         ("R"),
            .TAG_SEL_IDX  (1), // Skip 0 for NC flag
            .BUFFERED_REQ (1),
            .BUFFERED_RSP (1)
        ) mem_arb (
            .clk        (clk),
            .reset      (mem_arb_reset),
            .req_in_if  (per_core_mem_req_if),
            .req_out_if (mem_req_if),
            .rsp_out_if (per_core_mem_rsp_if),
            .rsp_in_if  (mem_rsp_if)
        );

    end

endmodule
