`include "VX_define.vh"
`include "VX_cache_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_cache_types::*;
`IGNORE_WARNINGS_END

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
        .TAG_WIDTH (`TEX_REQ_ARB_TAG_WIDTH)
    ) tex_req_if[`NUM_TEX_UNITS]();

    VX_tex_rsp_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`TEX_REQ_ARB_TAG_WIDTH)
    ) tex_rsp_if[`NUM_TEX_UNITS]();

    VX_tex_arb #(
        .NUM_INPUTS   (`NUM_CORES),
        .NUM_LANES    (`NUM_THREADS),
        .NUM_OUTPUTS  (`NUM_TEX_UNITS),
        .TAG_WIDTH    (`TEX_REQ_TAG_WIDTH),
        .BUFFERED_REQ ((`NUM_CORES != `NUM_TEX_UNITS) ? 1 : 0),
        .BUFFERED_RSP ((`NUM_CORES != `NUM_TEX_UNITS) ? 1 : 0)
    ) tex_arb (
        .clk        (clk),
        .reset      (reset),
        .req_in_if  (per_core_tex_req_if),
        .rsp_in_if  (per_core_tex_rsp_if),
        .req_out_if (tex_req_if),
        .rsp_out_if (tex_rsp_if)
    );

    VX_cache_req_if #(
        .NUM_REQS  (TCACHE_NUM_REQS), 
        .WORD_SIZE (TCACHE_WORD_SIZE), 
        .TAG_WIDTH (TCACHE_TAG_WIDTH)
    ) tcache_req_if[`NUM_TEX_UNITS]();

    VX_cache_rsp_if #(
        .NUM_REQS  (TCACHE_NUM_REQS), 
        .WORD_SIZE (TCACHE_WORD_SIZE), 
        .TAG_WIDTH (TCACHE_TAG_WIDTH)
    ) tcache_rsp_if[`NUM_TEX_UNITS]();

`ifdef PERF_ENABLE
    VX_perf_cache_if perf_tcache_if();
    VX_tex_perf_if tex_perf_if[`NUM_TEX_UNITS]();
`endif

    // Generate all tex units
    for (genvar i = 0; i < `NUM_TEX_UNITS; ++i) begin
        `RESET_RELAY (tex_reset, reset);

        VX_tex_unit #(
            .INSTANCE_ID ($sformatf("cluster%0d-tex%0d", CLUSTER_ID, i)),
            .NUM_LANES   (`NUM_THREADS),
            .TAG_WIDTH   (`TEX_REQ_ARB_TAG_WIDTH)
        ) tex_unit (
            .clk           (clk),
            .reset         (tex_reset),
        `ifdef PERF_ENABLE
            .tex_perf_if   (tex_perf_if[i]),
        `endif
            .tex_dcr_if    (tex_dcr_if),
            .tex_req_if    (tex_req_if[i]),
            .tex_rsp_if    (tex_rsp_if[i]),
            .cache_req_if  (tcache_req_if[i]),
            .cache_rsp_if  (tcache_rsp_if[i])
        );
    end    
            
`endif

`ifdef EXT_RASTER_ENABLE

    VX_raster_req_if #(
        .NUM_LANES (`NUM_THREADS)
    ) raster_req_if[`NUM_RASTER_UNITS]();

    VX_cache_req_if #(
        .NUM_REQS  (RCACHE_NUM_REQS), 
        .WORD_SIZE (RCACHE_WORD_SIZE), 
        .TAG_WIDTH (RCACHE_TAG_WIDTH)
    ) rcache_req_if[`NUM_RASTER_UNITS]();

    VX_cache_rsp_if #(
        .NUM_REQS  (RCACHE_NUM_REQS), 
        .WORD_SIZE (RCACHE_WORD_SIZE), 
        .TAG_WIDTH (RCACHE_TAG_WIDTH)
    ) rcache_rsp_if[`NUM_RASTER_UNITS]();

`ifdef PERF_ENABLE
    VX_perf_cache_if  perf_rcache_if();
    VX_raster_perf_if raster_perf_if[`NUM_RASTER_UNITS]();
`endif

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
            .raster_perf_if(raster_perf_if[i]),
        `endif
            .raster_dcr_if (raster_dcr_if),
            .raster_req_if (raster_req_if[i]),
            .cache_req_if  (rcache_req_if[i]),
            .cache_rsp_if  (rcache_rsp_if[i])
        );
    end

    VX_raster_req_if #(
        .NUM_LANES (`NUM_THREADS)
    ) per_core_raster_req_if[`NUM_CORES]();

    VX_raster_arb #(
        .NUM_INPUTS  (`NUM_RASTER_UNITS),
        .NUM_LANES   (`NUM_THREADS),
        .NUM_OUTPUTS (`NUM_CORES),
        .BUFFERED    ((`NUM_CORES != `NUM_RASTER_UNITS) ? 1 : 0)
    ) raster_arb (
        .clk        (clk),
        .reset      (reset),
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
    ) rop_req_if[`NUM_ROP_UNITS]();

    VX_rop_arb #(
        .NUM_INPUTS  (`NUM_CORES),
        .NUM_LANES   (`NUM_THREADS),
        .NUM_OUTPUTS (`NUM_ROP_UNITS),
        .BUFFERED    ((`NUM_CORES != `NUM_ROP_UNITS) ? 1 : 0)
    ) rop_arb (
        .clk        (clk),
        .reset      (reset),
        .req_in_if  (per_core_rop_req_if),
        .req_out_if (rop_req_if)
    );

    VX_cache_req_if #(
        .NUM_REQS  (OCACHE_NUM_REQS), 
        .WORD_SIZE (OCACHE_WORD_SIZE), 
        .TAG_WIDTH (OCACHE_TAG_WIDTH)
    ) ocache_req_if[`NUM_ROP_UNITS]();

    VX_cache_rsp_if #(
        .NUM_REQS  (OCACHE_NUM_REQS), 
        .WORD_SIZE (OCACHE_WORD_SIZE), 
        .TAG_WIDTH (OCACHE_TAG_WIDTH)
    ) ocache_rsp_if[`NUM_ROP_UNITS]();

`ifdef PERF_ENABLE
    VX_perf_cache_if perf_ocache_if();
    VX_rop_perf_if rop_perf_if[`NUM_ROP_UNITS]();
`endif

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
            .rop_perf_if   (rop_perf_if[i]),
        `endif
            .rop_dcr_if    (rop_dcr_if),
            .rop_req_if    (rop_req_if[i]),            
            .cache_req_if  (ocache_req_if[i]),
            .cache_rsp_if  (ocache_rsp_if[i])
        );
    end

`endif

`ifdef EXT_F_ENABLE

    VX_fpu_req_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`FPU_REQ_TAG_WIDTH)
    ) per_core_fpu_req_if[`NUM_CORES]();

    VX_fpu_rsp_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`FPU_REQ_TAG_WIDTH)
    ) per_core_fpu_rsp_if[`NUM_CORES]();

    VX_fpu_req_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`FPU_REQ_ARB_TAG_WIDTH)
    ) fpu_req_if[`NUM_FPU_UNITS]();

    VX_fpu_rsp_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`FPU_REQ_ARB_TAG_WIDTH)
    ) fpu_rsp_if[`NUM_FPU_UNITS]();

    VX_fpu_arb #(
        .NUM_INPUTS   (`NUM_CORES),
        .NUM_LANES    (`NUM_THREADS),
        .NUM_OUTPUTS  (`NUM_FPU_UNITS),
        .TAG_WIDTH    (`FPU_REQ_TAG_WIDTH),
        .BUFFERED_REQ ((`NUM_CORES != `NUM_FPU_UNITS) ? 1 : 0),
        .BUFFERED_RSP ((`NUM_CORES != `NUM_FPU_UNITS) ? 1 : 0)
    ) fpu_arb (
        .clk        (clk),
        .reset      (reset),
        .req_in_if  (per_core_fpu_req_if),
        .rsp_in_if  (per_core_fpu_rsp_if),
        .req_out_if (fpu_req_if),
        .rsp_out_if (fpu_rsp_if)
    );

    // Generate all floating-point units
    for (genvar i = 0; i < `NUM_FPU_UNITS; ++i) begin
        `RESET_RELAY (fpu_reset, reset);

        VX_fpu_unit #(
            .INSTANCE_ID ($sformatf("cluster%0d-fpu", CLUSTER_ID)),
            .NUM_LANES   (`NUM_THREADS),
            .TAG_WIDTH   (`FPU_REQ_ARB_TAG_WIDTH)
        ) fpu_unit (
            .clk        (clk),
            .reset      (fpu_reset),        
            .fpu_req_if (fpu_req_if[i]), 
            .fpu_rsp_if (fpu_rsp_if[i])  
        );
    end

`endif

`ifdef PERF_ENABLE
    VX_perf_memsys_if perf_memsys_if();
`endif

    VX_cache_req_if #(
        .NUM_REQS  (DCACHE_NUM_REQS), 
        .WORD_SIZE (DCACHE_WORD_SIZE), 
        .TAG_WIDTH (DCACHE_TAG_WIDTH)
    ) per_core_dcache_req_if[`NUM_CORES]();

    VX_cache_rsp_if #(
        .NUM_REQS  (DCACHE_NUM_REQS), 
        .WORD_SIZE (DCACHE_WORD_SIZE), 
        .TAG_WIDTH (DCACHE_TAG_WIDTH)
    ) per_core_dcache_rsp_if[`NUM_CORES]();
    
    VX_cache_req_if #(
        .NUM_REQS  (ICACHE_NUM_REQS), 
        .WORD_SIZE (ICACHE_WORD_SIZE), 
        .TAG_WIDTH (ICACHE_TAG_WIDTH)
    ) per_core_icache_req_if[`NUM_CORES]();

    VX_cache_rsp_if #(
        .NUM_REQS  (ICACHE_NUM_REQS), 
        .WORD_SIZE (ICACHE_WORD_SIZE), 
        .TAG_WIDTH (ICACHE_TAG_WIDTH)
    ) per_core_icache_rsp_if[`NUM_CORES]();

    wire [`NUM_CORES-1:0] per_core_sim_ebreak;
    wire [`NUM_CORES-1:0][`NUM_REGS-1:0][31:0] per_core_sim_wb_value;
    assign sim_ebreak = per_core_sim_ebreak[0];
    assign sim_wb_value = per_core_sim_wb_value[0];
    `UNUSED_VAR (per_core_sim_ebreak)
    `UNUSED_VAR (per_core_sim_wb_value)

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

         `ifdef PERF_ENABLE
            .perf_memsys_if (perf_memsys_if),
        `endif       

            .dcache_req_if  (per_core_dcache_req_if[i]),
            .dcache_rsp_if  (per_core_dcache_rsp_if[i]),

            .icache_req_if  (per_core_icache_req_if[i]),
            .icache_rsp_if  (per_core_icache_rsp_if[i]),

        `ifdef EXT_F_ENABLE
            .fpu_req_if     (per_core_fpu_req_if[i]),
            .fpu_rsp_if     (per_core_fpu_rsp_if[i]),
        `endif

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

            .sim_ebreak     (per_core_sim_ebreak[i]),
            .sim_wb_value   (per_core_sim_wb_value[i]),
            .busy           (per_core_busy[i])
        );
    end 

    VX_mem_unit #(
        .CLUSTER_ID (CLUSTER_ID)
    ) mem_unit (
        .clk                (clk),
        .reset              (reset),
    `ifdef PERF_ENABLE
        .perf_memsys_if     (perf_memsys_if),
    `endif
        .dcache_req_if      (per_core_dcache_req_if),
        .dcache_rsp_if      (per_core_dcache_rsp_if),
        
        .icache_req_if      (per_core_icache_req_if),
        .icache_rsp_if      (per_core_icache_rsp_if),

    `ifdef EXT_TEX_ENABLE
        .tcache_req_if      (tcache_req_if),
        .tcache_rsp_if      (tcache_rsp_if),
    `endif

    `ifdef EXT_RASTER_ENABLE
        .rcache_req_if      (rcache_req_if),
        .rcache_rsp_if      (rcache_rsp_if),
    `endif 

    `ifdef EXT_ROP_ENABLE
        .ocache_req_if      (ocache_req_if),
        .ocache_rsp_if      (ocache_rsp_if),
    `endif

        .mem_req_if         (mem_req_if),
        .mem_rsp_if         (mem_rsp_if)
    );
    
    assign busy = (| per_core_busy);

endmodule
