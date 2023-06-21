`include "VX_define.vh"
`include "VX_gpu_types.vh"

`ifdef EXT_TEX_ENABLE
`include "VX_tex_define.vh"
`endif

`ifdef EXT_RASTER_ENABLE
`include "VX_raster_define.vh"
`endif

`ifdef EXT_ROP_ENABLE
`include "VX_rop_define.vh"
`endif

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
`IGNORE_WARNINGS_END

module VX_cluster #(
    parameter CLUSTER_ID = 0
) ( 
    `SCOPE_IO_DECL

    // Clock
    input  wire                 clk,
    input  wire                 reset,

`ifdef PERF_ENABLE
    VX_perf_memsys_if.master    perf_memsys_if,
    VX_perf_memsys_if.slave     perf_memsys_total_if,
`endif

    VX_dcr_write_if.slave       dcr_write_if,

`ifdef EXT_TEX_ENABLE
`ifdef PERF_ENABLE
    VX_tex_perf_if.master       perf_tex_if,
    VX_perf_cache_if.master     perf_tcache_if,
    VX_tex_perf_if.slave        perf_tex_total_if,
    VX_perf_cache_if.slave      perf_tcache_total_if,
`endif
`endif

`ifdef EXT_RASTER_ENABLE
`ifdef PERF_ENABLE
    VX_raster_perf_if.master    perf_raster_if,
    VX_perf_cache_if.master     perf_rcache_if,
    VX_raster_perf_if.slave     perf_raster_total_if,
    VX_perf_cache_if.slave      perf_rcache_total_if,
`endif
`endif

`ifdef EXT_ROP_ENABLE
`ifdef PERF_ENABLE
    VX_rop_perf_if.master       perf_rop_if,
    VX_perf_cache_if.master     perf_ocache_if,
    VX_rop_perf_if.slave        perf_rop_total_if,
    VX_perf_cache_if.slave      perf_ocache_total_if,
`endif
`endif

    // Memory
    VX_mem_req_if.master        mem_req_if,
    VX_mem_rsp_if.slave         mem_rsp_if,

    // simulation helper signals
    output wire                 sim_ebreak,
    output wire [`NUM_REGS-1:0][`XLEN-1:0] sim_wb_value,

    // Status
    output wire                 busy
);

`ifdef SCOPE
    localparam scope_raster_units = `EXT_RASTER_ENABLED ? `NUM_RASTER_UNITS : 0;
    `SCOPE_IO_SWITCH (scope_raster_units + `NUM_SOCKETS);
`endif

    VX_gbar_if per_socket_gbar_if[`NUM_SOCKETS]();
    VX_gbar_if gbar_if();

    `RESET_RELAY (gbar_reset, reset);

    VX_gbar_arb #(
        .NUM_REQS (`NUM_SOCKETS)
    ) gbar_arb (
        .clk        (clk),
        .reset      (gbar_reset),
        .req_in_if  (per_socket_gbar_if),
        .req_out_if (gbar_if)
    );

    VX_gbar_unit #(
        .INSTANCE_ID ($sformatf("gbar%0d", CLUSTER_ID))
    ) gbar_unit (
        .clk     (clk),
        .reset   (gbar_reset),
        .gbar_if (gbar_if)
    );

`ifdef EXT_RASTER_ENABLE

`ifdef PERF_ENABLE
    VX_raster_perf_if perf_raster_unit_if[`NUM_RASTER_UNITS]();
    `PERF_RASTER_ADD (perf_raster_if, perf_raster_unit_if, `NUM_RASTER_UNITS);
`endif

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

    VX_raster_req_if #(
        .NUM_LANES (`NUM_THREADS)
    ) raster_req_if[`NUM_RASTER_UNITS]();

    VX_dcr_write_if raster_dcr_write_tmp_if();
    assign raster_dcr_write_tmp_if.valid = dcr_write_if.valid && (dcr_write_if.addr >= `DCR_RASTER_STATE_BEGIN && dcr_write_if.addr < `DCR_RASTER_STATE_END);
    assign raster_dcr_write_tmp_if.addr  = dcr_write_if.addr;
    assign raster_dcr_write_tmp_if.data  = dcr_write_if.data;

    `BUFFER_DCR_WRITE_IF (raster_dcr_write_if, raster_dcr_write_tmp_if, 1);

    // Generate all raster units
    for (genvar i = 0; i < `NUM_RASTER_UNITS; ++i) begin

        `RESET_RELAY (raster_reset, reset);

        VX_raster_unit #( 
            .INSTANCE_ID     ($sformatf("cluster%0d-raster%0d", CLUSTER_ID, i)),
            .INSTANCE_IDX    (CLUSTER_ID * `NUM_RASTER_UNITS + i),
            .NUM_INSTANCES   (`NUM_CLUSTERS * `NUM_RASTER_UNITS),
            .NUM_SLICES      (`RASTER_NUM_SLICES),
            .TILE_LOGSIZE    (`RASTER_TILE_LOGSIZE),
            .BLOCK_LOGSIZE   (`RASTER_BLOCK_LOGSIZE),
            .MEM_FIFO_DEPTH  (`RASTER_MEM_FIFO_DEPTH),
            .QUAD_FIFO_DEPTH (`RASTER_QUAD_FIFO_DEPTH),
            .OUTPUT_QUADS    (`NUM_THREADS)
        ) raster_unit (
            `SCOPE_IO_BIND (i)
            .clk           (clk),
            .reset         (raster_reset),
        `ifdef PERF_ENABLE
            .perf_raster_if(perf_raster_unit_if[i]),
        `endif
            .dcr_write_if  (raster_dcr_write_if),
            .raster_req_if (raster_req_if[i]),
            .cache_req_if  (rcache_req_if[i]),
            .cache_rsp_if  (rcache_rsp_if[i])
        );
    end

    VX_raster_req_if #(
        .NUM_LANES (`NUM_THREADS)
    ) per_socket_raster_req_if[`NUM_SOCKETS]();

    `RESET_RELAY (raster_arb_reset, reset);

    VX_raster_arb #(
        .NUM_INPUTS  (`NUM_RASTER_UNITS),
        .NUM_LANES   (`NUM_THREADS),
        .NUM_OUTPUTS (`NUM_SOCKETS),
        .ARBITER     ("R"),
        .BUFFERED    ((`NUM_SOCKETS != `NUM_RASTER_UNITS) ? 2 : 0)
    ) raster_arb (
        .clk        (clk),
        .reset      (raster_arb_reset),
        .req_in_if  (raster_req_if),
        .req_out_if (per_socket_raster_req_if)
    );   
`endif

`ifdef EXT_ROP_ENABLE

`ifdef PERF_ENABLE
    VX_rop_perf_if perf_rop_unit_if[`NUM_ROP_UNITS]();
    `PERF_ROP_ADD (perf_rop_if, perf_rop_unit_if, `NUM_ROP_UNITS);
`endif

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

    VX_rop_req_if #(
        .NUM_LANES (`NUM_THREADS)
    ) per_socket_rop_req_if[`NUM_SOCKETS]();

    VX_rop_req_if #(
        .NUM_LANES (`NUM_THREADS)
    ) rop_req_if[`NUM_ROP_UNITS]();

    `RESET_RELAY (rop_arb_reset, reset);

    VX_rop_arb #(
        .NUM_INPUTS  (`NUM_SOCKETS),
        .NUM_LANES   (`NUM_THREADS),
        .NUM_OUTPUTS (`NUM_ROP_UNITS),
        .ARBITER     ("R"),
        .BUFFERED    ((`NUM_SOCKETS != `NUM_ROP_UNITS) ? 2 : 0)
    ) rop_arb (
        .clk        (clk),
        .reset      (rop_arb_reset),
        .req_in_if  (per_socket_rop_req_if),
        .req_out_if (rop_req_if)
    );

    VX_dcr_write_if rop_dcr_write_tmp_if();
    assign rop_dcr_write_tmp_if.valid = dcr_write_if.valid && (dcr_write_if.addr >= `DCR_ROP_STATE_BEGIN && dcr_write_if.addr < `DCR_ROP_STATE_END);
    assign rop_dcr_write_tmp_if.addr  = dcr_write_if.addr;
    assign rop_dcr_write_tmp_if.data  = dcr_write_if.data;

    `BUFFER_DCR_WRITE_IF (rop_dcr_write_if, rop_dcr_write_tmp_if, 1);

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
            .perf_rop_if   (perf_rop_unit_if[i]),
        `endif
            .dcr_write_if  (rop_dcr_write_if),
            .rop_req_if    (rop_req_if[i]),            
            .cache_req_if  (ocache_req_if[i]),
            .cache_rsp_if  (ocache_rsp_if[i])
        );
    end

`endif

`ifdef EXT_TEX_ENABLE

`ifdef PERF_ENABLE
    VX_tex_perf_if perf_tex_unit_if[`NUM_TEX_UNITS]();
    `PERF_TEX_ADD (perf_tex_if, perf_tex_unit_if, `NUM_TEX_UNITS);
`endif

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

    VX_tex_req_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`TEX_REQ_ARB1_TAG_WIDTH)
    ) per_socket_tex_req_if[`NUM_SOCKETS]();

    VX_tex_rsp_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`TEX_REQ_ARB1_TAG_WIDTH)
    ) per_socket_tex_rsp_if[`NUM_SOCKETS]();

    VX_tex_req_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`TEX_REQ_ARB2_TAG_WIDTH)
    ) tex_req_if[`NUM_TEX_UNITS]();

    VX_tex_rsp_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`TEX_REQ_ARB2_TAG_WIDTH)
    ) tex_rsp_if[`NUM_TEX_UNITS]();

    `RESET_RELAY (tex_arb_reset, reset);

    VX_tex_arb #(
        .NUM_INPUTS   (`NUM_SOCKETS),
        .NUM_LANES    (`NUM_THREADS),
        .NUM_OUTPUTS  (`NUM_TEX_UNITS),
        .TAG_WIDTH    (`TEX_REQ_ARB1_TAG_WIDTH),
        .ARBITER      ("R"),
        .BUFFERED_REQ ((`NUM_SOCKETS != `NUM_TEX_UNITS) ? 2 : 0)
    ) tex_arb (
        .clk        (clk),
        .reset      (tex_arb_reset),
        .req_in_if  (per_socket_tex_req_if),
        .rsp_in_if  (per_socket_tex_rsp_if),
        .req_out_if (tex_req_if),
        .rsp_out_if (tex_rsp_if)
    );

    VX_dcr_write_if tex_dcr_write_tmp_if();
    assign tex_dcr_write_tmp_if.valid = dcr_write_if.valid && (dcr_write_if.addr >= `DCR_TEX_STATE_BEGIN && dcr_write_if.addr < `DCR_TEX_STATE_END);
    assign tex_dcr_write_tmp_if.addr  = dcr_write_if.addr;
    assign tex_dcr_write_tmp_if.data  = dcr_write_if.data;

    `BUFFER_DCR_WRITE_IF (tex_dcr_write_if, tex_dcr_write_tmp_if, 1);

    // Generate all texture units
    for (genvar i = 0; i < `NUM_TEX_UNITS; ++i) begin

        `RESET_RELAY (tex_reset, reset);

        VX_tex_unit #(
            .INSTANCE_ID ($sformatf("cluster%0d-tex%0d", CLUSTER_ID, i)),
            .NUM_LANES   (`NUM_THREADS),
            .TAG_WIDTH   (`TEX_REQ_ARB2_TAG_WIDTH)
        ) tex_unit (
            .clk          (clk),
            .reset        (tex_reset),
        `ifdef PERF_ENABLE
            .perf_tex_if  (perf_tex_unit_if[i]),
        `endif 
            .dcr_write_if (tex_dcr_write_if),
            .tex_req_if   (tex_req_if[i]),
            .tex_rsp_if   (tex_rsp_if[i]),
            .cache_req_if (tcache_req_if[i]),
            .cache_rsp_if (tcache_rsp_if[i])
        );
    end
            
`endif

`ifdef EXT_F_ENABLE

    VX_fpu_req_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`FPU_REQ_ARB1_TAG_WIDTH)
    ) per_socket_fpu_req_if[`NUM_SOCKETS]();

    VX_fpu_rsp_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`FPU_REQ_ARB1_TAG_WIDTH)
    ) per_socket_fpu_rsp_if[`NUM_SOCKETS]();

    VX_fpu_req_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`FPU_REQ_ARB2_TAG_WIDTH)
    ) fpu_req_if[`NUM_FPU_UNITS]();

    VX_fpu_rsp_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`FPU_REQ_ARB2_TAG_WIDTH)
    ) fpu_rsp_if[`NUM_FPU_UNITS]();

    `RESET_RELAY (fpu_arb_reset, reset);

    VX_fpu_arb #(
        .NUM_INPUTS   (`NUM_SOCKETS),
        .NUM_LANES    (`NUM_THREADS),
        .NUM_OUTPUTS  (`NUM_FPU_UNITS),
        .TAG_WIDTH    (`FPU_REQ_ARB1_TAG_WIDTH),
        .ARBITER      ("R"),
        .BUFFERED_REQ ((`NUM_SOCKETS != `NUM_FPU_UNITS) ? 2 : 0)
    ) fpu_arb (
        .clk        (clk),
        .reset      (fpu_arb_reset),
        .req_in_if  (per_socket_fpu_req_if),
        .rsp_in_if  (per_socket_fpu_rsp_if),
        .req_out_if (fpu_req_if),
        .rsp_out_if (fpu_rsp_if)
    );

    // Generate all floating-point units
    for (genvar i = 0; i < `NUM_FPU_UNITS; ++i) begin

        `RESET_RELAY (fpu_reset, reset);

        VX_fpu_unit #(
            .INSTANCE_ID ($sformatf("cluster%0d-fpu", CLUSTER_ID)),
            .NUM_LANES   (`NUM_THREADS),
            .TAG_WIDTH   (`FPU_REQ_ARB2_TAG_WIDTH)
        ) fpu_unit (
            .clk        (clk),
            .reset      (fpu_reset),        
            .fpu_req_if (fpu_req_if[i]), 
            .fpu_rsp_if (fpu_rsp_if[i])  
        );
    end

`endif

    VX_cache_req_if #(
        .NUM_REQS  (DCACHE_NUM_REQS), 
        .WORD_SIZE (DCACHE_WORD_SIZE), 
        .TAG_WIDTH (DCACHE_ARB_TAG_WIDTH)
    ) per_socket_dcache_req_if[`NUM_SOCKETS]();

    VX_cache_rsp_if #(
        .NUM_REQS  (DCACHE_NUM_REQS), 
        .WORD_SIZE (DCACHE_WORD_SIZE), 
        .TAG_WIDTH (DCACHE_ARB_TAG_WIDTH)
    ) per_socket_dcache_rsp_if[`NUM_SOCKETS]();
    
    VX_cache_req_if #(
        .NUM_REQS  (ICACHE_NUM_REQS), 
        .WORD_SIZE (ICACHE_WORD_SIZE), 
        .TAG_WIDTH (ICACHE_ARB_TAG_WIDTH)
    ) per_socket_icache_req_if[`NUM_SOCKETS]();

    VX_cache_rsp_if #(
        .NUM_REQS  (ICACHE_NUM_REQS), 
        .WORD_SIZE (ICACHE_WORD_SIZE), 
        .TAG_WIDTH (ICACHE_ARB_TAG_WIDTH)
    ) per_socket_icache_rsp_if[`NUM_SOCKETS]();

    `RESET_RELAY (mem_unit_reset, reset);

    VX_mem_unit #(
        .CLUSTER_ID (CLUSTER_ID)
    ) mem_unit (
        .clk                (clk),
        .reset              (mem_unit_reset),

    `ifdef PERF_ENABLE
        .perf_memsys_if     (perf_memsys_if),
    `endif

        .dcache_req_if      (per_socket_dcache_req_if),
        .dcache_rsp_if      (per_socket_dcache_rsp_if),
        
        .icache_req_if      (per_socket_icache_req_if),
        .icache_rsp_if      (per_socket_icache_rsp_if),

    `ifdef EXT_TEX_ENABLE
    `ifdef PERF_ENABLE
        .perf_tcache_if     (perf_tcache_if),
    `endif
        .tcache_req_if      (tcache_req_if),
        .tcache_rsp_if      (tcache_rsp_if),
    `endif

    `ifdef EXT_RASTER_ENABLE
    `ifdef PERF_ENABLE
        .perf_rcache_if     (perf_rcache_if),
    `endif
        .rcache_req_if      (rcache_req_if),
        .rcache_rsp_if      (rcache_rsp_if),
    `endif 

    `ifdef EXT_ROP_ENABLE
    `ifdef PERF_ENABLE
        .perf_ocache_if     (perf_ocache_if),
    `endif
        .ocache_req_if      (ocache_req_if),
        .ocache_rsp_if      (ocache_rsp_if),
    `endif

        .mem_req_if         (mem_req_if),
        .mem_rsp_if         (mem_rsp_if)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire [`NUM_SOCKETS-1:0] per_socket_sim_ebreak;
    wire [`NUM_SOCKETS-1:0][`NUM_REGS-1:0][`XLEN-1:0] per_socket_sim_wb_value;
    assign sim_ebreak = per_socket_sim_ebreak[0];
    assign sim_wb_value = per_socket_sim_wb_value[0];
    `UNUSED_VAR (per_socket_sim_ebreak)
    `UNUSED_VAR (per_socket_sim_wb_value)

    VX_dcr_write_if socket_dcr_write_tmp_if();
    assign socket_dcr_write_tmp_if.valid = dcr_write_if.valid && (dcr_write_if.addr >= `DCR_BASE_STATE_BEGIN && dcr_write_if.addr < `DCR_BASE_STATE_END);
    assign socket_dcr_write_tmp_if.addr  = dcr_write_if.addr;
    assign socket_dcr_write_tmp_if.data  = dcr_write_if.data;

    wire [`NUM_SOCKETS-1:0] per_socket_busy;

    `BUFFER_DCR_WRITE_IF (socket_dcr_write_if, socket_dcr_write_tmp_if, (`NUM_SOCKETS > 1));

    // Generate all sockets
    for (genvar i = 0; i < `NUM_SOCKETS; ++i) begin

        `RESET_RELAY (socket_reset, reset);

        VX_socket #(
            .SOCKET_ID ((CLUSTER_ID * `NUM_SOCKETS) + i)
        ) socket (
            `SCOPE_IO_BIND  (scope_raster_units+i)

            .clk            (clk),
            .reset          (socket_reset),

        `ifdef PERF_ENABLE
            .perf_memsys_if (perf_memsys_total_if),
        `endif
            
            .dcr_write_if   (socket_dcr_write_if),

            .dcache_req_if  (per_socket_dcache_req_if[i]),
            .dcache_rsp_if  (per_socket_dcache_rsp_if[i]),

            .icache_req_if  (per_socket_icache_req_if[i]),
            .icache_rsp_if  (per_socket_icache_rsp_if[i]),

        `ifdef EXT_F_ENABLE
            .fpu_req_if     (per_socket_fpu_req_if[i]),
            .fpu_rsp_if     (per_socket_fpu_rsp_if[i]),
        `endif

        `ifdef EXT_TEX_ENABLE
        `ifdef PERF_ENABLE
            .perf_tex_if    (perf_tex_total_if),
            .perf_tcache_if (perf_tcache_total_if),
        `endif
            .tex_req_if     (per_socket_tex_req_if[i]),
            .tex_rsp_if     (per_socket_tex_rsp_if[i]),
        `endif

        `ifdef EXT_RASTER_ENABLE
        `ifdef PERF_ENABLE
            .perf_raster_if (perf_raster_total_if),
            .perf_rcache_if (perf_rcache_total_if),
        `endif
            .raster_req_if  (per_socket_raster_req_if[i]),
        `endif
        
        `ifdef EXT_ROP_ENABLE
        `ifdef PERF_ENABLE
            .perf_rop_if    (perf_rop_total_if),
            .perf_ocache_if (perf_ocache_total_if),
        `endif
            .rop_req_if     (per_socket_rop_req_if[i]),
        `endif

            .gbar_if        (per_socket_gbar_if[i]),

            .sim_ebreak     (per_socket_sim_ebreak[i]),
            .sim_wb_value   (per_socket_sim_wb_value[i]),
            .busy           (per_socket_busy[i])
        );
    end

    `BUFFER_BUSY ((| per_socket_busy), (`NUM_SOCKETS > 1));

endmodule
