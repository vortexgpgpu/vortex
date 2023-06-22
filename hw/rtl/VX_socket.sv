`include "VX_define.vh"
`include "VX_gpu_types.vh"

`IGNORE_WARNINGS_BEGIN
import VX_gpu_types::*;
`IGNORE_WARNINGS_END

module VX_socket #( 
    parameter SOCKET_ID = 0
) (        
    `SCOPE_IO_DECL
    
    // Clock
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    VX_perf_memsys_if.slave perf_memsys_if,
`endif

    VX_dcr_write_if.slave   dcr_write_if,

    VX_cache_bus_if.master  dcache_bus_if,

    VX_cache_bus_if.master  icache_bus_if,

`ifdef EXT_F_ENABLE
    VX_fpu_bus_if.master    fpu_bus_if,
`endif

`ifdef EXT_TEX_ENABLE
`ifdef PERF_ENABLE
    VX_tex_perf_if.slave    perf_tex_if,
    VX_perf_cache_if.slave  perf_tcache_if,
`endif
    VX_tex_bus_if.master    tex_bus_if,
`endif

`ifdef EXT_RASTER_ENABLE
`ifdef PERF_ENABLE
    VX_raster_perf_if.slave perf_raster_if,
    VX_perf_cache_if.slave  perf_rcache_if,
`endif
    VX_raster_req_if.slave  raster_req_if,
`endif

`ifdef EXT_ROP_ENABLE
`ifdef PERF_ENABLE
    VX_rop_perf_if.slave    perf_rop_if,
    VX_perf_cache_if.slave  perf_ocache_if,
`endif
    VX_rop_req_if.master    rop_req_if,
`endif

    VX_gbar_bus_if.master   gbar_bus_if,


    // simulation helper signals
    output wire             sim_ebreak,
    output wire [`NUM_REGS-1:0][`XLEN-1:0] sim_wb_value,

    // Status
    output wire             busy
);

    VX_gbar_bus_if per_core_gbar_bus_if[`SOCKET_SIZE]();

    `RESET_RELAY (gbar_arb_reset, reset);

    VX_gbar_arb #(
        .NUM_REQS (`SOCKET_SIZE)
    ) gbar_arb (
        .clk        (clk),
        .reset      (gbar_arb_reset),
        .bus_in_if  (per_core_gbar_bus_if),
        .bus_out_if (gbar_bus_if)
    );

`ifdef EXT_RASTER_ENABLE

    VX_raster_req_if #(
        .NUM_LANES (`NUM_THREADS)
    ) per_core_raster_req_if[`SOCKET_SIZE](), raster_req_tmp_if[1]();

    `RESET_RELAY (raster_arb_reset, reset);

    VX_raster_arb #(
        .NUM_INPUTS  (1),
        .NUM_LANES   (`NUM_THREADS),
        .NUM_OUTPUTS (`SOCKET_SIZE),
        .ARBITER     ("R"),
        .BUFFERED    ((`SOCKET_SIZE > 1) ? 2 : 0)
    ) raster_arb (
        .clk        (clk),
        .reset      (raster_arb_reset),
        .req_in_if  (raster_req_tmp_if),
        .req_out_if (per_core_raster_req_if)
    );

    `ASSIGN_VX_RASTER_REQ_IF (raster_req_tmp_if[0], raster_req_if);

`endif

`ifdef EXT_ROP_ENABLE

    VX_rop_req_if #(
        .NUM_LANES (`NUM_THREADS)
    ) per_core_rop_req_if[`SOCKET_SIZE](), rop_req_tmp_if[1]();

    `RESET_RELAY (rop_arb_reset, reset);

    VX_rop_arb #(
        .NUM_INPUTS  (`SOCKET_SIZE),
        .NUM_LANES   (`NUM_THREADS),
        .NUM_OUTPUTS (1),
        .ARBITER     ("R"),
        .BUFFERED    ((`SOCKET_SIZE > 1) ? 2 : 0)
    ) rop_arb (
        .clk        (clk),
        .reset      (rop_arb_reset),
        .req_in_if  (per_core_rop_req_if),
        .req_out_if (rop_req_tmp_if)
    );

    `ASSIGN_VX_ROP_REQ_IF (rop_req_if, rop_req_tmp_if[0]);

`endif

`ifdef EXT_TEX_ENABLE

    VX_tex_bus_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`TEX_REQ_TAG_WIDTH)
    ) per_core_tex_bus_if[`SOCKET_SIZE]();

    VX_tex_bus_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`TEX_REQ_ARB1_TAG_WIDTH)
    ) tex_bus_tmp_if[1]();

    `RESET_RELAY (tex_arb_reset, reset);

    VX_tex_arb #(
        .NUM_INPUTS   (`SOCKET_SIZE),
        .NUM_LANES    (`NUM_THREADS),
        .NUM_OUTPUTS  (1),
        .TAG_WIDTH    (`TEX_REQ_TAG_WIDTH),
        .ARBITER      ("R"),
        .BUFFERED_REQ ((`SOCKET_SIZE > 1) ? 2 : 0)
    ) tex_arb (
        .clk        (clk),
        .reset      (tex_arb_reset),
        .bus_in_if  (per_core_tex_bus_if),
        .bus_out_if (tex_bus_tmp_if)
    );

    `ASSIGN_VX_TEX_BUS_IF (tex_bus_if, tex_bus_tmp_if[0]);
            
`endif

`ifdef EXT_F_ENABLE

    VX_fpu_bus_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`FPU_REQ_TAG_WIDTH)
    ) per_core_fpu_bus_if[`SOCKET_SIZE]();

    VX_fpu_bus_if #(
        .NUM_LANES (`NUM_THREADS),
        .TAG_WIDTH (`FPU_REQ_ARB1_TAG_WIDTH)
    ) fpu_bus_tmp_if[1]();

    `RESET_RELAY (fpu_arb_reset, reset);

    VX_fpu_arb #(
        .NUM_INPUTS   (`SOCKET_SIZE),
        .NUM_LANES    (`NUM_THREADS),
        .NUM_OUTPUTS  (1),
        .TAG_WIDTH    (`FPU_REQ_TAG_WIDTH),
        .ARBITER      ("R"),
        .BUFFERED_REQ ((`SOCKET_SIZE > 1) ? 2 : 0)
    ) fpu_arb (
        .clk        (clk),
        .reset      (fpu_arb_reset),
        .bus_in_if  (per_core_fpu_bus_if),
        .bus_out_if (fpu_bus_tmp_if)
    );

    `ASSIGN_VX_FPU_BUS_IF (fpu_bus_if, fpu_bus_tmp_if[0]);

`endif

    ///////////////////////////////////////////////////////////////////////////

    VX_cache_bus_if #(
        .NUM_REQS  (DCACHE_NUM_REQS), 
        .WORD_SIZE (DCACHE_WORD_SIZE), 
        .TAG_WIDTH (DCACHE_TAG_WIDTH)
    ) per_core_dcache_bus_if[`SOCKET_SIZE]();

    VX_cache_bus_if #(
        .NUM_REQS  (DCACHE_NUM_REQS), 
        .WORD_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_ARB_TAG_WIDTH)
    ) dcache_bus_tmp_if[1]();

    `RESET_RELAY (dcache_arb_reset, reset);

    VX_cache_arb #(
        .NUM_INPUTS   (`SOCKET_SIZE),
        .NUM_OUTPUTS  (1),
        .NUM_LANES    (DCACHE_NUM_REQS),
        .DATA_SIZE    (DCACHE_WORD_SIZE),
        .TAG_WIDTH    (DCACHE_TAG_WIDTH),
        .TAG_SEL_IDX  (`CACHE_ADDR_TYPE_BITS),
        .ARBITER      ("R"),
        .BUFFERED_REQ ((`SOCKET_SIZE > 1) ? 2 : 0),
        .BUFFERED_RSP ((`SOCKET_SIZE > 1) ? 2 : 0)
    ) dcache_arb (
        .clk        (clk),
        .reset      (dcache_arb_reset),
        .bus_in_if  (per_core_dcache_bus_if),
        .bus_out_if (dcache_bus_tmp_if)
    );

    `ASSIGN_VX_CACHE_BUS_IF (dcache_bus_if, dcache_bus_tmp_if[0]);

    ///////////////////////////////////////////////////////////////////////////
    
    VX_cache_bus_if #(
        .NUM_REQS  (ICACHE_NUM_REQS), 
        .WORD_SIZE (ICACHE_WORD_SIZE), 
        .TAG_WIDTH (ICACHE_TAG_WIDTH)
    ) per_core_icache_bus_if[`SOCKET_SIZE]();

    VX_cache_bus_if #(
        .NUM_REQS  (ICACHE_NUM_REQS), 
        .WORD_SIZE (ICACHE_WORD_SIZE),
        .TAG_WIDTH (ICACHE_ARB_TAG_WIDTH)
    ) icache_bus_tmp_if[1]();

    `RESET_RELAY (icache_arb_reset, reset);

    VX_cache_arb #(
        .NUM_INPUTS   (`SOCKET_SIZE),
        .NUM_OUTPUTS  (1),
        .NUM_LANES    (ICACHE_NUM_REQS),
        .DATA_SIZE    (ICACHE_WORD_SIZE),
        .TAG_WIDTH    (ICACHE_TAG_WIDTH),
        .TAG_SEL_IDX  (0),
        .ARBITER      ("R"),
        .BUFFERED_REQ ((`SOCKET_SIZE > 1) ? 2 : 0),
        .BUFFERED_RSP ((`SOCKET_SIZE > 1) ? 2 : 0)
    ) icache_arb (
        .clk        (clk),
        .reset      (icache_arb_reset),
        .bus_in_if  (per_core_icache_bus_if),
        .bus_out_if (icache_bus_tmp_if)
    );

    `ASSIGN_VX_CACHE_BUS_IF (icache_bus_if, icache_bus_tmp_if[0]);

    ///////////////////////////////////////////////////////////////////////////

    wire [`SOCKET_SIZE-1:0] per_core_sim_ebreak;
    wire [`SOCKET_SIZE-1:0][`NUM_REGS-1:0][`XLEN-1:0] per_core_sim_wb_value;
    assign sim_ebreak = per_core_sim_ebreak[0];
    assign sim_wb_value = per_core_sim_wb_value[0];
    `UNUSED_VAR (per_core_sim_ebreak)
    `UNUSED_VAR (per_core_sim_wb_value)

    wire [`SOCKET_SIZE-1:0] per_core_busy;

    `BUFFER_DCR_WRITE_IF (core_dcr_write_if, dcr_write_if, (`SOCKET_SIZE > 1));

    `SCOPE_IO_SWITCH (`SOCKET_SIZE)

    // Generate all cores
    for (genvar i = 0; i < `SOCKET_SIZE; ++i) begin

        `RESET_RELAY (core_reset, reset);

        VX_core #(
            .CORE_ID ((SOCKET_ID * `SOCKET_SIZE) + i)
        ) core (
            `SCOPE_IO_BIND  (i)

            .clk            (clk),
            .reset          (core_reset),

        `ifdef PERF_ENABLE
            .perf_memsys_if (perf_memsys_if),
        `endif
            
            .dcr_write_if   (core_dcr_write_if),

            .dcache_bus_if  (per_core_dcache_bus_if[i]),

            .icache_bus_if  (per_core_icache_bus_if[i]),

        `ifdef EXT_F_ENABLE
            .fpu_bus_if     (per_core_fpu_bus_if[i]),
        `endif

        `ifdef EXT_TEX_ENABLE
        `ifdef PERF_ENABLE
            .perf_tex_if    (perf_tex_if),
            .perf_tcache_if (perf_tcache_if),
        `endif
            .tex_bus_if     (per_core_tex_bus_if[i]),
        `endif

        `ifdef EXT_RASTER_ENABLE
        `ifdef PERF_ENABLE
            .perf_raster_if (perf_raster_if),
            .perf_rcache_if (perf_rcache_if),
        `endif
            .raster_req_if  (per_core_raster_req_if[i]),
        `endif
        
        `ifdef EXT_ROP_ENABLE
        `ifdef PERF_ENABLE
            .perf_rop_if    (perf_rop_if),
            .perf_ocache_if (perf_ocache_if),
        `endif
            .rop_req_if     (per_core_rop_req_if[i]),
        `endif

            .gbar_bus_if    (per_core_gbar_bus_if[i]),

            .sim_ebreak     (per_core_sim_ebreak[i]),
            .sim_wb_value   (per_core_sim_wb_value[i]),
            .busy           (per_core_busy[i])
        );
    end

    `BUFFER_BUSY ((| per_core_busy), (`SOCKET_SIZE > 1));
    
endmodule
