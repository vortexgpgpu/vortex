// Copyright 2024
// MMU: TLB + PTW for VA→PA translation

`include "VX_define.vh"
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off WIDTHEXPAND */

module VX_mmu import VX_gpu_pkg::*; #(
    parameter NUM_REQS       = DCACHE_NUM_REQS,
    parameter DATA_SIZE      = DCACHE_WORD_SIZE,
    parameter TAG_WIDTH      = DCACHE_TAG_WIDTH_BASE,
    parameter MEM_ADDR_WIDTH = `MEM_ADDR_WIDTH,
    parameter ADDR_WIDTH     = MEM_ADDR_WIDTH - `CLOG2(DATA_SIZE),
    parameter FLAGS_WIDTH    = MEM_FLAGS_WIDTH,
    parameter EBUF_SIZE      = 2
) (
    input wire clk,
    input wire reset,

`ifdef PERF_ENABLE
    output mmu_perf_t    mmu_perf,
`endif

    input wire [31:0]    satp,

    VX_mem_bus_if.slave  lsu_mem_if [NUM_REQS],
    VX_mem_bus_if.master dcache_mem_if [NUM_REQS]
);

    // =========================================================================
    // Bypass Control
    // =========================================================================
    //
    // The runtime installs identity PTEs at boot for every PA-addressed
    // region (IO MMIO, kernel image, page table, stacks), so any access
    // with SATP enabled walks the page table — no address-range bypass
    // is needed. The only access path that skips translation is one
    // issued in BARE mode (SATP MSB cleared), which covers the few
    // instruction fetches between reset and the kernel's csrw satp.

    function automatic logic needs_translation(input logic [31:0] full_addr);
        /* verilator lint_off UNUSEDSIGNAL */
        logic [31:0] addr_unused = full_addr;
        /* verilator lint_on UNUSEDSIGNAL */
        if (!satp[31]) return 1'b0;  // BARE mode
        return 1'b1;
    endfunction

    // =========================================================================
    // Local Parameters
    // =========================================================================

    localparam DATA_WIDTH      = DATA_SIZE * 8;
    localparam TLB_SOURCE_BITS = `UP(`CLOG2(NUM_REQS));
    localparam TAG_WIDTH_TLB   = TAG_WIDTH + TLB_SOURCE_BITS;
    localparam REQ_DATAW       = 1 + ADDR_WIDTH + DATA_WIDTH + DATA_SIZE + FLAGS_WIDTH + TAG_WIDTH;
    localparam RSP_DATAW       = DATA_WIDTH + TAG_WIDTH;

    // =========================================================================
    // Internal Interfaces
    // =========================================================================

    VX_mem_bus_if #(
        .DATA_SIZE   (DATA_SIZE),
        .TAG_WIDTH   (TAG_WIDTH),
        .FLAGS_WIDTH (FLAGS_WIDTH)
    ) buffered_if[NUM_REQS]();

    VX_mem_bus_if #(
        .DATA_SIZE   (DATA_SIZE),
        .TAG_WIDTH   (TAG_WIDTH_TLB),
        .FLAGS_WIDTH (FLAGS_WIDTH)
    ) tlb_out_if[NUM_REQS]();

    VX_mem_bus_if #(
        .DATA_SIZE   (DATA_SIZE),
        .TAG_WIDTH   (TAG_WIDTH_TLB),
        .FLAGS_WIDTH (FLAGS_WIDTH)
    ) bypass_dcache_if[NUM_REQS]();

    VX_mem_bus_if #(
        .DATA_SIZE   (DATA_SIZE),
        .TAG_WIDTH   (TAG_WIDTH_TLB),
        .FLAGS_WIDTH (FLAGS_WIDTH)
    ) ptw_mem_if();

`ifdef PERF_ENABLE
    /* verilator lint_off UNUSEDSIGNAL */
    mmu_perf_t mmu_perf_tlb;
    /* verilator lint_on UNUSEDSIGNAL */
    wire [PERF_CTR_BITS-1:0] ptw_latency_counter;
`endif

    // [0..NUM_REQS-1]=bypass, [NUM_REQS..2*NUM_REQS-1]=TLB, [2*NUM_REQS]=PTW
    VX_mem_bus_if #(
        .DATA_SIZE   (DATA_SIZE),
        .TAG_WIDTH   (TAG_WIDTH_TLB),
        .FLAGS_WIDTH (FLAGS_WIDTH)
    ) merge_in_if[2 * NUM_REQS + 1]();

    // =========================================================================
    // Elastic Buffers (TLB path only)
    // =========================================================================

    wire [NUM_REQS-1:0] ebuf_req_ready;
    wire [NUM_REQS-1:0] ebuf_rsp_valid;
    wire [DATA_WIDTH-1:0] ebuf_rsp_data [NUM_REQS];
    wire [TAG_WIDTH-1:0]  ebuf_rsp_tag  [NUM_REQS];
    wire [NUM_REQS-1:0]   ebuf_rsp_ready;
    wire [NUM_REQS-1:0] lane_needs_trans_ebuf;

    for (genvar i = 0; i < NUM_REQS; i++) begin : g_elastic_buffers

        wire [31:0] full_addr_ebuf = {lsu_mem_if[i].req_data.addr, {`CLOG2(DATA_SIZE){1'b0}}};
        assign lane_needs_trans_ebuf[i] = needs_translation(full_addr_ebuf);

        wire [REQ_DATAW-1:0] req_data_in_packed;
        wire [REQ_DATAW-1:0] req_data_out_packed;

        assign req_data_in_packed = {
            lsu_mem_if[i].req_data.rw,
            lsu_mem_if[i].req_data.addr,
            lsu_mem_if[i].req_data.data,
            lsu_mem_if[i].req_data.byteen,
            lsu_mem_if[i].req_data.flags[FLAGS_WIDTH-1:0],
            lsu_mem_if[i].req_data.tag[TAG_WIDTH-1:0]
        };

        VX_elastic_buffer #(
            .DATAW  (REQ_DATAW),
            .SIZE   (EBUF_SIZE),
            .OUT_REG(0)
        ) req_buffer (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (lsu_mem_if[i].req_valid && lane_needs_trans_ebuf[i]),
            .data_in   (req_data_in_packed),
            .ready_in  (ebuf_req_ready[i]),
            .valid_out (buffered_if[i].req_valid),
            .data_out  (req_data_out_packed),
            .ready_out (buffered_if[i].req_ready)
        );

        assign buffered_if[i].req_data.rw     = req_data_out_packed[REQ_DATAW-1];
        assign buffered_if[i].req_data.addr   = req_data_out_packed[REQ_DATAW-2 -: ADDR_WIDTH];
        assign buffered_if[i].req_data.data   = req_data_out_packed[REQ_DATAW-2-ADDR_WIDTH -: DATA_WIDTH];
        assign buffered_if[i].req_data.byteen = req_data_out_packed[REQ_DATAW-2-ADDR_WIDTH-DATA_WIDTH -: DATA_SIZE];
        assign buffered_if[i].req_data.flags  = req_data_out_packed[REQ_DATAW-2-ADDR_WIDTH-DATA_WIDTH-DATA_SIZE -: FLAGS_WIDTH];
        assign buffered_if[i].req_data.tag    = req_data_out_packed[TAG_WIDTH-1:0];

        wire [RSP_DATAW-1:0] rsp_data_in_packed;
        wire [RSP_DATAW-1:0] rsp_data_out_packed;

        assign rsp_data_in_packed = {
            buffered_if[i].rsp_data.data,
            buffered_if[i].rsp_data.tag[TAG_WIDTH-1:0]
        };

        VX_elastic_buffer #(
            .DATAW  (RSP_DATAW),
            .SIZE   (EBUF_SIZE),
            .OUT_REG(0)
        ) rsp_buffer (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (buffered_if[i].rsp_valid),
            .data_in   (rsp_data_in_packed),
            .ready_in  (buffered_if[i].rsp_ready),
            .valid_out (ebuf_rsp_valid[i]),
            .data_out  (rsp_data_out_packed),
            .ready_out (ebuf_rsp_ready[i])
        );

        assign ebuf_rsp_data[i] = rsp_data_out_packed[RSP_DATAW-1 -: DATA_WIDTH];
        assign ebuf_rsp_tag[i]  = rsp_data_out_packed[TAG_WIDTH-1:0];

    end

    // =========================================================================
    // TLB Miss/Fill Interface
    // =========================================================================

    wire        tlb_miss_valid;
    wire        tlb_miss_ready;
    wire [31:0] tlb_miss_vaddr;

    wire        tlb_fill_valid;
    wire        tlb_fill_ready;
    wire [31:0] tlb_fill_vaddr;
    wire [31:0] tlb_fill_paddr;
    wire [7:0]  tlb_fill_flags;

    // =========================================================================
    // TLB Module
    // =========================================================================

    VX_mmu_tlb #(
        .NUM_REQS      (NUM_REQS),
        .DATA_SIZE     (DATA_SIZE),
        .TAG_WIDTH_IN  (TAG_WIDTH),
        .TAG_WIDTH_OUT (TAG_WIDTH_TLB),
        .ADDR_WIDTH    (ADDR_WIDTH),
        .FLAGS_WIDTH   (FLAGS_WIDTH)
    ) tlb_unit (
        .clk           (clk),
        .reset         (reset),
    `ifdef PERF_ENABLE
        .mmu_perf      (mmu_perf_tlb).
    `endif
        .tlb_in_if     (buffered_if),
        .tlb_out_if    (tlb_out_if),
        .miss_valid    (tlb_miss_valid),
        .miss_ready    (tlb_miss_ready),
        .miss_vaddr    (tlb_miss_vaddr),
        .fill_valid    (tlb_fill_valid),
        .fill_ready    (tlb_fill_ready),
        .fill_vaddr    (tlb_fill_vaddr),
        .fill_paddr    (tlb_fill_paddr),
        .fill_flags    (tlb_fill_flags)
    );

    // =========================================================================
    // PTW Module
    // =========================================================================

    VX_mmu_ptw #(
        .DATA_SIZE     (DATA_SIZE),
        .TAG_WIDTH     (TAG_WIDTH_TLB),
        .ADDR_WIDTH    (ADDR_WIDTH),
        .FLAGS_WIDTH   (FLAGS_WIDTH)
    ) ptw_unit (
        .clk           (clk),
        .reset         (reset),
        .satp          (satp),
        .miss_valid    (tlb_miss_valid),
        .miss_ready    (tlb_miss_ready),
        .miss_vaddr    (tlb_miss_vaddr),
        .fill_valid    (tlb_fill_valid),
        .fill_ready    (tlb_fill_ready),
        .fill_vaddr    (tlb_fill_vaddr),
        .fill_paddr    (tlb_fill_paddr),
        .fill_flags    (tlb_fill_flags),
        .ptw_mem_if    (ptw_mem_if),
    `ifdef PERF_ENABLE
        .perf_ptw_latency (ptw_latency_counter)
    `else
        `UNUSED_PIN (perf_ptw_latency_placeholder)
    `endif
    );

    // =========================================================================
    // Bypass Path
    // =========================================================================

    wire [NUM_REQS-1:0] lane_needs_trans;
    wire [NUM_REQS-1:0] lane_bypass;

    for (genvar i = 0; i < NUM_REQS; i++) begin : g_bypass_path
        wire [31:0] full_addr = {lsu_mem_if[i].req_data.addr, {`CLOG2(DATA_SIZE){1'b0}}};

        assign lane_needs_trans[i] = needs_translation(full_addr);
        assign lane_bypass[i] = ~lane_needs_trans[i];

        assign bypass_dcache_if[i].req_valid = lsu_mem_if[i].req_valid && lane_bypass[i];
        assign bypass_dcache_if[i].req_data.rw     = lsu_mem_if[i].req_data.rw;
        assign bypass_dcache_if[i].req_data.addr   = lsu_mem_if[i].req_data.addr;
        assign bypass_dcache_if[i].req_data.data   = lsu_mem_if[i].req_data.data;
        assign bypass_dcache_if[i].req_data.byteen = lsu_mem_if[i].req_data.byteen;
        assign bypass_dcache_if[i].req_data.flags  = lsu_mem_if[i].req_data.flags;
        assign bypass_dcache_if[i].req_data.tag = {lsu_mem_if[i].req_data.tag[TAG_WIDTH-1:0],
                                                   {TLB_SOURCE_BITS{1'b0}}};
    end

    // =========================================================================
    // Merge Arbiter Inputs
    // =========================================================================

    for (genvar i = 0; i < NUM_REQS; i++) begin : g_bypass_to_merge
        assign merge_in_if[i].req_valid = bypass_dcache_if[i].req_valid;
        assign merge_in_if[i].req_data  = bypass_dcache_if[i].req_data;
        assign bypass_dcache_if[i].req_ready  = merge_in_if[i].req_ready;

        assign bypass_dcache_if[i].rsp_valid  = merge_in_if[i].rsp_valid;
        assign bypass_dcache_if[i].rsp_data   = merge_in_if[i].rsp_data;
        assign merge_in_if[i].rsp_ready = bypass_dcache_if[i].rsp_ready;
    end

    for (genvar i = 0; i < NUM_REQS; i++) begin : g_tlb_to_merge
        assign merge_in_if[NUM_REQS + i].req_valid = tlb_out_if[i].req_valid;
        assign merge_in_if[NUM_REQS + i].req_data  = tlb_out_if[i].req_data;
        assign tlb_out_if[i].req_ready  = merge_in_if[NUM_REQS + i].req_ready;

        assign tlb_out_if[i].rsp_valid  = merge_in_if[NUM_REQS + i].rsp_valid;
        assign tlb_out_if[i].rsp_data   = merge_in_if[NUM_REQS + i].rsp_data;
        assign merge_in_if[NUM_REQS + i].rsp_ready = tlb_out_if[i].rsp_ready;
    end

    assign merge_in_if[2 * NUM_REQS].req_valid = ptw_mem_if.req_valid;
    assign merge_in_if[2 * NUM_REQS].req_data  = ptw_mem_if.req_data;
    assign ptw_mem_if.req_ready            = merge_in_if[2 * NUM_REQS].req_ready;
    assign ptw_mem_if.rsp_valid            = merge_in_if[2 * NUM_REQS].rsp_valid;
    assign ptw_mem_if.rsp_data             = merge_in_if[2 * NUM_REQS].rsp_data;
    assign merge_in_if[2 * NUM_REQS].rsp_ready = ptw_mem_if.rsp_ready;

    // =========================================================================
    // Merge Arbiter
    // =========================================================================

    VX_mem_arb #(
        .NUM_INPUTS     (2 * NUM_REQS + 1),
        .NUM_OUTPUTS    (NUM_REQS),
        .DATA_SIZE      (DATA_SIZE),
        .TAG_WIDTH      (TAG_WIDTH_TLB),
        .TAG_SEL_IDX    (TAG_WIDTH_TLB),
        .ARBITER        ("R"),
        .MEM_ADDR_WIDTH (MEM_ADDR_WIDTH),
        .ADDR_WIDTH     (ADDR_WIDTH),
        .FLAGS_WIDTH    (FLAGS_WIDTH),
        .REQ_OUT_BUF    (2),
        .RSP_OUT_BUF    (2)
    ) merge_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (merge_in_if),
        .bus_out_if (dcache_mem_if)
    );

    // =========================================================================
    // LSU Interface
    // =========================================================================

    for (genvar i = 0; i < NUM_REQS; i++) begin : g_lsu_if

        assign lsu_mem_if[i].req_ready = lane_needs_trans_ebuf[i] ?
            ebuf_req_ready[i] : bypass_dcache_if[i].req_ready;

        localparam LSU_RSP_DATAW = DATA_WIDTH + TAG_WIDTH;

        wire [TAG_WIDTH-1:0] bypass_rsp_tag_restored =
            bypass_dcache_if[i].rsp_data.tag[TAG_WIDTH_TLB-1:TLB_SOURCE_BITS];
        wire [LSU_RSP_DATAW-1:0] bypass_rsp_packed = {
            bypass_dcache_if[i].rsp_data.data,
            bypass_rsp_tag_restored
        };

        wire [LSU_RSP_DATAW-1:0] ebuf_rsp_packed = {
            ebuf_rsp_data[i],
            ebuf_rsp_tag[i]
        };

        wire [1:0] rsp_arb_valid_in = {ebuf_rsp_valid[i], bypass_dcache_if[i].rsp_valid};
        wire [1:0][LSU_RSP_DATAW-1:0] rsp_arb_data_in = {ebuf_rsp_packed, bypass_rsp_packed};
        wire [1:0] rsp_arb_ready_in;
        wire rsp_arb_valid_out;
        wire [LSU_RSP_DATAW-1:0] rsp_arb_data_out;

        VX_stream_arb #(
            .NUM_INPUTS  (2),
            .NUM_OUTPUTS (1),
            .DATAW       (LSU_RSP_DATAW),
            .ARBITER     ("R"),
            .OUT_BUF     (0)
        ) rsp_arb (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (rsp_arb_valid_in),
            .data_in   (rsp_arb_data_in),
            .ready_in  (rsp_arb_ready_in),
            .valid_out (rsp_arb_valid_out),
            .data_out  (rsp_arb_data_out),
            .ready_out (lsu_mem_if[i].rsp_ready),
            `UNUSED_PIN (sel_out)
        );

        assign bypass_dcache_if[i].rsp_ready = rsp_arb_ready_in[0];
        assign ebuf_rsp_ready[i] = rsp_arb_ready_in[1];

        assign lsu_mem_if[i].rsp_valid = rsp_arb_valid_out;
        assign lsu_mem_if[i].rsp_data.data = rsp_arb_data_out[LSU_RSP_DATAW-1 -: DATA_WIDTH];
        assign lsu_mem_if[i].rsp_data.tag = rsp_arb_data_out[TAG_WIDTH-1:0];

    end

    // =========================================================================
    // Performance Counters
    // =========================================================================

`ifdef PERF_ENABLE
    assign mmu_perf.tlb_reads     = mmu_perf_tlb.tlb_reads;
    assign mmu_perf.tlb_hits      = mmu_perf_tlb.tlb_hits;
    assign mmu_perf.tlb_misses    = mmu_perf_tlb.tlb_misses;
    assign mmu_perf.tlb_evictions = mmu_perf_tlb.tlb_evictions;
    assign mmu_perf.ptw_walks     = mmu_perf_tlb.ptw_walks;
    assign mmu_perf.ptw_latency   = ptw_latency_counter;
`endif

endmodule
