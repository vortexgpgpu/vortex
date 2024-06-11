// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

`ifdef EXT_F_ENABLE
`include "VX_fpu_define.vh"
`endif

module VX_core import VX_gpu_pkg::*; #(
    parameter CORE_ID = 0
) (
    `SCOPE_IO_DECL

    // Clock
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    VX_mem_perf_if.slave    mem_perf_if,
`endif

    VX_dcr_bus_if.slave     dcr_bus_if,

    VX_mem_bus_if.master    dcache_bus_if [DCACHE_NUM_REQS],

    VX_mem_bus_if.master    icache_bus_if,

`ifdef GBAR_ENABLE
    VX_gbar_bus_if.master   gbar_bus_if,
`endif

    // Status
    output wire             busy
);
    VX_schedule_if      schedule_if();
    VX_fetch_if         fetch_if();
    VX_decode_if        decode_if();
    VX_sched_csr_if     sched_csr_if();
    VX_decode_sched_if  decode_sched_if();
    VX_commit_sched_if  commit_sched_if();
    VX_commit_csr_if    commit_csr_if();
    VX_branch_ctl_if    branch_ctl_if[`NUM_ALU_BLOCKS]();
    VX_warp_ctl_if      warp_ctl_if();

    VX_dispatch_if      dispatch_if[`NUM_EX_UNITS * `ISSUE_WIDTH]();
    VX_commit_if        commit_if[`NUM_EX_UNITS * `ISSUE_WIDTH]();
    VX_writeback_if     writeback_if[`ISSUE_WIDTH]();

    VX_lsu_mem_if #(
        .NUM_LANES (`NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_mem_if[`NUM_LSU_BLOCKS]();

`ifdef PERF_ENABLE
    VX_mem_perf_if mem_perf_tmp_if();
    VX_pipeline_perf_if pipeline_perf_if();

    assign mem_perf_tmp_if.icache  = mem_perf_if.icache;
    assign mem_perf_tmp_if.dcache  = mem_perf_if.dcache;
    assign mem_perf_tmp_if.l2cache = mem_perf_if.l2cache;
    assign mem_perf_tmp_if.l3cache = mem_perf_if.l3cache;
    assign mem_perf_tmp_if.mem     = mem_perf_if.mem;
`endif

    `RESET_RELAY (dcr_data_reset, reset);
    `RESET_RELAY (schedule_reset, reset);
    `RESET_RELAY (fetch_reset, reset);
    `RESET_RELAY (decode_reset, reset);
    `RESET_RELAY (issue_reset, reset);
    `RESET_RELAY (execute_reset, reset);
    `RESET_RELAY (commit_reset, reset);

    base_dcrs_t base_dcrs;

    VX_dcr_data dcr_data (
        .clk        (clk),
        .reset      (dcr_data_reset),
        .dcr_bus_if (dcr_bus_if),
        .base_dcrs  (base_dcrs)
    );

    `SCOPE_IO_SWITCH (3)

    VX_schedule #(
        .CORE_ID (CORE_ID)
    ) schedule (
        .clk            (clk),
        .reset          (schedule_reset),

    `ifdef PERF_ENABLE
        .perf_schedule_if (pipeline_perf_if.schedule),
    `endif

        .base_dcrs      (base_dcrs),

        .warp_ctl_if    (warp_ctl_if),
        .branch_ctl_if  (branch_ctl_if),

        .decode_sched_if(decode_sched_if),
        .commit_sched_if(commit_sched_if),

        .schedule_if    (schedule_if),
    `ifdef GBAR_ENABLE
        .gbar_bus_if    (gbar_bus_if),
    `endif
        .sched_csr_if   (sched_csr_if),

        .busy           (busy)
    );

    VX_fetch #(
        .CORE_ID (CORE_ID)
    ) fetch (
        `SCOPE_IO_BIND  (0)
        .clk            (clk),
        .reset          (fetch_reset),
        .icache_bus_if  (icache_bus_if),
        .schedule_if    (schedule_if),
        .fetch_if       (fetch_if)
    );

    VX_decode #(
        .CORE_ID (CORE_ID)
    ) decode (
        .clk            (clk),
        .reset          (decode_reset),
        .fetch_if       (fetch_if),
        .decode_if      (decode_if),
        .decode_sched_if(decode_sched_if)
    );

    VX_issue #(
        .CORE_ID (CORE_ID)
    ) issue (
        `SCOPE_IO_BIND  (1)

        .clk            (clk),
        .reset          (issue_reset),

    `ifdef PERF_ENABLE
        .perf_issue_if  (pipeline_perf_if.issue),
    `endif

        .decode_if      (decode_if),
        .writeback_if   (writeback_if),
        .dispatch_if    (dispatch_if)
    );

    VX_execute #(
        .CORE_ID (CORE_ID)
    ) execute (
        `SCOPE_IO_BIND  (2)

        .clk            (clk),
        .reset          (execute_reset),

    `ifdef PERF_ENABLE
        .mem_perf_if    (mem_perf_tmp_if),
        .pipeline_perf_if(pipeline_perf_if),
    `endif

        .base_dcrs      (base_dcrs),

        .lsu_mem_if     (lsu_mem_if),

        .dispatch_if    (dispatch_if),
        .commit_if      (commit_if),

        .commit_csr_if  (commit_csr_if),
        .sched_csr_if   (sched_csr_if),

        .warp_ctl_if    (warp_ctl_if),
        .branch_ctl_if  (branch_ctl_if)
    );

    VX_commit #(
        .CORE_ID (CORE_ID)
    ) commit (
        .clk            (clk),
        .reset          (commit_reset),

        .commit_if      (commit_if),

        .writeback_if   (writeback_if),

        .commit_csr_if  (commit_csr_if),
        .commit_sched_if(commit_sched_if)
    );

    VX_lsu_mem_if #(
        .NUM_LANES (`NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_dcache_if[`NUM_LSU_BLOCKS]();

`ifdef LMEM_ENABLE

    `RESET_RELAY (lmem_unit_reset, reset);

    VX_lmem_unit #(
        .CORE_ID (CORE_ID)
    ) lmem_unit (
        .clk            (clk),
        .reset          (lmem_unit_reset),
    `ifdef PERF_ENABLE
        .cache_perf     (mem_perf_tmp_if.lmem),
    `endif
        .lsu_mem_in_if  (lsu_mem_if),
        .lsu_mem_out_if (lsu_dcache_if)
    );

`else

    for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin
        `ASSIGN_VX_LSU_MEM_IF (lsu_dcache_if[i], lsu_mem_if[i]);
    end

`endif

    VX_lsu_mem_if #(
        .NUM_LANES (DCACHE_CHANNELS),
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_TAG_WIDTH)
    ) dcache_coalesced_if[`NUM_LSU_BLOCKS]();

    if (LSU_WORD_SIZE != DCACHE_WORD_SIZE) begin

        `RESET_RELAY (coalescer_reset, reset);

        for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin

            VX_mem_coalescer #(
                .INSTANCE_ID    ($sformatf("core%0d-coalescer", CORE_ID)),
                .NUM_REQS       (`NUM_LSU_LANES),
                .DATA_IN_SIZE   (LSU_WORD_SIZE),
                .DATA_OUT_SIZE  (DCACHE_WORD_SIZE),
                .ADDR_WIDTH     (LSU_ADDR_WIDTH),
                .ATYPE_WIDTH    (`ADDR_TYPE_WIDTH),
                .TAG_WIDTH      (LSU_TAG_WIDTH),
                .UUID_WIDTH     (`UUID_WIDTH),
                .QUEUE_SIZE     (`LSUQ_OUT_SIZE)
            ) coalescer (
                .clk   (clk),
                .reset (coalescer_reset),

                // Input request
                .in_req_valid   (lsu_dcache_if[i].req_valid),
                .in_req_mask    (lsu_dcache_if[i].req_data.mask),
                .in_req_rw      (lsu_dcache_if[i].req_data.rw),
                .in_req_byteen  (lsu_dcache_if[i].req_data.byteen),
                .in_req_addr    (lsu_dcache_if[i].req_data.addr),
                .in_req_atype   (lsu_dcache_if[i].req_data.atype),
                .in_req_data    (lsu_dcache_if[i].req_data.data),
                .in_req_tag     (lsu_dcache_if[i].req_data.tag),
                .in_req_ready   (lsu_dcache_if[i].req_ready),

                // Input response
                .in_rsp_valid   (lsu_dcache_if[i].rsp_valid),
                .in_rsp_mask    (lsu_dcache_if[i].rsp_data.mask),
                .in_rsp_data    (lsu_dcache_if[i].rsp_data.data),
                .in_rsp_tag     (lsu_dcache_if[i].rsp_data.tag),
                .in_rsp_ready   (lsu_dcache_if[i].rsp_ready),

                // Output request
                .out_req_valid  (dcache_coalesced_if[i].req_valid),
                .out_req_mask   (dcache_coalesced_if[i].req_data.mask),
                .out_req_rw     (dcache_coalesced_if[i].req_data.rw),
                .out_req_byteen (dcache_coalesced_if[i].req_data.byteen),
                .out_req_addr   (dcache_coalesced_if[i].req_data.addr),
                .out_req_atype  (dcache_coalesced_if[i].req_data.atype),
                .out_req_data   (dcache_coalesced_if[i].req_data.data),
                .out_req_tag    (dcache_coalesced_if[i].req_data.tag),
                .out_req_ready  (dcache_coalesced_if[i].req_ready),

                // Output response
                .out_rsp_valid  (dcache_coalesced_if[i].rsp_valid),
                .out_rsp_mask   (dcache_coalesced_if[i].rsp_data.mask),
                .out_rsp_data   (dcache_coalesced_if[i].rsp_data.data),
                .out_rsp_tag    (dcache_coalesced_if[i].rsp_data.tag),
                .out_rsp_ready  (dcache_coalesced_if[i].rsp_ready)
            );
        end

    end else begin

        for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin
            `ASSIGN_VX_LSU_MEM_IF (dcache_coalesced_if[i], lsu_dcache_if[i]);
        end

    end

    `RESET_RELAY (lsu_adapter_reset, reset);

    for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin

        VX_mem_bus_if #(
            .DATA_SIZE (DCACHE_WORD_SIZE),
            .TAG_WIDTH (DCACHE_TAG_WIDTH)
        ) dcache_bus_tmp_if[DCACHE_CHANNELS]();

        VX_lsu_adapter #(
            .NUM_LANES    (DCACHE_CHANNELS),
            .DATA_SIZE    (DCACHE_WORD_SIZE),
            .TAG_WIDTH    (DCACHE_TAG_WIDTH),
            .TAG_SEL_BITS (DCACHE_TAG_WIDTH - `UUID_WIDTH),
            .REQ_OUT_BUF  (0),
            .RSP_OUT_BUF  (0)
        ) lsu_adapter (
            .clk        (clk),
            .reset      (lsu_adapter_reset),
            .lsu_mem_if (dcache_coalesced_if[i]),
            .mem_bus_if (dcache_bus_tmp_if)
        );

        for (genvar j = 0; j < DCACHE_CHANNELS; ++j) begin
            `ASSIGN_VX_MEM_BUS_IF (dcache_bus_if[i * DCACHE_CHANNELS + j], dcache_bus_tmp_if[j]);
        end
    end

`ifdef PERF_ENABLE

    wire [`CLOG2(LSU_NUM_REQS+1)-1:0] perf_dcache_rd_req_per_cycle;
    wire [`CLOG2(LSU_NUM_REQS+1)-1:0] perf_dcache_wr_req_per_cycle;
    wire [`CLOG2(LSU_NUM_REQS+1)-1:0] perf_dcache_rsp_per_cycle;

    wire [1:0] perf_icache_pending_read_cycle;
    wire [`CLOG2(LSU_NUM_REQS+1)+1-1:0] perf_dcache_pending_read_cycle;

    reg [`PERF_CTR_BITS-1:0] perf_icache_pending_reads;
    reg [`PERF_CTR_BITS-1:0] perf_dcache_pending_reads;

    reg [`PERF_CTR_BITS-1:0] perf_ifetches;
    reg [`PERF_CTR_BITS-1:0] perf_loads;
    reg [`PERF_CTR_BITS-1:0] perf_stores;

    wire perf_icache_req_fire = icache_bus_if.req_valid && icache_bus_if.req_ready;
    wire perf_icache_rsp_fire = icache_bus_if.rsp_valid && icache_bus_if.rsp_ready;

    wire [LSU_NUM_REQS-1:0] perf_dcache_rd_req_fire, perf_dcache_rd_req_fire_r;
    wire [LSU_NUM_REQS-1:0] perf_dcache_wr_req_fire, perf_dcache_wr_req_fire_r;
    wire [LSU_NUM_REQS-1:0] perf_dcache_rsp_fire;

    for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin
        for (genvar j = 0; j < `NUM_LSU_LANES; ++j) begin
            assign perf_dcache_rd_req_fire[i * `NUM_LSU_LANES + j] = lsu_mem_if[i].req_valid && lsu_mem_if[i].req_data.mask[j] && lsu_mem_if[i].req_ready && ~lsu_mem_if[i].req_data.rw;
            assign perf_dcache_wr_req_fire[i * `NUM_LSU_LANES + j] = lsu_mem_if[i].req_valid && lsu_mem_if[i].req_data.mask[j] && lsu_mem_if[i].req_ready && lsu_mem_if[i].req_data.rw;
            assign perf_dcache_rsp_fire[i * `NUM_LSU_LANES + j] = lsu_mem_if[i].rsp_valid && lsu_mem_if[i].rsp_data.mask[j] && lsu_mem_if[i].rsp_ready;
        end
    end

    `BUFFER(perf_dcache_rd_req_fire_r, perf_dcache_rd_req_fire);
    `BUFFER(perf_dcache_wr_req_fire_r, perf_dcache_wr_req_fire);

    `POP_COUNT(perf_dcache_rd_req_per_cycle, perf_dcache_rd_req_fire_r);
    `POP_COUNT(perf_dcache_wr_req_per_cycle, perf_dcache_wr_req_fire_r);
    `POP_COUNT(perf_dcache_rsp_per_cycle, perf_dcache_rsp_fire);

    assign perf_icache_pending_read_cycle = perf_icache_req_fire - perf_icache_rsp_fire;
    assign perf_dcache_pending_read_cycle = perf_dcache_rd_req_per_cycle - perf_dcache_rsp_per_cycle;

    always @(posedge clk) begin
        if (reset) begin
            perf_icache_pending_reads <= '0;
            perf_dcache_pending_reads <= '0;
        end else begin
            perf_icache_pending_reads <= $signed(perf_icache_pending_reads) + `PERF_CTR_BITS'($signed(perf_icache_pending_read_cycle));
            perf_dcache_pending_reads <= $signed(perf_dcache_pending_reads) + `PERF_CTR_BITS'($signed(perf_dcache_pending_read_cycle));
        end
    end

    reg [`PERF_CTR_BITS-1:0] perf_icache_lat;
    reg [`PERF_CTR_BITS-1:0] perf_dcache_lat;

    always @(posedge clk) begin
        if (reset) begin
            perf_ifetches   <= '0;
            perf_loads      <= '0;
            perf_stores     <= '0;
            perf_icache_lat <= '0;
            perf_dcache_lat <= '0;
        end else begin
            perf_ifetches   <= perf_ifetches   + `PERF_CTR_BITS'(perf_icache_req_fire);
            perf_loads      <= perf_loads      + `PERF_CTR_BITS'(perf_dcache_rd_req_per_cycle);
            perf_stores     <= perf_stores     + `PERF_CTR_BITS'(perf_dcache_wr_req_per_cycle);
            perf_icache_lat <= perf_icache_lat + perf_icache_pending_reads;
            perf_dcache_lat <= perf_dcache_lat + perf_dcache_pending_reads;
        end
    end

    assign pipeline_perf_if.ifetches = perf_ifetches;
    assign pipeline_perf_if.loads = perf_loads;
    assign pipeline_perf_if.stores = perf_stores;
    assign pipeline_perf_if.load_latency = perf_dcache_lat;
    assign pipeline_perf_if.ifetch_latency = perf_icache_lat;
    assign pipeline_perf_if.load_latency = perf_dcache_lat;

`endif

endmodule
