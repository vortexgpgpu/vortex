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

// The memory path of one LSU block: its memory scheduler, the global/local
// switch, the local-memory adapter, the coalescer and the D-cache adapter.
// The D-cache port carries one extra tag LSB: requests leave with it clear, and
// a response with it set is the core's flush token, not this block's.
module VX_lsu_mem_path import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output wire [`VX_CFG_NUM_LSU_LANES-1:0] perf_rd_req_fire,
    output wire [`VX_CFG_NUM_LSU_LANES-1:0] perf_wr_req_fire,
    output wire [`VX_CFG_NUM_LSU_LANES-1:0] perf_rsp_fire,
    output wire [PERF_CTR_BITS-1:0] coalescer_misses,
`endif

    VX_lsu_sched_if.slave   client_if,
`ifdef TCU_META_ENABLE
    VX_lsu_sched_if.slave   tcu_mem_if,
`endif

    // the scheduler holds no request; the global-store path holds no store
    output wire             sched_empty,
    output wire             store_empty,

`ifdef VX_CFG_LMEM_ENABLE
    VX_mem_bus_if.master    lmem_bus_if [`VX_CFG_NUM_LSU_LANES],
`endif
    VX_mem_bus_if.master    dcache_bus_if [DCACHE_CHANNELS]
);
    `UNUSED_SPARAM (INSTANCE_ID)

    VX_lsu_mem_if #(
        .NUM_LANES (`VX_CFG_NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_mem_if();

    // scheduler: client 0 is the block's lanes, client 1 the TCU metadata AGU

`ifdef TCU_META_ENABLE
    localparam NUM_CLIENTS = 2;
`else
    localparam NUM_CLIENTS = 1;
`endif

    VX_lsu_sched_if sched_client_if [NUM_CLIENTS]();

    assign sched_client_if[0].req_valid = client_if.req_valid;
    assign sched_client_if[0].req_data  = client_if.req_data;
    assign client_if.req_ready          = sched_client_if[0].req_ready;
    assign client_if.rsp_valid          = sched_client_if[0].rsp_valid;
    assign client_if.rsp_data           = sched_client_if[0].rsp_data;
    assign sched_client_if[0].rsp_ready = client_if.rsp_ready;

`ifdef TCU_META_ENABLE
    assign sched_client_if[1].req_valid = tcu_mem_if.req_valid;
    assign sched_client_if[1].req_data  = tcu_mem_if.req_data;
    assign tcu_mem_if.req_ready         = sched_client_if[1].req_ready;
    assign tcu_mem_if.rsp_valid         = sched_client_if[1].rsp_valid;
    assign tcu_mem_if.rsp_data          = sched_client_if[1].rsp_data;
    assign sched_client_if[1].rsp_ready = tcu_mem_if.rsp_ready;
`endif

    VX_lsu_scheduler #(
        .INSTANCE_ID     (`SFORMATF(("%s-lsusched", INSTANCE_ID))),
        .NUM_CLIENTS     (NUM_CLIENTS),
        .NUM_LANES       (`VX_CFG_NUM_LSU_LANES),
        .CORE_QUEUE_SIZE (`VX_CFG_LSU_QUEUE_IN_SIZE),
        .MEM_QUEUE_SIZE  (LSU_QUEUE_OUT_SIZE),
        .PENDING_SIZE    (`VX_CFG_LSU_PENDING_SIZE)
    ) lsu_scheduler (
        .clk        (clk),
        .reset      (reset),
        .client_if  (sched_client_if),
        .empty      (sched_empty),
        .lsu_mem_if (lsu_mem_if)
    );

`ifdef PERF_ENABLE
    for (genvar j = 0; j < `VX_CFG_NUM_LSU_LANES; ++j) begin : g_perf
        assign perf_rd_req_fire[j] = lsu_mem_if.req_valid && lsu_mem_if.req_data.mask[j] && lsu_mem_if.req_ready && ~lsu_mem_if.req_data.rw;
        assign perf_wr_req_fire[j] = lsu_mem_if.req_valid && lsu_mem_if.req_data.mask[j] && lsu_mem_if.req_ready && lsu_mem_if.req_data.rw;
        assign perf_rsp_fire[j]    = lsu_mem_if.rsp_valid && lsu_mem_if.rsp_data.mask[j] && lsu_mem_if.rsp_ready;
    end
`endif

    // Two terms keep a store visible until it reaches the D-cache port: the
    // coalescer (or the adapter input without one), and the adapter's request
    // buffer.
    wire path_empty;
    wire adapter_empty;
    assign store_empty = path_empty && adapter_empty;

    VX_lsu_mem_if #(
        .NUM_LANES (`VX_CFG_NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_dcache_if();

`ifdef VX_CFG_LMEM_ENABLE

    VX_lsu_mem_if #(
        .NUM_LANES (`VX_CFG_NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_lmem_if();

    VX_lmem_switch #(
        .GLOBAL_OUT_BUF(1),
        .LOCAL_OUT_BUF(1),
        .RSP_OUT_BUF  (1),
        .ARBITER      ("P")
    ) lmem_switch (
        .clk          (clk),
        .reset        (reset),
        .lsu_in_if    (lsu_mem_if),
        .global_out_if(lsu_dcache_if),
        .local_out_if (lsu_lmem_if)
    );

    // A dedicated local-memory adapter per block: sharing one adapter's
    // unpack/pack buffers between blocks would deadlock.
    VX_lsu_adapter #(
        .NUM_LANES    (`VX_CFG_NUM_LSU_LANES),
        .DATA_SIZE    (LSU_WORD_SIZE),
        .TAG_WIDTH    (LSU_TAG_WIDTH),
        .TAG_SEL_BITS (LSU_TAG_WIDTH - UUID_WIDTH),
        .ARBITER      ("P"),
        .REQ_OUT_BUF  (3),
        .RSP_OUT_BUF  (0)
    ) lmem_adapter (
        .clk        (clk),
        .reset      (reset),
        .lsu_mem_if (lsu_lmem_if),
        .mem_bus_if (lmem_bus_if)
    );

`else

    `ASSIGN_VX_MEM_BUS_IF (lsu_dcache_if, lsu_mem_if);

`endif

    VX_lsu_mem_if #(
        .NUM_LANES (DCACHE_CHANNELS),
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_CORE_TAG_WIDTH)
    ) dcache_coalesced_if();

    if ((`VX_CFG_NUM_LSU_LANES > 1) && (DCACHE_WORD_SIZE > LSU_WORD_SIZE)) begin : g_coalescing

        VX_mem_coalescer #(
            .INSTANCE_ID    (`SFORMATF(("%s-coalescer", INSTANCE_ID))),
            .NUM_REQS       (`VX_CFG_NUM_LSU_LANES),
            .DATA_IN_SIZE   (LSU_WORD_SIZE),
            .DATA_OUT_SIZE  (DCACHE_WORD_SIZE),
            .ADDR_WIDTH     (LSU_ADDR_WIDTH),
            .USER_WIDTH     (MEM_ATTR_WIDTH),
            .TAG_WIDTH      (LSU_TAG_WIDTH),
            .UUID_WIDTH     (UUID_WIDTH),
            .QUEUE_SIZE     (LSU_QUEUE_OUT_SIZE),
            .PERF_CTR_BITS  (PERF_CTR_BITS)
        ) mem_coalescer (
            .clk            (clk),
            .reset          (reset),

        `ifdef PERF_ENABLE
            .misses         (coalescer_misses),
        `else
            `UNUSED_PIN (misses),
        `endif

            .empty          (path_empty),

            // Input request
            .in_req_valid   (lsu_dcache_if.req_valid),
            .in_req_mask    (lsu_dcache_if.req_data.mask),
            .in_req_rw      (lsu_dcache_if.req_data.rw),
            .in_req_byteen  (lsu_dcache_if.req_data.byteen),
            .in_req_addr    (lsu_dcache_if.req_data.addr),
            .in_req_user    (lsu_dcache_if.req_data.user),
            .in_req_no_merge(lsu_dcache_if.req_data.user[0][MEM_ATTR_AMO_OFFS]),
            .in_req_data    (lsu_dcache_if.req_data.data),
            .in_req_tag     (lsu_dcache_if.req_data.tag),
            .in_req_ready   (lsu_dcache_if.req_ready),

            // Input response
            .in_rsp_valid   (lsu_dcache_if.rsp_valid),
            .in_rsp_mask    (lsu_dcache_if.rsp_data.mask),
            .in_rsp_data    (lsu_dcache_if.rsp_data.data),
            .in_rsp_tag     (lsu_dcache_if.rsp_data.tag),
            .in_rsp_ready   (lsu_dcache_if.rsp_ready),

            // Output request
            .out_req_valid  (dcache_coalesced_if.req_valid),
            .out_req_mask   (dcache_coalesced_if.req_data.mask),
            .out_req_rw     (dcache_coalesced_if.req_data.rw),
            .out_req_byteen (dcache_coalesced_if.req_data.byteen),
            .out_req_addr   (dcache_coalesced_if.req_data.addr),
            .out_req_user   (dcache_coalesced_if.req_data.user),
            .out_req_data   (dcache_coalesced_if.req_data.data),
            .out_req_tag    (dcache_coalesced_if.req_data.tag),
            .out_req_ready  (dcache_coalesced_if.req_ready),

            // Output response
            .out_rsp_valid  (dcache_coalesced_if.rsp_valid),
            .out_rsp_mask   (dcache_coalesced_if.rsp_data.mask),
            .out_rsp_data   (dcache_coalesced_if.rsp_data.data),
            .out_rsp_tag    (dcache_coalesced_if.rsp_data.tag),
            .out_rsp_ready  (dcache_coalesced_if.rsp_ready)
        );

    end else begin : g_no_coalescing

        // The passthrough narrows the tag from LSU_TAG_WIDTH to
        // DCACHE_CORE_TAG_WIDTH: the dcache tag-id space must cover the
        // mem-scheduler slot id or responses alias.
        `STATIC_ASSERT (DCACHE_CORE_TAG_WIDTH >= LSU_TAG_WIDTH, ("dcache tag-id space cannot hold the LSU outstanding slot id"))

        `ASSIGN_VX_MEM_BUS_IF (dcache_coalesced_if, lsu_dcache_if);
    `ifdef PERF_ENABLE
        assign coalescer_misses = '0;
    `endif
        // Without a coalescer, a store still upstream of the adapter is one
        // presented at its input.
        assign path_empty = ~dcache_coalesced_if.req_valid;

    end

    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_CORE_TAG_WIDTH)
    ) dcache_adapter_if[DCACHE_CHANNELS]();

    VX_lsu_adapter #(
        .NUM_LANES    (DCACHE_CHANNELS),
        .DATA_SIZE    (DCACHE_WORD_SIZE),
        .TAG_WIDTH    (DCACHE_CORE_TAG_WIDTH),
        .TAG_SEL_BITS (DCACHE_CORE_TAG_WIDTH - UUID_WIDTH),
        .ARBITER      ("P"),
        .REQ_OUT_BUF  (3),
        .RSP_OUT_BUF  (0)
    ) dcache_adapter (
        .clk        (clk),
        .reset      (reset),
        .lsu_mem_if (dcache_coalesced_if),
        .mem_bus_if (dcache_adapter_if)
    );

    wire [DCACHE_CHANNELS-1:0] adapter_req_valid;
    for (genvar j = 0; j < DCACHE_CHANNELS; ++j) begin : g_dcache_bus_if
        assign adapter_req_valid[j] = dcache_adapter_if[j].req_valid;

        assign dcache_bus_if[j].req_valid       = dcache_adapter_if[j].req_valid;
        assign dcache_bus_if[j].req_data.rw     = dcache_adapter_if[j].req_data.rw;
        assign dcache_bus_if[j].req_data.addr   = dcache_adapter_if[j].req_data.addr;
        assign dcache_bus_if[j].req_data.data   = dcache_adapter_if[j].req_data.data;
        assign dcache_bus_if[j].req_data.byteen = dcache_adapter_if[j].req_data.byteen;
        assign dcache_bus_if[j].req_data.attr   = dcache_adapter_if[j].req_data.attr;
        assign dcache_bus_if[j].req_data.tag    = {dcache_adapter_if[j].req_data.tag, 1'b0};
        assign dcache_adapter_if[j].req_ready   = dcache_bus_if[j].req_ready;

        assign dcache_adapter_if[j].rsp_valid     = dcache_bus_if[j].rsp_valid && ~dcache_bus_if[j].rsp_data.tag[0];
        assign dcache_adapter_if[j].rsp_data.data = dcache_bus_if[j].rsp_data.data;
        assign dcache_adapter_if[j].rsp_data.tag  = dcache_bus_if[j].rsp_data.tag[DCACHE_TAG_WIDTH-1:1];
        assign dcache_bus_if[j].rsp_ready         = dcache_adapter_if[j].rsp_ready;
    end
    assign adapter_empty = ~(| adapter_req_valid);

endmodule
