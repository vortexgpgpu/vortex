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

module Vortex import VX_gpu_pkg::*; (
    `SCOPE_IO_DECL

    // Clock
    input  wire                             clk,
    input  wire                             reset,

    // Memory request
    output wire                             mem_req_valid [`VX_MEM_PORTS],
    output wire                             mem_req_rw [`VX_MEM_PORTS],
    output wire [`VX_MEM_BYTEEN_WIDTH-1:0]  mem_req_byteen [`VX_MEM_PORTS],
    output wire [`VX_MEM_ADDR_WIDTH-1:0]    mem_req_addr [`VX_MEM_PORTS],
    output wire [`VX_MEM_DATA_WIDTH-1:0]    mem_req_data [`VX_MEM_PORTS],
    output wire [`VX_MEM_TAG_WIDTH-1:0]     mem_req_tag [`VX_MEM_PORTS],
    input  wire                             mem_req_ready [`VX_MEM_PORTS],

    // Memory response
    input wire                              mem_rsp_valid [`VX_MEM_PORTS],
    input wire [`VX_MEM_DATA_WIDTH-1:0]     mem_rsp_data [`VX_MEM_PORTS],
    input wire [`VX_MEM_TAG_WIDTH-1:0]      mem_rsp_tag [`VX_MEM_PORTS],
    output wire                             mem_rsp_ready [`VX_MEM_PORTS],

    // DCR write request
    input  wire                             dcr_wr_valid,
    input  wire [`VX_DCR_ADDR_WIDTH-1:0]    dcr_wr_addr,
    input  wire [`VX_DCR_DATA_WIDTH-1:0]    dcr_wr_data,

    // Status
    output wire                             busy
);

`ifdef SCOPE
    localparam scope_cluster = 0;
    `SCOPE_IO_SWITCH (`NUM_CLUSTERS);
`endif

`ifdef PERF_ENABLE
    cache_perf_t l3_perf;
    mem_perf_t mem_perf;
    sysmem_perf_t sysmem_perf;
    always @(*) begin
        sysmem_perf = '0;
        sysmem_perf.l3cache = l3_perf;
        sysmem_perf.mem = mem_perf;
    end
`endif

    VX_mem_bus_if #(
        .DATA_SIZE (`L2_LINE_SIZE),
        .TAG_WIDTH (L2_MEM_TAG_WIDTH)
    ) per_cluster_mem_bus_if[`NUM_CLUSTERS * `L2_MEM_PORTS]();

    VX_mem_bus_if #(
        .DATA_SIZE (`L3_LINE_SIZE),
        .TAG_WIDTH (L3_MEM_TAG_WIDTH)
    ) mem_bus_if[`L3_MEM_PORTS]();

    `RESET_RELAY (l3_reset, reset);

    VX_cache_wrap #(
        .INSTANCE_ID    ("l3cache"),
        .CACHE_SIZE     (`L3_CACHE_SIZE),
        .LINE_SIZE      (`L3_LINE_SIZE),
        .NUM_BANKS      (`L3_NUM_BANKS),
        .NUM_WAYS       (`L3_NUM_WAYS),
        .WORD_SIZE      (L3_WORD_SIZE),
        .NUM_REQS       (L3_NUM_REQS),
        .MEM_PORTS      (`L3_MEM_PORTS),
        .CRSQ_SIZE      (`L3_CRSQ_SIZE),
        .MSHR_SIZE      (`L3_MSHR_SIZE),
        .MRSQ_SIZE      (`L3_MRSQ_SIZE),
        .MREQ_SIZE      (`L3_WRITEBACK ? `L3_MSHR_SIZE : `L3_MREQ_SIZE),
        .TAG_WIDTH      (L2_MEM_TAG_WIDTH),
        .WRITE_ENABLE   (1),
        .WRITEBACK      (`L3_WRITEBACK),
        .DIRTY_BYTES    (`L3_DIRTYBYTES),
        .REPL_POLICY    (`L3_REPL_POLICY),
        .UUID_WIDTH     (`UUID_WIDTH),
        .FLAGS_WIDTH    (`MEM_REQ_FLAGS_WIDTH),
        .CORE_OUT_BUF   (3),
        .MEM_OUT_BUF    (3),
        .NC_ENABLE      (1),
        .PASSTHRU       (!`L3_ENABLED)
    ) l3cache (
        .clk            (clk),
        .reset          (l3_reset),

    `ifdef PERF_ENABLE
        .cache_perf     (l3_perf),
    `endif

        .core_bus_if    (per_cluster_mem_bus_if),
        .mem_bus_if     (mem_bus_if)
    );

    for (genvar i = 0; i < `L3_MEM_PORTS; ++i) begin : g_mem_bus_if
        assign mem_req_valid[i]  = mem_bus_if[i].req_valid;
        assign mem_req_rw[i]     = mem_bus_if[i].req_data.rw;
        assign mem_req_byteen[i] = mem_bus_if[i].req_data.byteen;
        assign mem_req_addr[i]   = mem_bus_if[i].req_data.addr;
        assign mem_req_data[i]   = mem_bus_if[i].req_data.data;
        assign mem_req_tag[i]    = mem_bus_if[i].req_data.tag;
        `UNUSED_VAR (mem_bus_if[i].req_data.flags)
        assign mem_bus_if[i].req_ready = mem_req_ready[i];

        assign mem_bus_if[i].rsp_valid     = mem_rsp_valid[i];
        assign mem_bus_if[i].rsp_data.data = mem_rsp_data[i];
        assign mem_bus_if[i].rsp_data.tag  = mem_rsp_tag[i];
        assign mem_rsp_ready[i] = mem_bus_if[i].rsp_ready;
    end

    VX_dcr_bus_if dcr_bus_if();
    assign dcr_bus_if.write_valid = dcr_wr_valid;
    assign dcr_bus_if.write_addr  = dcr_wr_addr;
    assign dcr_bus_if.write_data  = dcr_wr_data;

    wire [`NUM_CLUSTERS-1:0] per_cluster_busy;

    // Generate all clusters
    for (genvar cluster_id = 0; cluster_id < `NUM_CLUSTERS; ++cluster_id) begin : g_clusters

        `RESET_RELAY (cluster_reset, reset);

        VX_dcr_bus_if cluster_dcr_bus_if();
        `BUFFER_DCR_BUS_IF (cluster_dcr_bus_if, dcr_bus_if, 1'b1, (`NUM_CLUSTERS > 1))

        VX_cluster #(
            .CLUSTER_ID (cluster_id),
            .INSTANCE_ID (`SFORMATF(("cluster%0d", cluster_id)))
        ) cluster (
            `SCOPE_IO_BIND (scope_cluster + cluster_id)

            .clk                (clk),
            .reset              (cluster_reset),

        `ifdef PERF_ENABLE
            .sysmem_perf        (sysmem_perf),
        `endif

            .dcr_bus_if         (cluster_dcr_bus_if),

            .mem_bus_if         (per_cluster_mem_bus_if[cluster_id * `L2_MEM_PORTS +: `L2_MEM_PORTS]),

            .busy               (per_cluster_busy[cluster_id])
        );
    end

    `BUFFER_EX(busy, (| per_cluster_busy), 1'b1, 1, (`NUM_CLUSTERS > 1));

`ifdef PERF_ENABLE

    localparam MEM_PORTS_CTR_W = `CLOG2(`VX_MEM_PORTS+1);

    wire [`VX_MEM_PORTS-1:0] mem_req_fire, mem_rsp_fire;
    wire [`VX_MEM_PORTS-1:0] mem_rd_req_fire, mem_wr_req_fire;

    for (genvar i = 0; i < `VX_MEM_PORTS; ++i) begin : g_perf_ctrs
        assign mem_req_fire[i] = mem_req_valid[i] & mem_req_ready[i];
        assign mem_rsp_fire[i] = mem_rsp_valid[i] & mem_rsp_ready[i];
        assign mem_rd_req_fire[i] = mem_req_fire[i] & ~mem_req_rw[i];
        assign mem_wr_req_fire[i] = mem_req_fire[i] & mem_req_rw[i];
    end

    wire [MEM_PORTS_CTR_W-1:0] perf_mem_reads_per_cycle;
    wire [MEM_PORTS_CTR_W-1:0] perf_mem_writes_per_cycle;
    wire [MEM_PORTS_CTR_W-1:0] perf_mem_rsps_per_cycle;

    `POP_COUNT(perf_mem_reads_per_cycle, mem_rd_req_fire);
    `POP_COUNT(perf_mem_writes_per_cycle, mem_wr_req_fire);
    `POP_COUNT(perf_mem_rsps_per_cycle, mem_rsp_fire);

    reg [`PERF_CTR_BITS-1:0] perf_mem_pending_reads;

    always @(posedge clk) begin
        if (reset) begin
            perf_mem_pending_reads <= '0;
        end else begin
            perf_mem_pending_reads <= $signed(perf_mem_pending_reads) +
                `PERF_CTR_BITS'($signed((MEM_PORTS_CTR_W+1)'(perf_mem_reads_per_cycle) - (MEM_PORTS_CTR_W+1)'(perf_mem_rsps_per_cycle)));
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            mem_perf <= '0;
        end else begin
            mem_perf.reads <= mem_perf.reads + `PERF_CTR_BITS'(perf_mem_reads_per_cycle);
            mem_perf.writes <= mem_perf.writes + `PERF_CTR_BITS'(perf_mem_writes_per_cycle);
            mem_perf.latency <= mem_perf.latency + perf_mem_pending_reads;
        end
    end

`endif

    // dump device configuration
    initial begin
        `TRACE(0, ("CONFIGS: num_threads=%0d, num_warps=%0d, num_cores=%0d, num_clusters=%0d, socket_size=%0d, local_mem_base=0x%0h, num_barriers=%0d\n",
                    `NUM_THREADS, `NUM_WARPS, `NUM_CORES, `NUM_CLUSTERS, `SOCKET_SIZE, `LMEM_BASE_ADDR, `NUM_BARRIERS))
    end

`ifdef DBG_TRACE_MEM
    for (genvar i = 0; i < `VX_MEM_PORTS; ++i) begin : g_trace
        always @(posedge clk) begin
            if (mem_bus_if[i].req_valid && mem_bus_if[i].req_ready) begin
                if (mem_bus_if[i].req_data.rw) begin
                    `TRACE(2, ("%t: MEM Wr Req[%0d]: addr=0x%0h, byteen=0x%h data=0x%h, tag=0x%0h (#%0d)\n", $time, i, `TO_FULL_ADDR(mem_bus_if[i].req_data.addr), mem_bus_if[i].req_data.byteen, mem_bus_if[i].req_data.data, mem_bus_if[i].req_data.tag.value, mem_bus_if[i].req_data.tag.uuid))
                end else begin
                    `TRACE(2, ("%t: MEM Rd Req[%0d]: addr=0x%0h, byteen=0x%h, tag=0x%0h (#%0d)\n", $time, i, `TO_FULL_ADDR(mem_bus_if[i].req_data.addr), mem_bus_if[i].req_data.byteen, mem_bus_if[i].req_data.tag.value, mem_bus_if[i].req_data.tag.uuid))
                end
            end
            if (mem_bus_if[i].rsp_valid && mem_bus_if[i].rsp_ready) begin
                `TRACE(2, ("%t: MEM Rd Rsp[%0d]: data=0x%h, tag=0x%0h (#%0d)\n", $time, i, mem_bus_if[i].rsp_data.data, mem_bus_if[i].rsp_data.tag.value, mem_bus_if[i].rsp_data.tag.uuid))
            end
        end
    end
`endif

`ifdef SIMULATION
    always @(posedge clk) begin
        $fflush(); // flush stdout buffer
    end
`endif

endmodule
