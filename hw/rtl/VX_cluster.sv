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

module VX_cluster import VX_gpu_pkg::*; #(
    parameter CLUSTER_ID = 0,
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    // Clock
    input  wire                 clk,
    input  wire                 reset,

`ifdef PERF_ENABLE
    input sysmem_perf_t         sysmem_perf,
`endif

    // DCRs
    VX_dcr_bus_if.slave         dcr_bus_if,

    // Memory
    VX_mem_bus_if.master        mem_bus_if [`L2_MEM_PORTS],

    // Status
    output wire                 busy
);

`ifdef SCOPE
    localparam scope_socket = 0;
    `SCOPE_IO_SWITCH (NUM_SOCKETS);
`endif

`ifdef PERF_ENABLE
    cache_perf_t l2_perf;
    sysmem_perf_t sysmem_perf_tmp;
    always @(*) begin
        sysmem_perf_tmp = sysmem_perf;
        sysmem_perf_tmp.l2cache = l2_perf;
    end
`endif

`ifdef GBAR_ENABLE

    VX_gbar_bus_if per_socket_gbar_bus_if[NUM_SOCKETS]();
    VX_gbar_bus_if gbar_bus_if();

    VX_gbar_arb #(
        .NUM_REQS (NUM_SOCKETS),
        .OUT_BUF  ((NUM_SOCKETS > 2) ? 1 : 0) // bgar_unit has no backpressure
    ) gbar_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (per_socket_gbar_bus_if),
        .bus_out_if (gbar_bus_if)
    );

    VX_gbar_unit #(
        .INSTANCE_ID (`SFORMATF(("gbar%0d", CLUSTER_ID)))
    ) gbar_unit (
        .clk         (clk),
        .reset       (reset),
        .gbar_bus_if (gbar_bus_if)
    );

`endif

    // L2 input buses (post-arb tag width when DXA enabled)
    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) per_socket_mem_bus_if[L2_NUM_REQS]();

    // Socket L1 output buses (pre-arb, original tag width)
    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
    ) socket_mem_bus_if[L2_SOCKET_REQS]();

`ifdef EXT_DXA_ENABLE
    VX_dxa_req_bus_if per_socket_dxa_req_bus_if[NUM_SOCKETS]();
    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
    ) dxa_gmem_bus_if[DXA_L2_GMEM_PORTS]();
    localparam DXA_CORE_LOCAL_BITS = `CLOG2(`SOCKET_SIZE);
    localparam DXA_NUM_CORE_OUTPUTS = NUM_SOCKETS * `SOCKET_SIZE;
    VX_dxa_bank_wr_if #(
        .NUM_BANKS       (`LMEM_NUM_BANKS),
        .BANK_ADDR_WIDTH (DXA_SMEM_BANK_ADDR_WIDTH),
        .WORD_SIZE       (`XLEN / 8),
        .TAG_WIDTH       (DXA_BANK_WR_TAG_WIDTH)
    ) per_core_bank_wr_if[DXA_NUM_CORE_OUTPUTS]();
`endif

    `RESET_RELAY (l2_reset, reset);

    VX_mem_bus_if #(
        .DATA_SIZE (`L2_LINE_SIZE),
        .TAG_WIDTH (L2_MEM_TAG_WIDTH)
    ) l2_mem_bus_if[`L2_MEM_PORTS]();

    VX_cache_wrap #(
        .INSTANCE_ID    (`SFORMATF(("%s-l2cache", INSTANCE_ID))),
        .CACHE_SIZE     (`L2_CACHE_SIZE),
        .LINE_SIZE      (`L2_LINE_SIZE),
        .NUM_BANKS      (`L2_NUM_BANKS),
        .NUM_WAYS       (`L2_NUM_WAYS),
        .WORD_SIZE      (L2_WORD_SIZE),
        .NUM_REQS       (L2_NUM_REQS),
        .MEM_PORTS      (`L2_MEM_PORTS),
        .CRSQ_SIZE      (`L2_CRSQ_SIZE),
        .MSHR_SIZE      (`L2_MSHR_SIZE),
        .MRSQ_SIZE      (`L2_MRSQ_SIZE),
        .MREQ_SIZE      (`L2_WRITEBACK ? `L2_MSHR_SIZE : `L2_MREQ_SIZE),
        .TAG_WIDTH      (L2_TAG_WIDTH),
        .WRITE_ENABLE   (1),
        .WRITEBACK      (`L2_WRITEBACK),
        .DIRTY_BYTES    (`L2_DIRTYBYTES),
        .REPL_POLICY    (`L2_REPL_POLICY),
        .CORE_OUT_BUF   (3),
        .MEM_OUT_BUF    (3),
        .NC_ENABLE      (1),
        .PASSTHRU       (!`L2_ENABLED)
    ) l2cache (
        .clk            (clk),
        .reset          (l2_reset),
    `ifdef PERF_ENABLE
        .cache_perf     (l2_perf),
    `endif
        .core_bus_if    (per_socket_mem_bus_if),
        .mem_bus_if     (l2_mem_bus_if)
    );

`ifdef EXT_DXA_ENABLE
    VX_dxa_core #(
        .INSTANCE_ID      (`SFORMATF(("%s-dxa-core", INSTANCE_ID))),
        .DXA_NUM_SOCKETS  (NUM_SOCKETS),
        .NUM_DXA_UNITS    (`NUM_DXA_UNITS),
        .GMEM_OUT_PORTS   (DXA_L2_GMEM_PORTS),
        .CORE_LOCAL_BITS  (DXA_CORE_LOCAL_BITS),
        .NUM_CORE_OUTPUTS (DXA_NUM_CORE_OUTPUTS),
        .ENABLE           (1)
    ) dxa_core (
        .clk                   (clk),
        .reset                 (reset),
        .dcr_bus_if            (dcr_bus_if),
        .per_socket_dxa_bus_if (per_socket_dxa_req_bus_if),
        .per_core_bank_wr_if   (per_core_bank_wr_if),
        .dxa_gmem_bus_if       (dxa_gmem_bus_if)
    );

    // DXA+LSU arb: all L2_SOCKET_REQS ports go through the 2:1 priority arb
    // to maintain correct tag width (arb adds DXA_L2_ARB_TAG_BITS).
    // DXA only connects to the first DXA_L2_GMEM_PORTS slots;
    // remaining DXA slots are tied off (valid=0) so they never contend.
    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
    ) l2_arb_in_if[2 * L2_SOCKET_REQS]();

    // Bind DXA gmem ports (first DXA_L2_GMEM_PORTS slots)
    for (genvar i = 0; i < DXA_L2_GMEM_PORTS; ++i) begin : g_dxa_l2_bind
        `ASSIGN_VX_MEM_BUS_IF (l2_arb_in_if[i], dxa_gmem_bus_if[i]);
    end

    // Tie off unused DXA slots
    for (genvar i = DXA_L2_GMEM_PORTS; i < L2_SOCKET_REQS; ++i) begin : g_dxa_l2_tieoff
        assign l2_arb_in_if[i].req_valid = 1'b0;
        assign l2_arb_in_if[i].req_data  = '0;
        assign l2_arb_in_if[i].rsp_ready = 1'b1;
    end

    // Bind all LSU ports
    for (genvar i = 0; i < L2_SOCKET_REQS; ++i) begin : g_lsu_l2_bind
        `ASSIGN_VX_MEM_BUS_IF (l2_arb_in_if[L2_SOCKET_REQS + i], socket_mem_bus_if[i]);
    end

    VX_mem_arb #(
        .NUM_INPUTS  (2 * L2_SOCKET_REQS),
        .NUM_OUTPUTS (L2_SOCKET_REQS),
        .DATA_SIZE   (`L1_LINE_SIZE),
        .TAG_WIDTH   (L1_MEM_ARB_TAG_WIDTH),
        .TAG_SEL_IDX (0),
        .ARBITER     ("R")
    ) dxa_l2_priority_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (l2_arb_in_if),
        .bus_out_if (per_socket_mem_bus_if)
    );
`else
    // No DXA: direct socket → L2
    for (genvar i = 0; i < L2_SOCKET_REQS; ++i) begin : g_no_dxa_l2
        `ASSIGN_VX_MEM_BUS_IF (per_socket_mem_bus_if[i], socket_mem_bus_if[i]);
    end
`endif

    for (genvar i = 0; i < `L2_MEM_PORTS; ++i) begin : g_l2_mem_out
        `ASSIGN_VX_MEM_BUS_IF (mem_bus_if[i], l2_mem_bus_if[i]);
    end

    ///////////////////////////////////////////////////////////////////////////

    wire [NUM_SOCKETS-1:0] per_socket_busy;

    // Generate all sockets
    for (genvar socket_id = 0; socket_id < NUM_SOCKETS; ++socket_id) begin : g_sockets

        `RESET_RELAY (socket_reset, reset);

        VX_dcr_bus_if socket_dcr_bus_if();
        wire is_base_dcr_addr = (dcr_bus_if.write_addr >= `VX_DCR_BASE_STATE_BEGIN && dcr_bus_if.write_addr < `VX_DCR_BASE_STATE_END);
        `BUFFER_DCR_BUS_IF (socket_dcr_bus_if, dcr_bus_if, is_base_dcr_addr, (NUM_SOCKETS > 1))

        VX_socket #(
            .SOCKET_ID ((CLUSTER_ID * NUM_SOCKETS) + socket_id),
            .INSTANCE_ID (`SFORMATF(("%s-socket%0d", INSTANCE_ID, socket_id)))
        ) socket (
            `SCOPE_IO_BIND  (scope_socket+socket_id)

            .clk            (clk),
            .reset          (socket_reset),

        `ifdef PERF_ENABLE
            .sysmem_perf    (sysmem_perf_tmp),
        `endif

            .dcr_bus_if     (socket_dcr_bus_if),

            .mem_bus_if     (socket_mem_bus_if[socket_id * `L1_MEM_PORTS +: `L1_MEM_PORTS]),

        `ifdef EXT_DXA_ENABLE
            .dxa_req_bus_if     (per_socket_dxa_req_bus_if[socket_id]),
            .per_core_bank_wr_if(per_core_bank_wr_if[socket_id * `SOCKET_SIZE +: `SOCKET_SIZE]),
        `endif

        `ifdef GBAR_ENABLE
            .gbar_bus_if    (per_socket_gbar_bus_if[socket_id]),
        `endif

            .busy           (per_socket_busy[socket_id])
        );
    end

    `BUFFER_EX(busy, (| per_socket_busy), 1'b1, 1, (NUM_SOCKETS > 1));

endmodule
