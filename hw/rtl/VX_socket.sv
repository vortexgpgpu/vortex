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

module VX_socket import VX_gpu_pkg::*; #(
    parameter SOCKET_ID = 0,
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    // Clock
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    input sysmem_perf_t     sysmem_perf,
`endif

    // DCRs
    VX_dcr_bus_if.slave     dcr_bus_if,

    // Memory
    VX_mem_bus_if.master    mem_bus_if [`L1_MEM_PORTS],

`ifdef GBAR_ENABLE
    // Barrier
    VX_gbar_bus_if.master   gbar_bus_if,
`endif
    // Status
    output wire             busy
);

`ifdef SCOPE
    localparam scope_core = 0;
    `SCOPE_IO_SWITCH (`SOCKET_SIZE);
`endif

`ifdef GBAR_ENABLE
    VX_gbar_bus_if per_core_gbar_bus_if[`SOCKET_SIZE]();

    VX_gbar_arb #(
        .NUM_REQS (`SOCKET_SIZE),
        .OUT_BUF  ((`SOCKET_SIZE > 1) ? 2 : 0)
    ) gbar_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (per_core_gbar_bus_if),
        .bus_out_if (gbar_bus_if)
    );
`endif

    ///////////////////////////////////////////////////////////////////////////

`ifdef PERF_ENABLE
    cache_perf_t icache_perf, dcache_perf;
    sysmem_perf_t sysmem_perf_tmp;
    always @(*) begin
        sysmem_perf_tmp = sysmem_perf;
        sysmem_perf_tmp.icache = icache_perf;
        sysmem_perf_tmp.dcache = dcache_perf;
    end
`endif

    ///////////////////////////////////////////////////////////////////////////

    VX_mem_bus_if #(
        .DATA_SIZE (ICACHE_WORD_SIZE),
        .TAG_WIDTH (ICACHE_TAG_WIDTH)
    ) per_core_icache_bus_if[`SOCKET_SIZE]();

    VX_mem_bus_if #(
        .DATA_SIZE (ICACHE_LINE_SIZE),
        .TAG_WIDTH (ICACHE_MEM_TAG_WIDTH)
    ) icache_mem_bus_if[1]();

    `RESET_RELAY (icache_reset, reset);

    VX_cache_cluster #(
        .INSTANCE_ID    (`SFORMATF(("%s-icache", INSTANCE_ID))),
        .NUM_UNITS      (`NUM_ICACHES),
        .NUM_INPUTS     (`SOCKET_SIZE),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`ICACHE_SIZE),
        .LINE_SIZE      (ICACHE_LINE_SIZE),
        .NUM_BANKS      (1),
        .NUM_WAYS       (`ICACHE_NUM_WAYS),
        .WORD_SIZE      (ICACHE_WORD_SIZE),
        .NUM_REQS       (1),
        .MEM_PORTS      (1),
        .CRSQ_SIZE      (`ICACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`ICACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`ICACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`ICACHE_MREQ_SIZE),
        .TAG_WIDTH      (ICACHE_TAG_WIDTH),
        .WRITE_ENABLE   (0),
        .REPL_POLICY    (`ICACHE_REPL_POLICY),
        .NC_ENABLE      (0),
        .CORE_OUT_BUF   (3),
        .MEM_OUT_BUF    (2)
    ) icache (
    `ifdef PERF_ENABLE
        .cache_perf     (icache_perf),
    `endif
        .clk            (clk),
        .reset          (icache_reset),
        .core_bus_if    (per_core_icache_bus_if),
        .mem_bus_if     (icache_mem_bus_if),
        .dfv_enable         (1'b0),
        .dfv_stall_dcache_fill_rsp     (1'b0),
        .dfv_fill_bank_mask (16'hFFFF),
        .dfv_stall_dcache_core_req (1'b0),
        .dfv_throttle_threshold (16'd0)
    );

    ///////////////////////////////////////////////////////////////////////////

    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_TAG_WIDTH)
    ) per_core_dcache_bus_if[`SOCKET_SIZE * DCACHE_NUM_REQS]();

    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_LINE_SIZE),
        .TAG_WIDTH (DCACHE_MEM_TAG_WIDTH)
    ) dcache_mem_bus_if[`L1_MEM_PORTS]();

    `RESET_RELAY (dcache_reset, reset);

    // DFV: aggregate stall signals from all cores (OR reduction)
    wire [`SOCKET_SIZE-1:0] per_core_dfv_stall_dcache_fill_rsp;
    wire [`SOCKET_SIZE-1:0] per_core_dfv_stall_icache_fill_req;
    wire [`SOCKET_SIZE-1:0] per_core_dfv_stall_dcache_fill_req;
    wire [`SOCKET_SIZE-1:0] per_core_dfv_stall_dcache_core_req;
    wire [`SOCKET_SIZE-1:0] per_core_dfv_enable;
    wire [15:0] per_core_dfv_throttle_threshold [`SOCKET_SIZE];
    wire [15:0]  per_core_dfv_fill_bank_mask [`SOCKET_SIZE];
    wire dfv_stall_dcache_fill_rsp_any            = |per_core_dfv_stall_dcache_fill_rsp;
    wire dfv_stall_icache_fill_req_any      = |per_core_dfv_stall_icache_fill_req;
    wire dfv_stall_dcache_fill_req_any      = |per_core_dfv_stall_dcache_fill_req;
    wire dfv_stall_dcache_core_req_any = |per_core_dfv_stall_dcache_core_req;
    wire dfv_enable_any                = |per_core_dfv_enable;
    wire [15:0] dfv_throttle_threshold_val = per_core_dfv_throttle_threshold[0];
    wire [15:0]  dfv_fill_bank_mask_val     = per_core_dfv_fill_bank_mask[0];

    VX_cache_cluster #(
        .INSTANCE_ID    (`SFORMATF(("%s-dcache", INSTANCE_ID))),
        .NUM_UNITS      (`NUM_DCACHES),
        .NUM_INPUTS     (`SOCKET_SIZE),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`DCACHE_SIZE),
        .LINE_SIZE      (DCACHE_LINE_SIZE),
        .NUM_BANKS      (`DCACHE_NUM_BANKS),
        .NUM_WAYS       (`DCACHE_NUM_WAYS),
        .WORD_SIZE      (DCACHE_WORD_SIZE),
        .NUM_REQS       (DCACHE_NUM_REQS),
        .MEM_PORTS      (`L1_MEM_PORTS),
        .CRSQ_SIZE      (`DCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`DCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`DCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`DCACHE_WRITEBACK ? `DCACHE_MSHR_SIZE : `DCACHE_MREQ_SIZE),
        .TAG_WIDTH      (DCACHE_TAG_WIDTH),
        .WRITE_ENABLE   (1),
        .WRITEBACK      (`DCACHE_WRITEBACK),
        .DIRTY_BYTES    (`DCACHE_DIRTYBYTES),
        .REPL_POLICY    (`DCACHE_REPL_POLICY),
        .NC_ENABLE           (1),
        .CORE_OUT_BUF        (3),
        .MEM_OUT_BUF         (2),
        .DFV_THROTTLE_ENABLE (1)
    ) dcache (
    `ifdef PERF_ENABLE
        .cache_perf     (dcache_perf),
    `endif
        .clk            (clk),
        .reset          (dcache_reset),
        .core_bus_if    (per_core_dcache_bus_if),
        .mem_bus_if     (dcache_mem_bus_if),
        .dfv_enable     (dfv_enable_any),
        .dfv_stall_dcache_fill_rsp (dfv_stall_dcache_fill_rsp_any),
        .dfv_fill_bank_mask (dfv_fill_bank_mask_val),
        .dfv_stall_dcache_core_req (dfv_stall_dcache_core_req_any),
        .dfv_throttle_threshold (dfv_throttle_threshold_val)
    );

    ///////////////////////////////////////////////////////////////////////////

    for (genvar i = 0; i < `L1_MEM_PORTS; ++i) begin : g_mem_bus_if
        if (i == 0) begin : g_i0
            VX_mem_bus_if #(
                .DATA_SIZE (`L1_LINE_SIZE),
                .TAG_WIDTH (L1_MEM_TAG_WIDTH)
            ) l1_mem_bus_if[2]();

            VX_mem_bus_if #(
                .DATA_SIZE (`L1_LINE_SIZE),
                .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
            ) l1_mem_arb_bus_if[1]();

            // DFV: gated icache fill request path
            // Intermediate interface before gate
            VX_mem_bus_if #(
                .DATA_SIZE (`L1_LINE_SIZE),
                .TAG_WIDTH (L1_MEM_TAG_WIDTH)
            ) l1_mem_bus_ungated_if[2]();

            `ASSIGN_VX_MEM_BUS_IF_EX (l1_mem_bus_ungated_if[0], icache_mem_bus_if[0], L1_MEM_TAG_WIDTH, ICACHE_MEM_TAG_WIDTH, UUID_WIDTH);
            `ASSIGN_VX_MEM_BUS_IF_EX (l1_mem_bus_ungated_if[1], dcache_mem_bus_if[0], L1_MEM_TAG_WIDTH, DCACHE_MEM_TAG_WIDTH, UUID_WIDTH);

            // Gate icache fill request (req path only, rsp passthrough)
            VX_dfv_req_gate dfv_icache_fillreq_gate (
                .clk          (clk),
                .reset        (reset),
                .dfv_enable   (dfv_enable_any),
                .dfv_stall    (dfv_stall_icache_fill_req_any),
                .master_valid (l1_mem_bus_ungated_if[0].req_valid),
                .master_ready (l1_mem_bus_ungated_if[0].req_ready),
                .slave_valid  (l1_mem_bus_if[0].req_valid),
                .slave_ready  (l1_mem_bus_if[0].req_ready)
            );
            assign l1_mem_bus_if[0].req_data = l1_mem_bus_ungated_if[0].req_data;
            assign l1_mem_bus_ungated_if[0].rsp_valid = l1_mem_bus_if[0].rsp_valid;
            assign l1_mem_bus_ungated_if[0].rsp_data  = l1_mem_bus_if[0].rsp_data;
            assign l1_mem_bus_if[0].rsp_ready = l1_mem_bus_ungated_if[0].rsp_ready;

            // Gate dcache fill request (req path only, rsp passthrough)
            VX_dfv_req_gate dfv_dcache_fillreq_gate (
                .clk          (clk),
                .reset        (reset),
                .dfv_enable   (dfv_enable_any),
                .dfv_stall    (dfv_stall_dcache_fill_req_any),
                .master_valid (l1_mem_bus_ungated_if[1].req_valid),
                .master_ready (l1_mem_bus_ungated_if[1].req_ready),
                .slave_valid  (l1_mem_bus_if[1].req_valid),
                .slave_ready  (l1_mem_bus_if[1].req_ready)
            );
            assign l1_mem_bus_if[1].req_data = l1_mem_bus_ungated_if[1].req_data;
            assign l1_mem_bus_ungated_if[1].rsp_valid = l1_mem_bus_if[1].rsp_valid;
            assign l1_mem_bus_ungated_if[1].rsp_data  = l1_mem_bus_if[1].rsp_data;
            assign l1_mem_bus_if[1].rsp_ready = l1_mem_bus_ungated_if[1].rsp_ready;

            // DFV: count simultaneous icache+dcache fill request collisions
            wire [11:0] dfv_l1_arb_natural_edge_count;
            wire [11:0] dfv_l1_arb_dfv_edge_count;
            `UNUSED_VAR (dfv_l1_arb_natural_edge_count)
            `UNUSED_VAR (dfv_l1_arb_dfv_edge_count)
            VX_dfv_collision_ctr dfv_l1_arb_ctr (
                .clk                (clk),
                .reset              (reset),
                .enable             (dfv_enable_any),
                .event_a            (l1_mem_bus_if[0].req_valid),
                .event_b            (l1_mem_bus_if[1].req_valid),
                .natural_edge_count (dfv_l1_arb_natural_edge_count),
                .dfv_edge_count     (dfv_l1_arb_dfv_edge_count)
            );

            VX_mem_arb #(
                .NUM_INPUTS (2),
                .NUM_OUTPUTS(1),
                .DATA_SIZE  (`L1_LINE_SIZE),
                .TAG_WIDTH  (L1_MEM_TAG_WIDTH),
                .TAG_SEL_IDX(0),
                .ARBITER    ("P"), // prioritize the icache
                .REQ_OUT_BUF(3),
                .RSP_OUT_BUF(3)
            ) mem_arb (
                .clk        (clk),
                .reset      (reset),
                .bus_in_if  (l1_mem_bus_if),
                .bus_out_if (l1_mem_arb_bus_if)
            );

            `ASSIGN_VX_MEM_BUS_IF (mem_bus_if[0], l1_mem_arb_bus_if[0]);
        end else begin : g_i
            VX_mem_bus_if #(
                .DATA_SIZE (`L1_LINE_SIZE),
                .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
            ) l1_mem_arb_bus_if();

            `ASSIGN_VX_MEM_BUS_IF_EX (l1_mem_arb_bus_if, dcache_mem_bus_if[i], L1_MEM_ARB_TAG_WIDTH, DCACHE_MEM_TAG_WIDTH, UUID_WIDTH);
            `ASSIGN_VX_MEM_BUS_IF (mem_bus_if[i], l1_mem_arb_bus_if);
        end
    end

    ///////////////////////////////////////////////////////////////////////////

    wire [`SOCKET_SIZE-1:0] per_core_busy;

    // Generate all cores
    for (genvar core_id = 0; core_id < `SOCKET_SIZE; ++core_id) begin : g_cores

        `RESET_RELAY (core_reset, reset);

        VX_dcr_bus_if core_dcr_bus_if();
        `BUFFER_DCR_BUS_IF (core_dcr_bus_if, dcr_bus_if, 1'b1, (`SOCKET_SIZE > 1))

        VX_core #(
            .CORE_ID  ((SOCKET_ID * `SOCKET_SIZE) + core_id),
            .INSTANCE_ID (`SFORMATF(("%s-core%0d", INSTANCE_ID, core_id)))
        ) core (
            `SCOPE_IO_BIND  (scope_core + core_id)

            .clk            (clk),
            .reset          (core_reset),

        `ifdef PERF_ENABLE
            .sysmem_perf    (sysmem_perf_tmp),
        `endif

            .dcr_bus_if     (core_dcr_bus_if),

            .dcache_bus_if  (per_core_dcache_bus_if[core_id * DCACHE_NUM_REQS +: DCACHE_NUM_REQS]),

            .icache_bus_if  (per_core_icache_bus_if[core_id]),

        `ifdef GBAR_ENABLE
            .gbar_bus_if    (per_core_gbar_bus_if[core_id]),
        `endif

            .busy           (per_core_busy[core_id]),

            .dfv_stall_dcache_fill_rsp_out            (per_core_dfv_stall_dcache_fill_rsp[core_id]),
            .dfv_stall_icache_fill_req_out      (per_core_dfv_stall_icache_fill_req[core_id]),
            .dfv_stall_dcache_fill_req_out      (per_core_dfv_stall_dcache_fill_req[core_id]),
            .dfv_stall_dcache_core_req_out (per_core_dfv_stall_dcache_core_req[core_id]),
            .dfv_enable_out             (per_core_dfv_enable[core_id]),
            .dfv_fill_bank_mask_out     (per_core_dfv_fill_bank_mask[core_id]),
            .dfv_throttle_threshold_out (per_core_dfv_throttle_threshold[core_id])
        );
    end

    `BUFFER_EX(busy, (| per_core_busy), 1'b1, 1, (`SOCKET_SIZE > 1));

endmodule
