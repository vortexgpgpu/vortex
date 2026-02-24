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

`ifdef EXT_DXA_ENABLE
    // DXA control path
    VX_dxa_req_bus_if.master    dxa_req_bus_if,
    // DXA shared-memory data path
    VX_mem_bus_if.slave         dxa_smem_bus_if,
`endif

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
        .mem_bus_if     (icache_mem_bus_if)
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
        .NC_ENABLE      (1),
        .CORE_OUT_BUF   (3),
        .MEM_OUT_BUF    (2)
    ) dcache (
    `ifdef PERF_ENABLE
        .cache_perf     (dcache_perf),
    `endif
        .clk            (clk),
        .reset          (dcache_reset),
        .core_bus_if    (per_core_dcache_bus_if),
        .mem_bus_if     (dcache_mem_bus_if)
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

            `ASSIGN_VX_MEM_BUS_IF_EX (l1_mem_bus_if[0], icache_mem_bus_if[0], L1_MEM_TAG_WIDTH, ICACHE_MEM_TAG_WIDTH, UUID_WIDTH);
            `ASSIGN_VX_MEM_BUS_IF_EX (l1_mem_bus_if[1], dcache_mem_bus_if[0], L1_MEM_TAG_WIDTH, DCACHE_MEM_TAG_WIDTH, UUID_WIDTH);

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

`ifdef EXT_DXA_ENABLE

    VX_dxa_req_bus_if per_core_dxa_req_bus_if[`SOCKET_SIZE]();
    VX_mem_bus_if #(
        .DATA_SIZE (DXA_SMEM_WORD_SIZE),
        .TAG_WIDTH (LMEM_TAG_WIDTH)
    ) per_core_dxa_smem_bus_if[`SOCKET_SIZE]();

    localparam DXA_REQ_DATAW = NC_WIDTH + UUID_WIDTH + NW_WIDTH + 3 + (2 * `XLEN);
    localparam DXA_RSP_DATAW = NC_WIDTH + UUID_WIDTH + NW_WIDTH + BAR_ADDR_W + 2;
    localparam DXA_CORE_SEL_BITS = `CLOG2(`SOCKET_SIZE);
    localparam DXA_CORE_SEL_W = `UP(DXA_CORE_SEL_BITS);

    wire [`SOCKET_SIZE-1:0] dxa_req_valid_in;
    wire [`SOCKET_SIZE-1:0][DXA_REQ_DATAW-1:0] dxa_req_data_in;
    wire [`SOCKET_SIZE-1:0] dxa_req_ready_in;

    wire [`SOCKET_SIZE-1:0] dxa_rsp_ready_out;

    for (genvar i = 0; i < `SOCKET_SIZE; ++i) begin : g_dxa_req_in
        assign dxa_req_valid_in[i] = per_core_dxa_req_bus_if[i].req_valid;
        assign dxa_req_data_in[i] = per_core_dxa_req_bus_if[i].req_data;
        assign per_core_dxa_req_bus_if[i].req_ready = dxa_req_ready_in[i];
        assign dxa_rsp_ready_out[i] = per_core_dxa_req_bus_if[i].rsp_ready;
    end

    wire [0:0] dxa_req_valid_out;
    wire [0:0][DXA_REQ_DATAW-1:0] dxa_req_data_out;
    wire [0:0] dxa_req_ready_out;
    wire [0:0][`UP(`CLOG2(`SOCKET_SIZE))-1:0] dxa_req_sel_out;

    VX_stream_arb #(
        .NUM_INPUTS  (`SOCKET_SIZE),
        .NUM_OUTPUTS (1),
        .DATAW       (DXA_REQ_DATAW),
        .ARBITER     ("R"),
        .OUT_BUF     (2)
    ) dxa_req_arb (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (dxa_req_valid_in),
        .data_in    (dxa_req_data_in),
        .ready_in   (dxa_req_ready_in),
        .valid_out  (dxa_req_valid_out),
        .data_out   (dxa_req_data_out),
        .ready_out  (dxa_req_ready_out),
        .sel_out    (dxa_req_sel_out)
    );

    assign dxa_req_bus_if.req_valid = dxa_req_valid_out[0];
    assign dxa_req_bus_if.req_data  = dxa_req_data_out[0];
    assign dxa_req_ready_out[0] = dxa_req_bus_if.req_ready;
    `UNUSED_VAR (dxa_req_sel_out)

    wire [DXA_CORE_SEL_W-1:0] dxa_rsp_core_sel = DXA_CORE_SEL_W'(dxa_req_bus_if.rsp_data.core_id);
    wire [0:0] dxa_rsp_valid_in;
    wire [0:0][DXA_RSP_DATAW-1:0] dxa_rsp_data_in;
    wire [0:0][DXA_CORE_SEL_W-1:0] dxa_rsp_sel_in;
    wire [0:0] dxa_rsp_ready_in;

    wire [`SOCKET_SIZE-1:0] dxa_rsp_valid_out;
    wire [`SOCKET_SIZE-1:0][DXA_RSP_DATAW-1:0] dxa_rsp_data_out;

    assign dxa_rsp_valid_in[0] = dxa_req_bus_if.rsp_valid;
    assign dxa_rsp_data_in[0] = dxa_req_bus_if.rsp_data;
    assign dxa_rsp_sel_in[0] = dxa_rsp_core_sel;

    VX_stream_switch #(
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (`SOCKET_SIZE),
        .DATAW       (DXA_RSP_DATAW),
        .OUT_BUF     (2)
    ) dxa_rsp_switch (
        .clk       (clk),
        .reset     (reset),
        .sel_in    (dxa_rsp_sel_in),
        .valid_in  (dxa_rsp_valid_in),
        .data_in   (dxa_rsp_data_in),
        .ready_in  (dxa_rsp_ready_in),
        .valid_out (dxa_rsp_valid_out),
        .data_out  (dxa_rsp_data_out),
        .ready_out (dxa_rsp_ready_out)
    );

    assign dxa_req_bus_if.rsp_ready = dxa_rsp_ready_in[0];

    for (genvar i = 0; i < `SOCKET_SIZE; ++i) begin : g_dxa_rsp_out
        assign per_core_dxa_req_bus_if[i].rsp_valid = dxa_rsp_valid_out[i];
        assign per_core_dxa_req_bus_if[i].rsp_data  = dxa_rsp_data_out[i];
    end

    localparam DXA_SMEM_REQ_DATAW = 1
                                  + DXA_SMEM_ADDR_WIDTH
                                  + (DXA_SMEM_WORD_SIZE * 8)
                                  + DXA_SMEM_WORD_SIZE
                                  + MEM_FLAGS_WIDTH
                                  + LMEM_TAG_WIDTH;
    localparam DXA_SMEM_RSP_DATAW = (DXA_SMEM_WORD_SIZE * 8) + LMEM_TAG_WIDTH;
    localparam DXA_SMEM_TAG_WIDTH = LMEM_TAG_WIDTH;
    localparam DXA_SMEM_ENGINE_BITS = `CLOG2(`NUM_DXA_UNITS);
    localparam DXA_SMEM_ENGINE_W = `UP(DXA_SMEM_ENGINE_BITS);
    localparam DXA_CORE_LOCAL_BITS = `CLOG2(`SOCKET_SIZE);
    localparam DXA_CORE_LOCAL_W = `UP(DXA_CORE_LOCAL_BITS);

    wire [DXA_SMEM_TAG_WIDTH-1:0] dxa_smem_req_tag_route = {
        dxa_smem_bus_if.req_data.tag.uuid,
        dxa_smem_bus_if.req_data.tag.value
    };
    wire [DXA_SMEM_ENGINE_W-1:0] dxa_smem_req_engine_id =
        dxa_smem_req_tag_route[DXA_SMEM_ENGINE_W-1:0];
    wire [NC_WIDTH-1:0] dxa_smem_req_core_id =
        NC_WIDTH'(dxa_smem_req_tag_route[DXA_SMEM_ENGINE_W +: NC_WIDTH]);

    wire [DXA_CORE_LOCAL_W-1:0] dxa_smem_req_core_sel =
        DXA_CORE_LOCAL_W'(dxa_smem_req_core_id);

    wire [0:0] dxa_smem_req_valid_in;
    wire [0:0][DXA_SMEM_REQ_DATAW-1:0] dxa_smem_req_data_in;
    wire [0:0][DXA_CORE_LOCAL_W-1:0] dxa_smem_req_sel_in;
    wire [0:0] dxa_smem_req_ready_in;
    wire [`SOCKET_SIZE-1:0] dxa_smem_req_valid_out;
    wire [`SOCKET_SIZE-1:0][DXA_SMEM_REQ_DATAW-1:0] dxa_smem_req_data_out;
    wire [`SOCKET_SIZE-1:0] dxa_smem_req_ready_out;

    assign dxa_smem_req_valid_in[0] = dxa_smem_bus_if.req_valid;
    assign dxa_smem_req_data_in[0] = dxa_smem_bus_if.req_data;
    assign dxa_smem_req_sel_in[0] = dxa_smem_req_core_sel;

    VX_stream_switch #(
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (`SOCKET_SIZE),
        .DATAW       (DXA_SMEM_REQ_DATAW),
        .OUT_BUF     (2)
    ) dxa_smem_req_switch (
        .clk       (clk),
        .reset     (reset),
        .sel_in    (dxa_smem_req_sel_in),
        .valid_in  (dxa_smem_req_valid_in),
        .data_in   (dxa_smem_req_data_in),
        .ready_in  (dxa_smem_req_ready_in),
        .valid_out (dxa_smem_req_valid_out),
        .data_out  (dxa_smem_req_data_out),
        .ready_out (dxa_smem_req_ready_out)
    );

    assign dxa_smem_bus_if.req_ready = dxa_smem_req_ready_in[0];

    for (genvar i = 0; i < `SOCKET_SIZE; ++i) begin : g_dxa_smem_req_out
        assign per_core_dxa_smem_bus_if[i].req_valid = dxa_smem_req_valid_out[i];
        assign per_core_dxa_smem_bus_if[i].req_data = dxa_smem_req_data_out[i];
        assign dxa_smem_req_ready_out[i] = per_core_dxa_smem_bus_if[i].req_ready;
    end

    wire [`SOCKET_SIZE-1:0] dxa_smem_rsp_valid_in;
    wire [`SOCKET_SIZE-1:0][DXA_SMEM_RSP_DATAW-1:0] dxa_smem_rsp_data_in;
    wire [`SOCKET_SIZE-1:0] dxa_smem_rsp_ready_in;

    for (genvar i = 0; i < `SOCKET_SIZE; ++i) begin : g_dxa_smem_rsp_in
        assign dxa_smem_rsp_valid_in[i] = per_core_dxa_smem_bus_if[i].rsp_valid;
        assign dxa_smem_rsp_data_in[i] = per_core_dxa_smem_bus_if[i].rsp_data;
        assign per_core_dxa_smem_bus_if[i].rsp_ready = dxa_smem_rsp_ready_in[i];
    end

    wire [0:0] dxa_smem_rsp_valid_out;
    wire [0:0][DXA_SMEM_RSP_DATAW-1:0] dxa_smem_rsp_data_out;
    wire [0:0] dxa_smem_rsp_ready_out;

    VX_stream_arb #(
        .NUM_INPUTS  (`SOCKET_SIZE),
        .NUM_OUTPUTS (1),
        .DATAW       (DXA_SMEM_RSP_DATAW),
        .ARBITER     ("R"),
        .OUT_BUF     (2)
    ) dxa_smem_rsp_arb (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (dxa_smem_rsp_valid_in),
        .data_in    (dxa_smem_rsp_data_in),
        .ready_in   (dxa_smem_rsp_ready_in),
        .valid_out  (dxa_smem_rsp_valid_out),
        .data_out   (dxa_smem_rsp_data_out),
        .ready_out  (dxa_smem_rsp_ready_out),
        `UNUSED_PIN (sel_out)
    );

    assign dxa_smem_bus_if.rsp_valid = dxa_smem_rsp_valid_out[0];
    assign dxa_smem_bus_if.rsp_data  = dxa_smem_rsp_data_out[0];
    assign dxa_smem_rsp_ready_out[0] = dxa_smem_bus_if.rsp_ready;

    `UNUSED_VAR (dxa_smem_req_engine_id)

`endif

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

        `ifdef EXT_DXA_ENABLE
            .dxa_req_bus_if     (per_core_dxa_req_bus_if[core_id]),
            .dxa_smem_bus_if    (per_core_dxa_smem_bus_if[core_id]),
        `endif

        `ifdef GBAR_ENABLE
            .gbar_bus_if    (per_core_gbar_bus_if[core_id]),
        `endif

            .busy           (per_core_busy[core_id])
        );
    end

    `BUFFER_EX(busy, (| per_core_busy), 1'b1, 1, (`SOCKET_SIZE > 1));

endmodule
