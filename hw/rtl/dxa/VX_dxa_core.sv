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

module VX_dxa_core import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter DXA_NUM_SOCKETS = 1,
    parameter NUM_DXA_UNITS = 1,
    parameter GMEM_OUT_PORTS = 1,
    parameter CORE_LOCAL_BITS = 0
) (
    input wire clk,
    input wire reset,
`ifdef PERF_ENABLE
    output dxa_perf_t dxa_perf,
`endif
    VX_dcr_bus_if.slave dcr_bus_if,

    VX_dxa_req_bus_if.slave req_bus_if[DXA_NUM_SOCKETS],
    VX_mem_bus_if.master lmem_bus_if[DXA_NUM_SOCKETS * `SOCKET_SIZE],
    VX_mem_bus_if.master gmem_bus_if[GMEM_OUT_PORTS],
    output wire busy
);
    localparam NUM_LMEM_OUTPUTS = DXA_NUM_SOCKETS * `SOCKET_SIZE;
    localparam ROUTER_SEL_W     = `UP(`CLOG2(NUM_LMEM_OUTPUTS));
    `UNUSED_PARAM (CORE_LOCAL_BITS)

    VX_dxa_req_bus_if cluster_dxa_bus_if[NUM_DXA_UNITS]();

    VX_mem_bus_if #(
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (LMEM_DMA_TAG_W),
        .FLAGS_WIDTH (DXA_LMEM_FLAGS_WIDTH),
        .ADDR_WIDTH  (DXA_LMEM_BANK_ADDR_WIDTH)
    ) worker_lmem_bus_if[NUM_DXA_UNITS]();

    wire [NUM_DXA_UNITS-1:0][NC_WIDTH-1:0] worker_lmem_core_id;

    // Request distribution: DXA_NUM_SOCKETS → NUM_DXA_UNITS
    wire [DXA_NUM_SOCKETS-1:0]                       req_valid_in;
    wire [DXA_NUM_SOCKETS-1:0][DXA_REQ_DATAW-1:0]    req_data_in;
    wire [DXA_NUM_SOCKETS-1:0]                       req_ready_in;

    for (genvar i = 0; i < DXA_NUM_SOCKETS; ++i) begin : g_req_in
        assign req_valid_in[i] = req_bus_if[i].req_valid;
        assign req_data_in[i]  = req_bus_if[i].req_data;
        assign req_bus_if[i].req_ready = req_ready_in[i];
    end

    wire [NUM_DXA_UNITS-1:0]                       req_valid_out;
    wire [NUM_DXA_UNITS-1:0][DXA_REQ_DATAW-1:0]    req_data_out;
    wire [NUM_DXA_UNITS-1:0]                       req_ready_out;

    VX_stream_arb #(
        .NUM_INPUTS  (DXA_NUM_SOCKETS),
        .NUM_OUTPUTS (NUM_DXA_UNITS),
        .DATAW       (DXA_REQ_DATAW),
        .ARBITER     ("R"),
        .OUT_BUF     ((DXA_NUM_SOCKETS != NUM_DXA_UNITS) ? 2 : 0)
    ) req_arb (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (req_valid_in),
        .data_in    (req_data_in),
        .ready_in   (req_ready_in),
        .valid_out  (req_valid_out),
        .data_out   (req_data_out),
        .ready_out  (req_ready_out),
        `UNUSED_PIN (sel_out)
    );

    for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_req_out
        assign cluster_dxa_bus_if[i].req_valid = req_valid_out[i];
        assign cluster_dxa_bus_if[i].req_data  = req_data_out[i];
        assign req_ready_out[i] = cluster_dxa_bus_if[i].req_ready;
    end

    // Internal worker gmem buses (pre-distribution)
    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
    ) worker_gmem_bus_if[NUM_DXA_UNITS]();

    wire engine_busy;
    VX_dxa_unified_engine #(
        .INSTANCE_ID  (`SFORMATF(("%s-unified", INSTANCE_ID))),
        .NUM_DXA_UNITS(NUM_DXA_UNITS)
    ) dxa_unified_engine (
        .clk                (clk),
        .reset              (reset),
    `ifdef PERF_ENABLE
        .dxa_perf           (dxa_perf),
    `endif
        .dcr_bus_if         (dcr_bus_if),
        .cluster_dxa_bus_if (cluster_dxa_bus_if),
        .dxa_gmem_bus_if    (worker_gmem_bus_if),
        .dxa_lmem_bus_if    (worker_lmem_bus_if),
        .dxa_lmem_core_id   (worker_lmem_core_id),
        .busy               (engine_busy)
    );

    // Collect incoming request-valid from all sockets.
    wire [DXA_NUM_SOCKETS-1:0] req_bus_valid;
    for (genvar i = 0; i < DXA_NUM_SOCKETS; ++i) begin : g_req_valid
        assign req_bus_valid[i] = req_bus_if[i].req_valid;
    end

    // Registered hold: set on DCR or request activity, clear when engine drains.
    // Combinatorial assertion OR-ed in for immediate ICG wake-up.
    reg dxa_busy_r;
    always @(posedge clk) begin
        if (reset)
            dxa_busy_r <= 1'b0;
        else
            dxa_busy_r <= dcr_bus_if.req_valid | (|req_bus_valid) | engine_busy;
    end
    assign busy = dxa_busy_r | dcr_bus_if.req_valid | (|req_bus_valid);

    // Distribute NUM_DXA_UNITS worker gmem buses → GMEM_OUT_PORTS L2-facing buses
    VX_mem_arb #(
        .NUM_INPUTS  (NUM_DXA_UNITS),
        .NUM_OUTPUTS (GMEM_OUT_PORTS),
        .DATA_SIZE   (`L1_LINE_SIZE),
        .TAG_WIDTH   (L1_MEM_ARB_TAG_WIDTH),
        .ARBITER     ("R")
    ) dxa_gmem_l2_dist (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (worker_gmem_bus_if),
        .bus_out_if (gmem_bus_if)
    );

    // Compute routing sel for each worker based on target core_id.
    wire [NUM_DXA_UNITS-1:0][ROUTER_SEL_W-1:0] worker_output_sel;

    for (genvar w = 0; w < NUM_DXA_UNITS; ++w) begin : g_worker_sel
        assign worker_output_sel[w] = ROUTER_SEL_W'(worker_lmem_core_id[w]);
    end

    // Route worker LMEM writes to per-core output ports.
    VX_mem_xbar #(
        .NUM_INPUTS  (NUM_DXA_UNITS),
        .NUM_OUTPUTS (NUM_LMEM_OUTPUTS),
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (LMEM_DMA_TAG_W),
        .FLAGS_WIDTH (DXA_LMEM_FLAGS_WIDTH),
        .ADDR_WIDTH  (DXA_LMEM_BANK_ADDR_WIDTH),
        .ARBITER     ("R"),
        .REQ_OUT_BUF ((NUM_DXA_UNITS != NUM_LMEM_OUTPUTS) ? 2 : 0)
    ) dxa_lmem_xbar (
        .clk        (clk),
        .reset      (reset),
        .sel_in     (worker_output_sel),
        .bus_in_if  (worker_lmem_bus_if),
        .bus_out_if (lmem_bus_if)
    );

endmodule
