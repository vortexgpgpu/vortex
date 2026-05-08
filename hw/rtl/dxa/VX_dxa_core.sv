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
    parameter NUM_REQS = 1,
    parameter NUM_DXA_UNITS = 1,
    parameter GMEM_OUT_PORTS = 1
) (
    input wire clk,
    input wire reset,
`ifdef PERF_ENABLE
    output dxa_perf_t dxa_perf,
`endif
    VX_dcr_bus_if.slave dcr_bus_if,
    VX_dxa_req_bus_if.slave req_bus_if[NUM_REQS],
    VX_mem_bus_if.master gmem_bus_if[GMEM_OUT_PORTS],
    VX_mem_bus_if.master smem_bus_if[1],
    output wire busy
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // ================================================================
    // Descriptor table (single read port, driven by dispatch queue)
    // ================================================================
    wire [DXA_DESC_SLOT_W-1:0] desc_read_addr;
    dxa_desc_t desc_read_data;

    VX_dxa_desc_table desc_table (
        .clk        (clk),
        .reset      (reset),
        .dcr_bus_if (dcr_bus_if),
        .read_addr  (desc_read_addr),
        .read_desc  (desc_read_data)
    );

    // ================================================================
    // Request path: sockets → arb(N:1) → queue → dispatch(1:N) → workers
    // ================================================================
    localparam REQ_DATAW = $bits(dxa_req_data_t);

    VX_dxa_req_bus_if arb_out_bus_if[1]();

    VX_dxa_req_arb #(
        .NUM_INPUTS  (NUM_REQS),
        .NUM_OUTPUTS (1)
    ) req_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (req_bus_if),
        .bus_out_if (arb_out_bus_if)
    );

    VX_dxa_req_bus_if queue_out_bus_if[1]();

    VX_elastic_buffer #(
        .DATAW  (REQ_DATAW),
        .SIZE   (`DXA_QUEUE_SIZE),
        .LUTRAM (1)
    ) req_queue (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (arb_out_bus_if[0].req_valid),
        .ready_in  (arb_out_bus_if[0].req_ready),
        .data_in   (arb_out_bus_if[0].req_data),
        .valid_out (queue_out_bus_if[0].req_valid),
        .ready_out (queue_out_bus_if[0].req_ready),
        .data_out  (queue_out_bus_if[0].req_data)
    );

    // Read desc_table using the queued request's desc_slot
    assign desc_read_addr = DXA_DESC_SLOT_W'(queue_out_bus_if[0].req_data.meta[DXA_DESC_SLOT_W-1:0]);

    // Bundle request + descriptor into dispatch input
    VX_dxa_worker_req_if dispatch_in_if[1]();

    assign dispatch_in_if[0].valid     = queue_out_bus_if[0].req_valid;
    assign dispatch_in_if[0].req_data  = queue_out_bus_if[0].req_data;
    assign dispatch_in_if[0].desc_data = desc_read_data;
    assign queue_out_bus_if[0].req_ready = dispatch_in_if[0].ready;

    VX_dxa_worker_req_if worker_req_if[NUM_DXA_UNITS]();

    VX_dxa_dispatch #(
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (NUM_DXA_UNITS)
    ) issue_dispatch (
        .clk       (clk),
        .reset     (reset),
        .req_in    (dispatch_in_if),
        .req_out   (worker_req_if)
    );

    // ================================================================
    // Workers
    // ================================================================
    localparam GMEM_ARB_SEL_BITS = `ARB_SEL_BITS(NUM_DXA_UNITS, GMEM_OUT_PORTS);
    localparam WORKER_GMEM_TAG_WIDTH = L1_MEM_ARB_TAG_WIDTH - GMEM_ARB_SEL_BITS;

    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (WORKER_GMEM_TAG_WIDTH)
    ) worker_gmem_bus_if[NUM_DXA_UNITS]();

    VX_mem_bus_if #(
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (DXA_LMEM_TAG_W),
        .ATTR_WIDTH  (DXA_LMEM_ATTR_W),
        .ADDR_WIDTH  (DXA_LMEM_ADDR_W)
    ) worker_smem_bus_if[NUM_DXA_UNITS]();

`ifdef PERF_ENABLE
    dxa_perf_t worker_dxa_perf [NUM_DXA_UNITS];
`endif

    for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_workers
        VX_dxa_worker #(
            .INSTANCE_ID(`SFORMATF(("%s-worker%0d", INSTANCE_ID, i))),
            .WORKER_ID  (i),
            .GMEM_TAG_WIDTH (WORKER_GMEM_TAG_WIDTH)
        ) worker (
            .clk            (clk),
            .reset          (reset),
        `ifdef PERF_ENABLE
            .dxa_perf       (worker_dxa_perf[i]),
        `endif
            .req_if         (worker_req_if[i]),
            .gmem_bus_if    (worker_gmem_bus_if[i]),
            .smem_bus_if    (worker_smem_bus_if[i])
        );
    end

    // ================================================================
    // Output arbitration
    // ================================================================
    VX_mem_arb #(
        .NUM_INPUTS  (NUM_DXA_UNITS),
        .NUM_OUTPUTS (GMEM_OUT_PORTS),
        .DATA_SIZE   (`L1_LINE_SIZE),
        .TAG_WIDTH   (WORKER_GMEM_TAG_WIDTH),
        .ARBITER     ("R")
    ) gmem_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (worker_gmem_bus_if),
        .bus_out_if (gmem_bus_if)
    );

    VX_mem_arb #(
        .NUM_INPUTS  (NUM_DXA_UNITS),
        .NUM_OUTPUTS (1),
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (DXA_LMEM_TAG_W),
        .TAG_SEL_IDX (DXA_LMEM_TAG_W - UUID_WIDTH),
        .ATTR_WIDTH  (DXA_LMEM_ATTR_W),
        .ADDR_WIDTH  (DXA_LMEM_ADDR_W),
        .ARBITER     ("R")
    ) lmem_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (worker_smem_bus_if),
        .bus_out_if (smem_bus_if)
    );

    // ================================================================
    // Busy / perf
    // ================================================================
    wire [NUM_REQS-1:0] req_bus_valid;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_valid
        assign req_bus_valid[i] = req_bus_if[i].req_valid;
    end

    wire [NUM_DXA_UNITS-1:0] worker_idle;
    for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_worker_idle
        assign worker_idle[i] = worker_req_if[i].ready;
    end

    reg dxa_busy_r;
    always @(posedge clk) begin
        if (reset)
            dxa_busy_r <= 1'b0;
        else
            dxa_busy_r <= dcr_bus_if.req_valid | (|req_bus_valid) | ~(&worker_idle);
    end
    assign busy = dxa_busy_r | dcr_bus_if.req_valid | (|req_bus_valid);

`ifdef PERF_ENABLE
    always_comb begin
        dxa_perf = '0;
        for (int w = 0; w < NUM_DXA_UNITS; ++w) begin
            dxa_perf.transfers   += worker_dxa_perf[w].transfers;
            dxa_perf.gmem_reads  += worker_dxa_perf[w].gmem_reads;
            dxa_perf.gmem_dedup  += worker_dxa_perf[w].gmem_dedup;
            dxa_perf.lmem_writes += worker_dxa_perf[w].lmem_writes;
            dxa_perf.gmem_latency += worker_dxa_perf[w].gmem_latency;
        end
    end
`endif

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset) begin
            for (integer w = 0; w < NUM_DXA_UNITS; ++w) begin
                if (worker_req_if[w].valid) begin
                    `TRACE(1, ("%t: %s dispatch-issue: worker=%0d, core=%0d, wid=%0d, meta=0x%0h\n",
                        $time, INSTANCE_ID, w,
                        worker_req_if[w].req_data.core_id,
                        worker_req_if[w].req_data.wid,
                        worker_req_if[w].req_data.meta))
                end
            end
        end
    end
`endif

endmodule
