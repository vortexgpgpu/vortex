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

`include "VX_cache_define.vh"

module VX_cache_cluster import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID    = "",

    parameter NUM_UNITS             = 1,
    parameter NUM_INPUTS            = 1,
    parameter TAG_SEL_IDX           = 0,

    // Number of requests per cycle
    parameter NUM_REQS              = 4,

    // Number of memory ports
    parameter MEM_PORTS             = 1,

    // Size of cache in bytes
    parameter CACHE_SIZE            = 32768,
    // Size of line inside a bank in bytes
    parameter LINE_SIZE             = 64,
    // Number of banks
    parameter NUM_BANKS             = 4,
    // Number of associative ways
    parameter NUM_WAYS              = 4,
    // Size of a word in bytes
    parameter WORD_SIZE             = 16,

    // Core Response Queue Size
    parameter CRSQ_SIZE             = 4,
    // Miss Reserv Queue Knob
    parameter MSHR_SIZE             = 16,
    // Memory Response Queue Size
    parameter MRSQ_SIZE             = 4,
    // Memory Request Queue Size
    parameter MREQ_SIZE             = 4,

    // Enable cache writeable
    parameter WRITE_ENABLE          = 1,

    // Enable cache writeback
    parameter WRITEBACK             = 0,

    // Enable dirty bytes on writeback
    parameter DIRTY_BYTES           = 0,

    // Replacement policy
    parameter REPL_POLICY           = `CS_REPL_FIFO,

    // Request debug identifier
    parameter UUID_WIDTH            = 0,

    // core request tag size
    parameter TAG_WIDTH             = UUID_WIDTH + 1,

    // core request flags
    parameter FLAGS_WIDTH           = 0,

    // enable bypass for non-cacheable addresses
    parameter NC_ENABLE             = 0,

    // Core response output buffer
    parameter CORE_OUT_BUF          = 3,

    // Memory request output buffer
    parameter MEM_OUT_BUF           = 3
 ) (
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    output cache_perf_t     cache_perf,
`endif

    VX_mem_bus_if.slave     core_bus_if [NUM_INPUTS * NUM_REQS],
    VX_mem_bus_if.master    mem_bus_if [MEM_PORTS]
);
    localparam NUM_CACHES = `UP(NUM_UNITS);
    localparam PASSTHRU   = (NUM_UNITS == 0);
    localparam ARB_TAG_WIDTH = TAG_WIDTH + `ARB_SEL_BITS(NUM_INPUTS, NUM_CACHES);

    localparam CACHE_MEM_TAG_WIDTH = `CACHE_MEM_TAG_WIDTH(MSHR_SIZE, NUM_BANKS, MEM_PORTS, UUID_WIDTH);
    localparam BYPASS_TAG_WIDTH = `CACHE_BYPASS_TAG_WIDTH(NUM_REQS, MEM_PORTS, LINE_SIZE, WORD_SIZE, ARB_TAG_WIDTH);
    localparam NC_TAG_WIDTH = `MAX(CACHE_MEM_TAG_WIDTH, BYPASS_TAG_WIDTH) + 1;
    localparam MEM_TAG_WIDTH = PASSTHRU ? BYPASS_TAG_WIDTH : (NC_ENABLE ? NC_TAG_WIDTH : CACHE_MEM_TAG_WIDTH);

    `STATIC_ASSERT(NUM_INPUTS >= NUM_CACHES, ("invalid parameter"))

`ifdef PERF_ENABLE
    cache_perf_t perf_cache_unit[NUM_CACHES];
    `PERF_CACHE_ADD (cache_perf, perf_cache_unit, NUM_CACHES)
`endif

    VX_mem_bus_if #(
        .DATA_SIZE (LINE_SIZE),
        .TAG_WIDTH (MEM_TAG_WIDTH)
    ) cache_mem_bus_if[NUM_CACHES * MEM_PORTS]();

    VX_mem_bus_if #(
        .DATA_SIZE (WORD_SIZE),
        .TAG_WIDTH (ARB_TAG_WIDTH)
    ) arb_core_bus_if[NUM_CACHES * NUM_REQS]();

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_arb
        VX_mem_bus_if #(
            .DATA_SIZE (WORD_SIZE),
            .TAG_WIDTH (TAG_WIDTH)
        ) core_bus_tmp_if[NUM_INPUTS]();

        VX_mem_bus_if #(
            .DATA_SIZE (WORD_SIZE),
            .TAG_WIDTH (ARB_TAG_WIDTH)
        ) arb_core_bus_tmp_if[NUM_CACHES]();

        for (genvar j = 0; j < NUM_INPUTS; ++j) begin : g_core_bus_tmp_if
            `ASSIGN_VX_MEM_BUS_IF (core_bus_tmp_if[j], core_bus_if[j * NUM_REQS + i]);
        end

        VX_mem_arb #(
            .NUM_INPUTS   (NUM_INPUTS),
            .NUM_OUTPUTS  (NUM_CACHES),
            .DATA_SIZE    (WORD_SIZE),
            .TAG_WIDTH    (TAG_WIDTH),
            .TAG_SEL_IDX  (TAG_SEL_IDX),
            .ARBITER      ("R"),
            .REQ_OUT_BUF  ((NUM_INPUTS != NUM_CACHES) ? 2 : 0),
            .RSP_OUT_BUF  ((NUM_INPUTS != NUM_CACHES) ? CORE_OUT_BUF : 0)
        ) core_arb (
            .clk        (clk),
            .reset      (reset),
            .bus_in_if  (core_bus_tmp_if),
            .bus_out_if (arb_core_bus_tmp_if)
        );

        for (genvar k = 0; k < NUM_CACHES; ++k) begin : g_arb_core_bus_if
            `ASSIGN_VX_MEM_BUS_IF (arb_core_bus_if[k * NUM_REQS + i], arb_core_bus_tmp_if[k]);
        end
    end

     for (genvar i = 0; i < NUM_CACHES; ++i) begin : g_cache_wrap
        VX_cache_wrap #(
            .INSTANCE_ID  (`SFORMATF(("%s%0d", INSTANCE_ID, i))),
            .CACHE_SIZE   (CACHE_SIZE),
            .LINE_SIZE    (LINE_SIZE),
            .NUM_BANKS    (NUM_BANKS),
            .NUM_WAYS     (NUM_WAYS),
            .WORD_SIZE    (WORD_SIZE),
            .NUM_REQS     (NUM_REQS),
            .MEM_PORTS    (MEM_PORTS),
            .WRITE_ENABLE (WRITE_ENABLE),
            .WRITEBACK    (WRITEBACK),
            .DIRTY_BYTES  (DIRTY_BYTES),
            .REPL_POLICY  (REPL_POLICY),
            .CRSQ_SIZE    (CRSQ_SIZE),
            .MSHR_SIZE    (MSHR_SIZE),
            .MRSQ_SIZE    (MRSQ_SIZE),
            .MREQ_SIZE    (MREQ_SIZE),
            .UUID_WIDTH   (UUID_WIDTH),
            .TAG_WIDTH    (ARB_TAG_WIDTH),
            .FLAGS_WIDTH  (FLAGS_WIDTH),
            .TAG_SEL_IDX  (TAG_SEL_IDX),
            .CORE_OUT_BUF ((NUM_INPUTS != NUM_CACHES) ? 2 : CORE_OUT_BUF),
            .MEM_OUT_BUF  ((NUM_CACHES > 1) ? 2 : MEM_OUT_BUF),
            .NC_ENABLE    (NC_ENABLE),
            .PASSTHRU     (PASSTHRU)
        ) cache_wrap (
        `ifdef PERF_ENABLE
            .cache_perf  (perf_cache_unit[i]),
        `endif
            .clk         (clk),
            .reset       (reset),
            .core_bus_if (arb_core_bus_if[i * NUM_REQS +: NUM_REQS]),
            .mem_bus_if  (cache_mem_bus_if[i * MEM_PORTS +: MEM_PORTS])
        );
    end

    for (genvar i = 0; i < MEM_PORTS; ++i) begin : g_mem_bus_if
        VX_mem_bus_if #(
            .DATA_SIZE (LINE_SIZE),
            .TAG_WIDTH (MEM_TAG_WIDTH)
        ) arb_core_bus_tmp_if[NUM_CACHES]();

        VX_mem_bus_if #(
            .DATA_SIZE (LINE_SIZE),
            .TAG_WIDTH (MEM_TAG_WIDTH + `ARB_SEL_BITS(NUM_CACHES, 1))
        ) mem_bus_tmp_if[1]();

        for (genvar j = 0; j < NUM_CACHES; ++j) begin : g_arb_core_bus_tmp_if
            `ASSIGN_VX_MEM_BUS_IF (arb_core_bus_tmp_if[j], cache_mem_bus_if[j * MEM_PORTS + i]);
        end

        VX_mem_arb #(
            .NUM_INPUTS  (NUM_CACHES),
            .NUM_OUTPUTS (1),
            .DATA_SIZE   (LINE_SIZE),
            .TAG_WIDTH   (MEM_TAG_WIDTH),
            .TAG_SEL_IDX (TAG_SEL_IDX),
            .ARBITER     ("R"),
            .REQ_OUT_BUF ((NUM_CACHES > 1) ? MEM_OUT_BUF : 0),
            .RSP_OUT_BUF ((NUM_CACHES > 1) ? 2 : 0)
        ) mem_arb (
            .clk        (clk),
            .reset      (reset),
            .bus_in_if  (arb_core_bus_tmp_if),
            .bus_out_if (mem_bus_tmp_if)
        );

        if (WRITE_ENABLE) begin : g_we
            `ASSIGN_VX_MEM_BUS_IF (mem_bus_if[i], mem_bus_tmp_if[0]);
        end else begin : g_ro
            `ASSIGN_VX_MEM_BUS_RO_IF (mem_bus_if[i], mem_bus_tmp_if[0]);
        end
    end

endmodule
