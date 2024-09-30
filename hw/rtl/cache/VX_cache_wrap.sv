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

module VX_cache_wrap import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID    = "",

    parameter TAG_SEL_IDX           = 0,

    // Number of Word requests per cycle
    parameter NUM_REQS              = 4,


    // Size of cache in bytes
    parameter CACHE_SIZE            = 4096,
    // Size of line inside a bank in bytes
    parameter LINE_SIZE             = 64,
    // Number of banks
    parameter NUM_BANKS             = 1,
    // Number of associative ways
    parameter NUM_WAYS              = 1,
    // Size of a word in bytes
    parameter WORD_SIZE             = 4,

    // Core Response Queue Size
    parameter CRSQ_SIZE             = 2,
    // Miss Reserv Queue Knob
    parameter MSHR_SIZE             = 8,
    // Memory Response Queue Size
    parameter MRSQ_SIZE             = 0,
    // Memory Request Queue Size
    parameter MREQ_SIZE             = 4,

    // Enable cache writeable
    parameter WRITE_ENABLE          = 1,

    // Enable cache writeback
    parameter WRITEBACK             = 0,

    // Enable dirty bytes on writeback
    parameter DIRTY_BYTES           = 0,

    // Request debug identifier
    parameter UUID_WIDTH            = 0,

    // core request tag size
    parameter TAG_WIDTH             = UUID_WIDTH + 1,

    // enable bypass for non-cacheable addresses
    parameter NC_ENABLE             = 0,

    // Force bypass for all requests
    parameter PASSTHRU              = 0,

    // Core response output buffer
    parameter CORE_OUT_BUF          = 0,

    // Memory request output buffer
    parameter MEM_OUT_BUF           = 0
 ) (

    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    output cache_perf_t     cache_perf,
`endif

    VX_mem_bus_if.slave     core_bus_if [NUM_REQS],
    VX_mem_bus_if.master    mem_bus_if
);

    `STATIC_ASSERT(NUM_BANKS == (1 << `CLOG2(NUM_BANKS)), ("invalid parameter"))

    localparam CACHE_MEM_TAG_WIDTH = `CACHE_MEM_TAG_WIDTH(MSHR_SIZE, NUM_BANKS, UUID_WIDTH);

    localparam MEM_TAG_WIDTH = PASSTHRU ? `CACHE_BYPASS_TAG_WIDTH(NUM_REQS, LINE_SIZE, WORD_SIZE, TAG_WIDTH) :
                                          (NC_ENABLE ? `CACHE_NC_MEM_TAG_WIDTH(MSHR_SIZE, NUM_BANKS, NUM_REQS, LINE_SIZE, WORD_SIZE, TAG_WIDTH, UUID_WIDTH) :
                                                       CACHE_MEM_TAG_WIDTH);

    localparam NC_OR_BYPASS = (NC_ENABLE || PASSTHRU);

    VX_mem_bus_if #(
        .DATA_SIZE (WORD_SIZE),
        .TAG_WIDTH (TAG_WIDTH)
    ) core_bus_cache_if[NUM_REQS]();

    VX_mem_bus_if #(
        .DATA_SIZE (LINE_SIZE),
        .TAG_WIDTH (CACHE_MEM_TAG_WIDTH)
    ) mem_bus_cache_if();

    VX_mem_bus_if #(
        .DATA_SIZE (LINE_SIZE),
        .TAG_WIDTH (MEM_TAG_WIDTH)
    ) mem_bus_tmp_if();

    if (NC_OR_BYPASS) begin : g_bypass

        VX_cache_bypass #(
            .NUM_REQS          (NUM_REQS),
            .TAG_SEL_IDX       (TAG_SEL_IDX),

            .PASSTHRU          (PASSTHRU),
            .NC_ENABLE         (PASSTHRU ? 0 : NC_ENABLE),

            .WORD_SIZE         (WORD_SIZE),
            .LINE_SIZE         (LINE_SIZE),

            .CORE_ADDR_WIDTH   (`CS_WORD_ADDR_WIDTH),
            .CORE_TAG_WIDTH    (TAG_WIDTH),

            .MEM_ADDR_WIDTH    (`CS_MEM_ADDR_WIDTH),
            .MEM_TAG_IN_WIDTH  (CACHE_MEM_TAG_WIDTH),
            .MEM_TAG_OUT_WIDTH (MEM_TAG_WIDTH),

            .UUID_WIDTH        (UUID_WIDTH),

            .CORE_OUT_BUF      (CORE_OUT_BUF),
            .MEM_OUT_BUF       (MEM_OUT_BUF)
        ) cache_bypass (
            .clk            (clk),
            .reset          (reset),

            .core_bus_in_if (core_bus_if),
            .core_bus_out_if(core_bus_cache_if),

            .mem_bus_in_if  (mem_bus_cache_if),
            .mem_bus_out_if (mem_bus_tmp_if)
        );

    end else begin : g_no_bypass

        for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_bus_cache_if
            `ASSIGN_VX_MEM_BUS_IF (core_bus_cache_if[i], core_bus_if[i]);
        end

        `ASSIGN_VX_MEM_BUS_IF (mem_bus_tmp_if, mem_bus_cache_if);
    end

    if (WRITE_ENABLE) begin : g_mem_bus_if
        `ASSIGN_VX_MEM_BUS_IF (mem_bus_if, mem_bus_tmp_if);
    end else begin : g_mem_bus_if_ro
        `ASSIGN_VX_MEM_BUS_RO_IF (mem_bus_if, mem_bus_tmp_if);
    end

    if (PASSTHRU == 0) begin : g_cache

        VX_cache #(
            .INSTANCE_ID  (INSTANCE_ID),
            .CACHE_SIZE   (CACHE_SIZE),
            .LINE_SIZE    (LINE_SIZE),
            .NUM_BANKS    (NUM_BANKS),
            .NUM_WAYS     (NUM_WAYS),
            .WORD_SIZE    (WORD_SIZE),
            .NUM_REQS     (NUM_REQS),
            .CRSQ_SIZE    (CRSQ_SIZE),
            .MSHR_SIZE    (MSHR_SIZE),
            .MRSQ_SIZE    (MRSQ_SIZE),
            .MREQ_SIZE    (MREQ_SIZE),
            .WRITE_ENABLE (WRITE_ENABLE),
            .WRITEBACK    (WRITEBACK),
            .DIRTY_BYTES  (DIRTY_BYTES),
            .UUID_WIDTH   (UUID_WIDTH),
            .TAG_WIDTH    (TAG_WIDTH),
            .CORE_OUT_BUF (NC_OR_BYPASS ? 1 : CORE_OUT_BUF),
            .MEM_OUT_BUF  (NC_OR_BYPASS ? 1 : MEM_OUT_BUF)
        ) cache (
            .clk            (clk),
            .reset          (reset),
        `ifdef PERF_ENABLE
            .cache_perf     (cache_perf),
        `endif
            .core_bus_if    (core_bus_cache_if),
            .mem_bus_if     (mem_bus_cache_if)
        );

    end else begin : g_passthru

        for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_bus_cache_if
            `UNUSED_VAR (core_bus_cache_if[i].req_valid)
            `UNUSED_VAR (core_bus_cache_if[i].req_data)
            assign core_bus_cache_if[i].req_ready = 0;

            assign core_bus_cache_if[i].rsp_valid = 0;
            assign core_bus_cache_if[i].rsp_data  = '0;
            `UNUSED_VAR (core_bus_cache_if[i].rsp_ready)
        end

        assign mem_bus_cache_if.req_valid = 0;
        assign mem_bus_cache_if.req_data = '0;
        `UNUSED_VAR (mem_bus_cache_if.req_ready)

        `UNUSED_VAR (mem_bus_cache_if.rsp_valid)
        `UNUSED_VAR (mem_bus_cache_if.rsp_data)
        assign mem_bus_cache_if.rsp_ready = 0;

    `ifdef PERF_ENABLE
        assign cache_perf = '0;
    `endif

    end

`ifdef DBG_TRACE_CACHE
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_trace
        wire [`UP(UUID_WIDTH)-1:0] core_req_uuid;
        wire [`UP(UUID_WIDTH)-1:0] core_rsp_uuid;

        if (UUID_WIDTH != 0) begin : g_core_rsp_uuid
            assign core_req_uuid = core_bus_if[i].req_data.tag[TAG_WIDTH-1 -: UUID_WIDTH];
            assign core_rsp_uuid = core_bus_if[i].rsp_data.tag[TAG_WIDTH-1 -: UUID_WIDTH];
        end else begin : g_no_core_rsp_uuid
            assign core_req_uuid = 0;
            assign core_rsp_uuid = 0;
        end

        wire core_req_fire = core_bus_if[i].req_valid && core_bus_if[i].req_ready;
        wire core_rsp_fire = core_bus_if[i].rsp_valid && core_bus_if[i].rsp_ready;

        always @(posedge clk) begin
            if (core_req_fire) begin
                if (core_bus_if[i].req_data.rw) begin
                    `TRACE(1, ("%t: %s core-wr-req: addr=0x%0h, tag=0x%0h, req_idx=%0d, byteen=0x%h, data=0x%h (#%0d)\n", $time, INSTANCE_ID, `TO_FULL_ADDR(core_bus_if[i].req_data.addr), core_bus_if[i].req_data.tag, i, core_bus_if[i].req_data.byteen, core_bus_if[i].req_data.data, core_req_uuid))
                end else begin
                    `TRACE(1, ("%t: %s core-rd-req: addr=0x%0h, tag=0x%0h, req_idx=%0d (#%0d)\n", $time, INSTANCE_ID, `TO_FULL_ADDR(core_bus_if[i].req_data.addr), core_bus_if[i].req_data.tag, i, core_req_uuid))
                end
            end
            if (core_rsp_fire) begin
                `TRACE(1, ("%t: %s core-rd-rsp: tag=0x%0h, req_idx=%0d, data=0x%h (#%0d)\n", $time, INSTANCE_ID, core_bus_if[i].rsp_data.tag, i, core_bus_if[i].rsp_data.data, core_rsp_uuid))
            end
        end
    end

    wire [`UP(UUID_WIDTH)-1:0] mem_req_uuid;
    wire [`UP(UUID_WIDTH)-1:0] mem_rsp_uuid;

    if ((UUID_WIDTH != 0) && (NC_OR_BYPASS != 0)) begin : g_mem_req_uuid
        assign mem_req_uuid = mem_bus_if.req_data.tag[MEM_TAG_WIDTH-1 -: UUID_WIDTH];
        assign mem_rsp_uuid = mem_bus_if.rsp_data.tag[MEM_TAG_WIDTH-1 -: UUID_WIDTH];
    end else begin : g_no_mem_req_uuid
        assign mem_req_uuid = 0;
        assign mem_rsp_uuid = 0;
    end

    wire mem_req_fire = mem_bus_if.req_valid && mem_bus_if.req_ready;
    wire mem_rsp_fire = mem_bus_if.rsp_valid && mem_bus_if.rsp_ready;

    always @(posedge clk) begin
        if (mem_req_fire) begin
            if (mem_bus_if.req_data.rw) begin
                `TRACE(1, ("%t: %s mem-wr-req: addr=0x%0h, tag=0x%0h, byteen=0x%h, data=0x%h (#%0d)\n",
                    $time, INSTANCE_ID, `TO_FULL_ADDR(mem_bus_if.req_data.addr), mem_bus_if.req_data.tag, mem_bus_if.req_data.byteen, mem_bus_if.req_data.data, mem_req_uuid))
            end else begin
                `TRACE(1, ("%t: %s mem-rd-req: addr=0x%0h, tag=0x%0h (#%0d)\n",
                    $time, INSTANCE_ID, `TO_FULL_ADDR(mem_bus_if.req_data.addr), mem_bus_if.req_data.tag, mem_req_uuid))
            end
        end
        if (mem_rsp_fire) begin
            `TRACE(1, ("%t: %s mem-rd-rsp: tag=0x%0h, data=0x%h (#%0d)\n",
                $time, INSTANCE_ID, mem_bus_if.rsp_data.tag, mem_bus_if.rsp_data.data, mem_rsp_uuid))
        end
    end
`endif

endmodule
