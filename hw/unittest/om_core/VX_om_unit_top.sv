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

`include "VX_om_define.vh"

// The OM as a cluster attaches it: the export-aperture ingress in front of the
// OM core, which hands the ingress its decoded state.
module VX_om_unit_top import VX_gpu_pkg::*; import VX_om_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = OM_CORE_LANES,
    parameter DATA_SIZE = `VX_CFG_L1_LINE_SIZE,
    parameter TAG_WIDTH = UUID_WIDTH + 8,
    parameter ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(DATA_SIZE)
) (
    input wire                              clk,
    input wire                              reset,

    input wire                              dcr_write_valid,
    input wire [VX_DCR_ADDR_WIDTH-1:0]     dcr_write_addr,
    input wire [VX_DCR_DATA_WIDTH-1:0]     dcr_write_data,

    // Export-aperture writes.
    input  wire                             ap_req_valid,
    input  wire                             ap_req_rw,
    input  wire [ADDR_WIDTH-1:0]            ap_req_addr,
    input  wire [DATA_SIZE*8-1:0]           ap_req_data,
    input  wire [DATA_SIZE-1:0]             ap_req_byteen,
    input  wire [TAG_WIDTH-1:0]             ap_req_tag,
    output wire                             ap_req_ready,

    output wire [OCACHE_NUM_REQS-1:0]       cache_req_valid,
    output wire [OCACHE_NUM_REQS-1:0]       cache_req_rw,
    output wire [OCACHE_NUM_REQS-1:0][OCACHE_WORD_SIZE-1:0] cache_req_byteen,
    output wire [OCACHE_NUM_REQS-1:0][OCACHE_ADDR_WIDTH-1:0] cache_req_addr,
    output wire [OCACHE_NUM_REQS-1:0][OCACHE_WORD_SIZE*8-1:0] cache_req_data,
    output wire [OCACHE_NUM_REQS-1:0][OCACHE_TAG_WIDTH-1:0] cache_req_tag,
    input  wire [OCACHE_NUM_REQS-1:0]       cache_req_ready,

    input wire  [OCACHE_NUM_REQS-1:0]       cache_rsp_valid,
    input wire  [OCACHE_NUM_REQS-1:0][OCACHE_WORD_SIZE*8-1:0] cache_rsp_data,
    input wire  [OCACHE_NUM_REQS-1:0][OCACHE_TAG_WIDTH-1:0] cache_rsp_tag,
    output wire [OCACHE_NUM_REQS-1:0]       cache_rsp_ready
);

    VX_om_perf_if perf_om_if();
`ifndef PERF_ENABLE
    assign perf_om_if.mem_reads    = '0;
    assign perf_om_if.mem_writes   = '0;
    assign perf_om_if.mem_latency  = '0;
    assign perf_om_if.stall_cycles = '0;
    `UNUSED_VAR (perf_om_if.mem_reads)
    `UNUSED_VAR (perf_om_if.mem_writes)
    `UNUSED_VAR (perf_om_if.mem_latency)
    `UNUSED_VAR (perf_om_if.stall_cycles)
`endif

    VX_dcr_bus_if dcr_bus_if();

    assign dcr_bus_if.req_valid     = dcr_write_valid;
    assign dcr_bus_if.req_data.rw   = 1'b1;
    assign dcr_bus_if.req_data.addr = dcr_write_addr;
    assign dcr_bus_if.req_data.data = dcr_write_data;
    `UNUSED_VAR (dcr_bus_if.rsp_valid)
    `UNUSED_VAR (dcr_bus_if.rsp_data)

    VX_mem_bus_if #(
        .DATA_SIZE (DATA_SIZE),
        .TAG_WIDTH (TAG_WIDTH)
    ) ap_bus_if();

    assign ap_bus_if.req_valid        = ap_req_valid;
    assign ap_bus_if.req_data.rw      = ap_req_rw;
    assign ap_bus_if.req_data.addr    = ap_req_addr;
    assign ap_bus_if.req_data.data    = ap_req_data;
    assign ap_bus_if.req_data.byteen  = ap_req_byteen;
    assign ap_bus_if.req_data.attr    = '0;
    assign ap_bus_if.req_data.tag     = ap_req_tag;
    assign ap_req_ready               = ap_bus_if.req_ready;
    assign ap_bus_if.rsp_ready        = 1'b1;
    `UNUSED_VAR (ap_bus_if.rsp_valid)
    `UNUSED_VAR (ap_bus_if.rsp_data)

    om_dcrs_t om_dcrs;

    VX_om_bus_if #(
        .NUM_LANES (NUM_LANES)
    ) om_bus_if();

    VX_om_ingress #(
        .INSTANCE_ID ($sformatf("%s-ingress", INSTANCE_ID)),
        .DATA_SIZE   (DATA_SIZE),
        .TAG_WIDTH   (TAG_WIDTH),
        .NUM_LANES   (NUM_LANES),
        .OUT_BUF     (3)
    ) om_ingress (
        .clk        (clk),
        .reset      (reset),
        .dcrs       (om_dcrs),
        .mem_bus_if (ap_bus_if),
        .om_bus_if  (om_bus_if),
        `UNUSED_PIN (busy)
    );

    VX_mem_bus_if #(
        .DATA_SIZE (OCACHE_WORD_SIZE),
        .TAG_WIDTH (OCACHE_TAG_WIDTH)
    ) cache_bus_if[OCACHE_NUM_REQS]();

    for (genvar i = 0; i < OCACHE_NUM_REQS; ++i) begin : g_cache
        assign cache_req_valid[i]  = cache_bus_if[i].req_valid;
        assign cache_req_rw[i]     = cache_bus_if[i].req_data.rw;
        assign cache_req_byteen[i] = cache_bus_if[i].req_data.byteen;
        assign cache_req_addr[i]   = cache_bus_if[i].req_data.addr;
        assign cache_req_data[i]   = cache_bus_if[i].req_data.data;
        assign cache_req_tag[i]    = cache_bus_if[i].req_data.tag;
        assign cache_bus_if[i].req_ready = cache_req_ready[i];
        `UNUSED_VAR (cache_bus_if[i].req_data.attr)

        assign cache_bus_if[i].rsp_valid     = cache_rsp_valid[i];
        assign cache_bus_if[i].rsp_data.tag  = cache_rsp_tag[i];
        assign cache_bus_if[i].rsp_data.data = cache_rsp_data[i];
        assign cache_rsp_ready[i] = cache_bus_if[i].rsp_ready;
    end

    VX_om_core #(
        .INSTANCE_ID (INSTANCE_ID),
        .NUM_LANES   (NUM_LANES)
    ) om_core (
        .clk           (clk),
        .reset         (reset),
    `ifdef PERF_ENABLE
        .perf_om_if   (perf_om_if),
    `endif
        .dcr_bus_if    (dcr_bus_if),
        .om_bus_if     (om_bus_if),
        .cache_bus_if  (cache_bus_if),
        `UNUSED_PIN (busy),
        .om_dcrs       (om_dcrs),
        `UNUSED_PIN (occ_count)
    );

endmodule
