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

// The cluster translation complex as VX_cluster and Vortex wire it at one
// cluster: the sockets' miss buses arbitrated onto VX_tlb_l2, whose miss
// station feeds the device walker VX_ptw. Buses flattened for the C++ bench.
module VX_tlb_l2_top import VX_gpu_pkg::*, VX_tlb_pkg::*; #(
    parameter NS   = 4,
    parameter IDW  = TLB_SOCKET_ID_WIDTH,
    parameter DS   = `VX_CFG_L1_LINE_SIZE,
    parameter MTW  = DCACHE_TAG_WIDTH_BASE + `ARB_SEL_BITS(`VX_CFG_PTW_NUM_WALKERS, 1),
    parameter MAW  = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(DS)
) (
    input  wire clk,
    input  wire reset,
    input  wire [`VX_CFG_XLEN-1:0] satp,
    input  wire flush_req,
    output wire flush_done,
    output wire empty,
    output wire [7:0] id_width,

    input  wire [NS-1:0]                      req_valid,
    input  wire [NS-1:0][IDW-1:0]             req_id,
    input  wire [NS-1:0][TLB_VPN_WIDTH-1:0]   req_vpn,
    input  wire [NS-1:0][1:0]                 req_access,
    input  wire [NS-1:0]                      req_amo,
    output wire [NS-1:0]                      req_ready,

    output wire [NS-1:0]                      rsp_valid,
    output wire [NS-1:0][IDW-1:0]             rsp_id,
    output wire [NS-1:0]                      rsp_fault,
    output wire [NS-1:0][TLB_LEVEL_WIDTH-1:0] rsp_level,
    output wire [NS-1:0][TLB_PPN_WIDTH-1:0]   rsp_ppn,
    output wire [NS-1:0][TLB_FLAGS_WIDTH-1:0] rsp_flags,
    input  wire [NS-1:0]                      rsp_ready,

    output wire             mem_req_valid,
    output wire             mem_req_rw,
    output wire [MAW-1:0]   mem_req_addr,
    output wire [MTW-1:0]   mem_req_tag,
    input  wire             mem_req_ready,
    input  wire             mem_rsp_valid,
    input  wire [DS*8-1:0]  mem_rsp_data,
    input  wire [MTW-1:0]   mem_rsp_tag,
    output wire             mem_rsp_ready,

    output wire             fault_valid,
    output wire [`VX_CFG_XLEN-1:0] fault_va
);
    `STATIC_ASSERT ((IDW == TLB_SOCKET_ID_WIDTH), ("socket id width"))
    assign id_width = 8'(IDW);

    VX_tlb_bus_if #(.ID_WIDTH (TLB_SOCKET_ID_WIDTH))   socket_if [NS] ();
    VX_tlb_bus_if #(.ID_WIDTH (TLB_SOCKET_ID_WIDTH + `ARB_SEL_BITS(NS, 1))) l2_client_if ();
    VX_tlb_bus_if #(.ID_WIDTH (L2_TLB_SLOT_WIDTH))     l2_ptw_if ();

    for (genvar i = 0; i < NS; ++i) begin : g_sockets
        assign socket_if[i].req_valid       = req_valid[i];
        assign socket_if[i].req_data.id     = req_id[i];
        assign socket_if[i].req_data.access = tlb_access_e'(req_access[i]);
        assign socket_if[i].req_data.amo    = req_amo[i];
        assign socket_if[i].req_data.vpn    = req_vpn[i];
        assign req_ready[i]                 = socket_if[i].req_ready;
        assign rsp_valid[i] = socket_if[i].rsp_valid;
        assign rsp_id[i]    = socket_if[i].rsp_data.id;
        assign rsp_fault[i] = socket_if[i].rsp_data.fault;
        assign rsp_level[i] = socket_if[i].rsp_data.level;
        assign rsp_ppn[i]   = socket_if[i].rsp_data.ppn;
        assign rsp_flags[i] = socket_if[i].rsp_data.flags;
        assign socket_if[i].rsp_ready = rsp_ready[i];
    end

    VX_tlb_bus_arb #(
        .NUM_INPUTS  (NS),
        .ID_WIDTH_IN (TLB_SOCKET_ID_WIDTH),
        .OUT_BUF     (3)
    ) tlb_cluster_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (socket_if),
        .bus_out_if (l2_client_if)
    );

    VX_tlb_flush_if l2_flush_if ();
    VX_tlb_flush_if ptw_flush_if ();
    assign l2_flush_if.req  = flush_req;
    assign ptw_flush_if.req = flush_req;
    assign flush_done = l2_flush_if.done && ptw_flush_if.done;

    wire l2_empty, ptw_empty;

    VX_tlb_l2 #(
        .ID_WIDTH (TLB_SOCKET_ID_WIDTH + `ARB_SEL_BITS(NS, 1))
    ) l2tlb (
        .clk       (clk),
        .reset     (reset),
        .client_if (l2_client_if),
        .ptw_if    (l2_ptw_if),
        .flush_if  (l2_flush_if),
        .empty     (l2_empty)
    );

    VX_mem_bus_if #(
        .DATA_SIZE (DS),
        .TAG_WIDTH (MTW)
    ) ptw_mem_if ();
    VX_mmu_fault_if ptw_fault_if ();

    VX_ptw #(
        .ID_WIDTH      (L2_TLB_SLOT_WIDTH),
        .DATA_SIZE     (DS),
        .MEM_TAG_WIDTH (MTW)
    ) ptw (
        .clk        (clk),
        .reset      (reset),
        .satp       (satp),
        .miss_if    (l2_ptw_if),
        .mem_bus_if (ptw_mem_if),
        .flush_if   (ptw_flush_if),
        .fault_if   (ptw_fault_if),
        .empty      (ptw_empty)
    );

    assign empty = l2_empty && ptw_empty;

    assign mem_req_valid = ptw_mem_if.req_valid;
    assign mem_req_rw    = ptw_mem_if.req_data.rw;
    assign mem_req_addr  = ptw_mem_if.req_data.addr;
    assign mem_req_tag   = ptw_mem_if.req_data.tag;
    assign ptw_mem_if.req_ready = mem_req_ready;
    `UNUSED_VAR (ptw_mem_if.req_data.data)
    `UNUSED_VAR (ptw_mem_if.req_data.byteen)
    `UNUSED_VAR (ptw_mem_if.req_data.attr)

    assign ptw_mem_if.rsp_valid     = mem_rsp_valid;
    assign ptw_mem_if.rsp_data.data = mem_rsp_data;
    assign ptw_mem_if.rsp_data.tag  = mem_rsp_tag;
    assign mem_rsp_ready = ptw_mem_if.rsp_ready;

    assign fault_valid = ptw_fault_if.valid;
    assign fault_va    = ptw_fault_if.va;
    `UNUSED_VAR (ptw_fault_if.access)
    `UNUSED_VAR (ptw_fault_if.amo)

endmodule
