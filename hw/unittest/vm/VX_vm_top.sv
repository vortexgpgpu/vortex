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

// Standalone MMU-network synthesis sandbox.
//
// Reproduces one cluster's address-translation subsystem exactly as
// VX_socket/VX_cluster wire it, with the cores/FPU/graphics removed:
//
//   per socket (NUM_SOCKETS, from config):
//     VX_mmu_snoop  -> vm_active + L1 TLB flush (from the DCR / satp.MODE)
//     VX_mmu dmmu   : translation lane between the request stream and L1 dcache
//     VX_mmu immu   : instruction-side translation into L1 icache (INCLUDE_IMMU)
//     VX_cache_cluster dcache / icache : the real L1 caches the MMU drives
//     VX_tlb_bus_arb : folds the dTLB/iTLB miss ports into the cluster fabric
//   per cluster:
//     VX_tlb_bus_arb : folds all sockets' miss ports -> the L2 TLB client bus
//     VX_tlb_l2      : shared L2 TLB
//     VX_ptw         : page-table walker (fed satp; PTE reads through the L2)
//     VX_cache_wrap  : the shared L2, fed by the sockets' L1 mem ports + the PTW
//
// Memory hierarchy: the L1 dcaches and the PTW are clients of the shared L2
// (the real L2_NUM_REQS client set, minus graphics), so the L2 input-arb and
// its PTE-fetch boundary carry realistic timing. The icaches' memory side and
// the L2's memory side exit to top-level ports.
//
// Fidelity: every MMU/PTW timing boundary is present and loaded by its real
// neighbour (MMU->real L1 cache, PTW->real L2). What is removed (cores/FPU/
// graphics) never drove an MMU boundary net. All widths follow the production
// VX_gpu_pkg localparams so structure matches the real RTL.
//
// No component may be optimized out: every DUT input is driven by a top-level
// port (no constant folding of satp / requests / flush) and every DUT output is
// observed at a top-level port (no dangling-net removal).

module VX_vm_top import VX_gpu_pkg::*, VX_tlb_pkg::*; #(
    parameter `STRING INSTANCE_ID = "vm_top",
    parameter INCLUDE_IMMU = 1,
    // Derived port geometry (do not override).
    parameter DMMU_REQS     = NUM_SOCKETS * `VX_CFG_SOCKET_SIZE * DCACHE_NUM_REQS,
    parameter IMMU_REQS     = NUM_SOCKETS * `VX_CFG_SOCKET_SIZE,
    parameter IC_MEM_PORTS  = NUM_SOCKETS,
    parameter DC_ADDR_W     = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(DCACHE_WORD_SIZE),
    parameter IC_ADDR_W     = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(ICACHE_WORD_SIZE),
    parameter IC_MEM_ADDR_W = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(ICACHE_LINE_SIZE),
    parameter L2_MEM_ADDR_W = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(L2_SECTOR_SIZE)
) (
    input  wire clk,
    input  wire reset,

    // -----------------------------------------------------------------------
    // Translation config
    // -----------------------------------------------------------------------
    input  wire [`VX_CFG_XLEN-1:0]      satp,

    // DCR write stream feeding every socket's MMU-snoop (satp.MODE => vm_active
    // + TLB flush). Broadcast to all sockets, as the cluster DCR arb does.
    input  wire                         dcr_req_valid,
    input  dcr_req_t                    dcr_req_data,
    output wire                         dcr_rsp_valid,   // observed: keeps snoop live

    // Cluster TLB + walker flush handshake
    input  wire                         mmu_flush_req,
    output wire                         mmu_flush_done,

    // -----------------------------------------------------------------------
    // dTLB translation request stream (slave), all sockets flattened
    // -----------------------------------------------------------------------
    input  wire [DMMU_REQS-1:0]                          dmmu_req_valid,
    input  wire [DMMU_REQS-1:0]                          dmmu_req_rw,
    input  wire [DMMU_REQS-1:0][DCACHE_WORD_SIZE-1:0]    dmmu_req_byteen,
    input  wire [DMMU_REQS-1:0][DC_ADDR_W-1:0]           dmmu_req_addr,
    input  wire [DMMU_REQS-1:0][MEM_ATTR_WIDTH-1:0]      dmmu_req_attr,
    input  wire [DMMU_REQS-1:0][DCACHE_WORD_SIZE*8-1:0]  dmmu_req_data,
    input  wire [DMMU_REQS-1:0][DCACHE_TAG_WIDTH-1:0]    dmmu_req_tag,
    output wire [DMMU_REQS-1:0]                          dmmu_req_ready,

    output wire [DMMU_REQS-1:0]                          dmmu_rsp_valid,
    output wire [DMMU_REQS-1:0][DCACHE_WORD_SIZE*8-1:0]  dmmu_rsp_data,
    output wire [DMMU_REQS-1:0][DCACHE_TAG_WIDTH-1:0]    dmmu_rsp_tag,
    input  wire [DMMU_REQS-1:0]                          dmmu_rsp_ready,

    // -----------------------------------------------------------------------
    // iTLB translation request stream (slave), all sockets flattened
    // -----------------------------------------------------------------------
    input  wire [IMMU_REQS-1:0]                          immu_req_valid,
    input  wire [IMMU_REQS-1:0][IC_ADDR_W-1:0]           immu_req_addr,
    input  wire [IMMU_REQS-1:0][ICACHE_TAG_WIDTH-1:0]    immu_req_tag,
    output wire [IMMU_REQS-1:0]                          immu_req_ready,

    output wire [IMMU_REQS-1:0]                          immu_rsp_valid,
    output wire [IMMU_REQS-1:0][ICACHE_WORD_SIZE*8-1:0]  immu_rsp_data,
    output wire [IMMU_REQS-1:0][ICACHE_TAG_WIDTH-1:0]    immu_rsp_tag,
    input  wire [IMMU_REQS-1:0]                          immu_rsp_ready,

    // -----------------------------------------------------------------------
    // L1 icache memory side (master), all sockets flattened
    // -----------------------------------------------------------------------
    output wire [IC_MEM_PORTS-1:0]                       ic_mem_req_valid,
    output wire [IC_MEM_PORTS-1:0][IC_MEM_ADDR_W-1:0]    ic_mem_req_addr,
    output wire [IC_MEM_PORTS-1:0][ICACHE_MEM_TAG_WIDTH-1:0] ic_mem_req_tag,
    input  wire [IC_MEM_PORTS-1:0]                       ic_mem_req_ready,

    input  wire [IC_MEM_PORTS-1:0]                       ic_mem_rsp_valid,
    input  wire [IC_MEM_PORTS-1:0][ICACHE_LINE_SIZE*8-1:0] ic_mem_rsp_data,
    input  wire [IC_MEM_PORTS-1:0][ICACHE_MEM_TAG_WIDTH-1:0] ic_mem_rsp_tag,
    output wire [IC_MEM_PORTS-1:0]                       ic_mem_rsp_ready,

    // -----------------------------------------------------------------------
    // L2 memory side (master): L1 dcache + PTE traffic reach memory here
    // -----------------------------------------------------------------------
    output wire [L2_MEM_PORTS-1:0]                       l2_mem_req_valid,
    output wire [L2_MEM_PORTS-1:0]                       l2_mem_req_rw,
    output wire [L2_MEM_PORTS-1:0][L2_SECTOR_SIZE-1:0]   l2_mem_req_byteen,
    output wire [L2_MEM_PORTS-1:0][L2_MEM_ADDR_W-1:0]    l2_mem_req_addr,
    output wire [L2_MEM_PORTS-1:0][L2_MEM_TAG_WIDTH-1:0] l2_mem_req_tag,
    output wire [L2_MEM_PORTS-1:0][L2_SECTOR_SIZE*8-1:0] l2_mem_req_data,
    input  wire [L2_MEM_PORTS-1:0]                       l2_mem_req_ready,

    input  wire [L2_MEM_PORTS-1:0]                       l2_mem_rsp_valid,
    input  wire [L2_MEM_PORTS-1:0][L2_SECTOR_SIZE*8-1:0] l2_mem_rsp_data,
    input  wire [L2_MEM_PORTS-1:0][L2_MEM_TAG_WIDTH-1:0] l2_mem_rsp_tag,
    output wire [L2_MEM_PORTS-1:0]                       l2_mem_rsp_ready,

    // -----------------------------------------------------------------------
    // Page-fault sideband (from the walker) + drain
    // -----------------------------------------------------------------------
    output wire                         fault_valid,
    output wire [`VX_CFG_XLEN-1:0]      fault_va,
    output wire [1:0]                   fault_access,
    output wire                         fault_amo,

    output wire                         empty
);
    localparam NS       = NUM_SOCKETS;
    localparam SS       = `VX_CFG_SOCKET_SIZE;
    localparam D_LANES  = SS * DCACHE_NUM_REQS;  // dmmu lanes per socket
    localparam I_LANES  = SS;                    // immu lanes per socket

    // ------------------------------------------------------------------------
    // Config / snoop plumbing (one DCR bus per socket; write stream broadcast)
    // ------------------------------------------------------------------------
    VX_dcr_bus_if dcr_in_if [NS] ();
    wire [NS-1:0] dcr_out_valid;
    wire [NS-1:0] dcr_in_rsp_valid;

    VX_tlb_flush_if l2_flush_if ();
    VX_tlb_flush_if ptw_flush_if ();
    assign l2_flush_if.req  = mmu_flush_req;
    assign ptw_flush_if.req = mmu_flush_req;
    assign mmu_flush_done   = l2_flush_if.done && ptw_flush_if.done;

    // ------------------------------------------------------------------------
    // Per-socket miss ports into the cluster fabric + L1 dcache memory ports
    // ------------------------------------------------------------------------
    VX_tlb_bus_if #(.ID_WIDTH (TLB_SOCKET_ID_WIDTH))
        per_socket_tlb_bus_if [NS] ();

    // Every socket's L1 dcache memory ports, collected for the shared L2.
    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_SECTOR_SIZE),
        .TAG_WIDTH (DCACHE_MEM_TAG_WIDTH)
    ) all_dc_mem_if [NS * L1_MEM_PORTS] ();

    wire [NS-1:0] dmmu_empty, immu_empty;

    // ========================================================================
    // Per-socket MMU + L1 caches
    // ========================================================================
    for (genvar s = 0; s < NS; ++s) begin : g_sockets

        // --- request-stream buses (virtual, from top ports) ---
        VX_mem_bus_if #(
            .DATA_SIZE (DCACHE_WORD_SIZE),
            .TAG_WIDTH (DCACHE_TAG_WIDTH)
        ) dmmu_core_if [D_LANES] ();

        VX_mem_bus_if #(
            .DATA_SIZE (DCACHE_WORD_SIZE),
            .TAG_WIDTH (DCACHE_TAG_WIDTH)
        ) dmmu_phys_if [D_LANES] ();   // translated, to L1 dcache

        for (genvar i = 0; i < D_LANES; ++i) begin : g_dmmu_bind
            localparam g = s * D_LANES + i;
            assign dmmu_core_if[i].req_valid       = dmmu_req_valid[g];
            assign dmmu_core_if[i].req_data.rw     = dmmu_req_rw[g];
            assign dmmu_core_if[i].req_data.byteen = dmmu_req_byteen[g];
            assign dmmu_core_if[i].req_data.addr   = dmmu_req_addr[g];
            assign dmmu_core_if[i].req_data.attr   = dmmu_req_attr[g];
            assign dmmu_core_if[i].req_data.data   = dmmu_req_data[g];
            assign dmmu_core_if[i].req_data.tag    = dmmu_req_tag[g];
            assign dmmu_req_ready[g]               = dmmu_core_if[i].req_ready;
            assign dmmu_rsp_valid[g]               = dmmu_core_if[i].rsp_valid;
            assign dmmu_rsp_data[g]                = dmmu_core_if[i].rsp_data.data;
            assign dmmu_rsp_tag[g]                 = dmmu_core_if[i].rsp_data.tag;
            assign dmmu_core_if[i].rsp_ready       = dmmu_rsp_ready[g];
        end

        // --- snoop: DCR -> vm_active + L1 TLB flush ---
        wire vm_active;
        VX_dcr_bus_if   socket_dcr_out_if ();
        VX_tlb_flush_if dmmu_flush_if ();
        VX_tlb_flush_if immu_flush_if ();
        VX_mmu_fault_if dmmu_fault_if ();

        assign dcr_in_if[s].req_valid = dcr_req_valid;
        assign dcr_in_if[s].req_data  = dcr_req_data;
        assign dcr_in_rsp_valid[s]    = dcr_in_if[s].rsp_valid;
        `UNUSED_VAR (dcr_in_if[s].rsp_data)

        VX_mmu_snoop mmu_snoop (
            .clk            (clk),
            .reset          (reset),
            .dcr_bus_in_if  (dcr_in_if[s]),
            .dcr_bus_out_if (socket_dcr_out_if),
            .vm_active      (vm_active),
            .dmmu_flush_if  (dmmu_flush_if),
            .immu_flush_if  (immu_flush_if)
        );
        assign dcr_out_valid[s] = socket_dcr_out_if.req_valid;
        assign socket_dcr_out_if.rsp_valid = 1'b0;
        assign socket_dcr_out_if.rsp_data  = '0;
        `UNUSED_VAR (socket_dcr_out_if.req_data)

        // --- socket miss ports (dTLB, iTLB) into the socket arbiter ---
        VX_tlb_bus_if #(.ID_WIDTH (L1_TLB_ID_WIDTH))
            socket_tlb_bus_if [2] ();

        VX_mmu #(
            .INSTANCE_ID ("dmmu"),
            .NUM_REQS    (D_LANES),
            .TLB_SIZE    (`VX_CFG_DTLB_SIZE),
            .EXEC_SIDE   (0),
            .DATA_SIZE   (DCACHE_WORD_SIZE),
            .TAG_WIDTH   (DCACHE_TAG_WIDTH_BASE)
        ) dmmu (
            .clk         (clk),
            .reset       (reset),
            .vm_active   (vm_active),
            .core_bus_if (dmmu_core_if),
            .mem_bus_if  (dmmu_phys_if),
            .tlb_bus_if  (socket_tlb_bus_if[0]),
            .flush_if    (dmmu_flush_if),
            .fault_if    (dmmu_fault_if),
            .empty       (dmmu_empty[s])
        );

        // L1 MMU faults are sunk here, matching VX_socket (the walker reports the
        // authoritative fault to the top-level fault ports).
        `UNUSED_VAR (dmmu_fault_if.valid)
        `UNUSED_VAR (dmmu_fault_if.va)
        `UNUSED_VAR (dmmu_fault_if.access)
        `UNUSED_VAR (dmmu_fault_if.amo)

        VX_cache_cluster #(
            .INSTANCE_ID    (`SFORMATF(("%s-dcache%0d", INSTANCE_ID, s))),
            .NUM_UNITS      (`VX_CFG_NUM_DCACHES),
            .NUM_INPUTS     (SS),
            .TAG_SEL_IDX    (0),
            .CACHE_SIZE     (`VX_CFG_DCACHE_SIZE),
            .LINE_SIZE      (DCACHE_LINE_SIZE),
            .SECTOR_SIZE    (DCACHE_SECTOR_SIZE),
            .NUM_BANKS      (DCACHE_NUM_BANKS),
            .NUM_WAYS       (`VX_CFG_DCACHE_NUM_WAYS),
            .WORD_SIZE      (DCACHE_WORD_SIZE),
            .NUM_REQS       (DCACHE_NUM_REQS),
            .MEM_PORTS      (L1_MEM_PORTS),
            .CRSQ_SIZE      (`VX_CFG_DCACHE_CRSQ_SIZE),
            .MSHR_SIZE      (`VX_CFG_DCACHE_MSHR_SIZE),
            .MRSQ_SIZE      (`VX_CFG_DCACHE_MRSQ_SIZE),
            .MREQ_SIZE      (`VX_CFG_DCACHE_MREQ_SIZE),
            .LATENCY        (`VX_CFG_DCACHE_LATENCY),
            .TAG_WIDTH      (DCACHE_TAG_WIDTH),
            .WRITE_ENABLE   (1),
            .WRITEBACK      (`VX_CFG_DCACHE_WRITEBACK),
            .DIRTY_BYTES    (`VX_CFG_DCACHE_DIRTYBYTES),
            .REPL_POLICY    (`VX_CFG_DCACHE_REPL_POLICY),
            .NC_ENABLE      (1),
            .CORE_OUT_BUF   (3),
            .MEM_OUT_BUF    (3),
            .IS_LLC         (DCACHE_IS_LLC),
            .AMO_ENABLE     (`VX_CFG_EXT_A_ENABLED)
        ) dcache (
            .clk            (clk),
            .reset          (reset),
            .core_bus_if    (dmmu_phys_if),
            .mem_bus_if     (all_dc_mem_if[s * L1_MEM_PORTS +: L1_MEM_PORTS])
        );

        if (INCLUDE_IMMU != 0) begin : g_immu
            VX_mem_bus_if #(
                .DATA_SIZE (ICACHE_WORD_SIZE),
                .TAG_WIDTH (ICACHE_TAG_WIDTH)
            ) immu_core_if [I_LANES] ();

            VX_mem_bus_if #(
                .DATA_SIZE (ICACHE_WORD_SIZE),
                .TAG_WIDTH (ICACHE_TAG_WIDTH)
            ) immu_phys_if [I_LANES] ();

            for (genvar i = 0; i < I_LANES; ++i) begin : g_immu_bind
                localparam g = s * I_LANES + i;
                assign immu_core_if[i].req_valid       = immu_req_valid[g];
                assign immu_core_if[i].req_data.rw     = 1'b0;
                assign immu_core_if[i].req_data.byteen = '0;
                assign immu_core_if[i].req_data.addr   = immu_req_addr[g];
                assign immu_core_if[i].req_data.attr   = '0;
                assign immu_core_if[i].req_data.data   = '0;
                assign immu_core_if[i].req_data.tag    = immu_req_tag[g];
                assign immu_req_ready[g]               = immu_core_if[i].req_ready;
                assign immu_rsp_valid[g]               = immu_core_if[i].rsp_valid;
                assign immu_rsp_data[g]                = immu_core_if[i].rsp_data.data;
                assign immu_rsp_tag[g]                 = immu_core_if[i].rsp_data.tag;
                assign immu_core_if[i].rsp_ready       = immu_rsp_ready[g];
            end

            VX_mmu_fault_if immu_fault_if ();

            VX_mem_bus_if #(
                .DATA_SIZE (ICACHE_LINE_SIZE),
                .TAG_WIDTH (ICACHE_MEM_TAG_WIDTH)
            ) ic_mem_if [1] ();

            assign ic_mem_req_valid[s]        = ic_mem_if[0].req_valid;
            assign ic_mem_req_addr[s]         = ic_mem_if[0].req_data.addr;
            assign ic_mem_req_tag[s]          = ic_mem_if[0].req_data.tag;
            assign ic_mem_if[0].req_ready     = ic_mem_req_ready[s];
            assign ic_mem_if[0].rsp_valid     = ic_mem_rsp_valid[s];
            assign ic_mem_if[0].rsp_data.data = ic_mem_rsp_data[s];
            assign ic_mem_if[0].rsp_data.tag  = ic_mem_rsp_tag[s];
            assign ic_mem_rsp_ready[s]        = ic_mem_if[0].rsp_ready;
            `UNUSED_VAR (ic_mem_if[0].req_data.rw)
            `UNUSED_VAR (ic_mem_if[0].req_data.byteen)
            `UNUSED_VAR (ic_mem_if[0].req_data.data)
            `UNUSED_VAR (ic_mem_if[0].req_data.attr)

            VX_mmu #(
                .INSTANCE_ID ("immu"),
                .NUM_REQS    (I_LANES),
                .TLB_SIZE    (`VX_CFG_ITLB_SIZE),
                .EXEC_SIDE   (1),
                .DATA_SIZE   (ICACHE_WORD_SIZE),
                .TAG_WIDTH   (ICACHE_TAG_WIDTH)
            ) immu (
                .clk         (clk),
                .reset       (reset),
                .vm_active   (vm_active),
                .core_bus_if (immu_core_if),
                .mem_bus_if  (immu_phys_if),
                .tlb_bus_if  (socket_tlb_bus_if[1]),
                .flush_if    (immu_flush_if),
                .fault_if    (immu_fault_if),
                .empty       (immu_empty[s])
            );

            `UNUSED_VAR (immu_fault_if.valid)
            `UNUSED_VAR (immu_fault_if.va)
            `UNUSED_VAR (immu_fault_if.access)
            `UNUSED_VAR (immu_fault_if.amo)

            VX_cache_cluster #(
                .INSTANCE_ID    (`SFORMATF(("%s-icache%0d", INSTANCE_ID, s))),
                .NUM_UNITS      (`VX_CFG_NUM_ICACHES),
                .NUM_INPUTS     (SS),
                .TAG_SEL_IDX    (0),
                .CACHE_SIZE     (`VX_CFG_ICACHE_SIZE),
                .LINE_SIZE      (ICACHE_LINE_SIZE),
                .NUM_BANKS      (1),
                .NUM_WAYS       (`VX_CFG_ICACHE_NUM_WAYS),
                .WORD_SIZE      (ICACHE_WORD_SIZE),
                .NUM_REQS       (1),
                .MEM_PORTS      (1),
                .CRSQ_SIZE      (`VX_CFG_ICACHE_CRSQ_SIZE),
                .MSHR_SIZE      (`VX_CFG_ICACHE_MSHR_SIZE),
                .MRSQ_SIZE      (`VX_CFG_ICACHE_MRSQ_SIZE),
                .MREQ_SIZE      (`VX_CFG_ICACHE_MREQ_SIZE),
                .LATENCY        (`VX_CFG_ICACHE_LATENCY),
                .TAG_WIDTH      (ICACHE_TAG_WIDTH),
                .WRITE_ENABLE   (0),
                .REPL_POLICY    (`VX_CFG_ICACHE_REPL_POLICY),
                .NC_ENABLE      (0),
                .CORE_OUT_BUF   (3),
                .MEM_OUT_BUF    (3)
            ) icache (
                .clk            (clk),
                .reset          (reset),
                .core_bus_if    (immu_phys_if),
                .mem_bus_if     (ic_mem_if)
            );
        end else begin : g_no_immu
            // Keep the socket arbiter 2-wide (so TLB_SOCKET_ID_WIDTH matches the
            // package) with the iTLB port idle.
            assign socket_tlb_bus_if[1].req_valid = 1'b0;
            assign socket_tlb_bus_if[1].req_data  = '0;
            assign socket_tlb_bus_if[1].rsp_ready = 1'b0;
            `UNUSED_VAR (socket_tlb_bus_if[1].req_ready)
            `UNUSED_VAR (socket_tlb_bus_if[1].rsp_valid)
            `UNUSED_VAR (socket_tlb_bus_if[1].rsp_data)
            assign immu_empty[s] = 1'b1;
            for (genvar i = 0; i < I_LANES; ++i) begin : g_immu_tie
                localparam g = s * I_LANES + i;
                assign immu_req_ready[g] = 1'b0;
                assign immu_rsp_valid[g] = 1'b0;
                assign immu_rsp_data[g]  = '0;
                assign immu_rsp_tag[g]   = '0;
                `UNUSED_VAR (immu_req_valid[g])
                `UNUSED_VAR (immu_req_addr[g])
                `UNUSED_VAR (immu_req_tag[g])
                `UNUSED_VAR (immu_rsp_ready[g])
            end
            assign ic_mem_req_valid[s] = 1'b0;
            assign ic_mem_req_addr[s]  = '0;
            assign ic_mem_req_tag[s]   = '0;
            assign ic_mem_rsp_ready[s] = 1'b0;
            `UNUSED_VAR (ic_mem_req_ready[s])
            `UNUSED_VAR (ic_mem_rsp_valid[s])
            `UNUSED_VAR (ic_mem_rsp_data[s])
            `UNUSED_VAR (ic_mem_rsp_tag[s])
        end

        // --- socket TLB arbiter: {dTLB, iTLB} -> cluster fabric ---
        VX_tlb_bus_arb #(
            .NUM_INPUTS  (2),
            .ID_WIDTH_IN (L1_TLB_ID_WIDTH),
            .OUT_BUF     (3)
        ) tlb_socket_arb (
            .clk        (clk),
            .reset      (reset),
            .bus_in_if  (socket_tlb_bus_if),
            .bus_out_if (per_socket_tlb_bus_if[s])
        );
    end

    assign dcr_rsp_valid = (| dcr_out_valid) | (| dcr_in_rsp_valid);

    // ========================================================================
    // Cluster TLB + walker tail
    // ========================================================================
    VX_tlb_bus_if #(.ID_WIDTH (TLB_CLUSTER_ID_WIDTH)) l2_client_if ();
    VX_tlb_bus_if #(.ID_WIDTH (L2_TLB_SLOT_WIDTH))    l2_ptw_if ();

    VX_tlb_bus_arb #(
        .NUM_INPUTS  (NS),
        .ID_WIDTH_IN (TLB_SOCKET_ID_WIDTH),
        .OUT_BUF     (3)
    ) tlb_cluster_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (per_socket_tlb_bus_if),
        .bus_out_if (l2_client_if)
    );

    wire l2tlb_empty, ptw_empty;

    VX_tlb_l2 #(
        .INSTANCE_ID (`SFORMATF(("%s-l2tlb", INSTANCE_ID))),
        .ID_WIDTH    (TLB_CLUSTER_ID_WIDTH)
    ) l2tlb (
        .clk       (clk),
        .reset     (reset),
        .client_if (l2_client_if),
        .ptw_if    (l2_ptw_if),
        .flush_if  (l2_flush_if),
        .empty     (l2tlb_empty)
    );

    VX_mmu_fault_if ptw_fault_if ();
    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) ptw_mem_if ();

    VX_ptw #(
        .ID_WIDTH      (L2_TLB_SLOT_WIDTH),
        .MEM_TAG_WIDTH (L2_TAG_WIDTH)
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

    assign fault_valid  = ptw_fault_if.valid;
    assign fault_va     = ptw_fault_if.va;
    assign fault_access = ptw_fault_if.access;
    assign fault_amo    = ptw_fault_if.amo;

    // ========================================================================
    // Shared L2: sockets' L1 dcache mem ports + the PTW (the real client set,
    // minus graphics) -> memory ports
    // ========================================================================
    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) l2_core_if [L2_NUM_REQS] ();

    for (genvar i = 0; i < L2_NUM_REQS; ++i) begin : g_l2_clients
        if (i < L2_SOCKET_REQS) begin : g_socket
            // socket L1 mem port (dcache), widened to the L2 client tag width
            `ASSIGN_VX_MEM_BUS_IF_EX (l2_core_if[i], all_dc_mem_if[i], L2_TAG_WIDTH, DCACHE_MEM_TAG_WIDTH, UUID_WIDTH);
        end else if (i == L2_PTW_IDX) begin : g_ptw
            `ASSIGN_VX_MEM_BUS_IF (l2_core_if[i], ptw_mem_if);
        end else begin : g_tie
            // graphics client slots (unused in this DUT)
            assign l2_core_if[i].req_valid = 1'b0;
            assign l2_core_if[i].req_data  = '0;
            assign l2_core_if[i].rsp_ready = 1'b0;
            `UNUSED_VAR (l2_core_if[i].req_ready)
            `UNUSED_VAR (l2_core_if[i].rsp_valid)
            `UNUSED_VAR (l2_core_if[i].rsp_data)
        end
    end

    VX_mem_bus_if #(
        .DATA_SIZE (L2_SECTOR_SIZE),
        .TAG_WIDTH (L2_MEM_TAG_WIDTH)
    ) l2_mem_if [L2_MEM_PORTS] ();

    for (genvar i = 0; i < L2_MEM_PORTS; ++i) begin : g_l2_mem_bind
        assign l2_mem_req_valid[i]     = l2_mem_if[i].req_valid;
        assign l2_mem_req_rw[i]        = l2_mem_if[i].req_data.rw;
        assign l2_mem_req_byteen[i]    = l2_mem_if[i].req_data.byteen;
        assign l2_mem_req_addr[i]      = l2_mem_if[i].req_data.addr;
        assign l2_mem_req_tag[i]       = l2_mem_if[i].req_data.tag;
        assign l2_mem_req_data[i]      = l2_mem_if[i].req_data.data;
        assign l2_mem_if[i].req_ready  = l2_mem_req_ready[i];
        assign l2_mem_if[i].rsp_valid  = l2_mem_rsp_valid[i];
        assign l2_mem_if[i].rsp_data.data = l2_mem_rsp_data[i];
        assign l2_mem_if[i].rsp_data.tag  = l2_mem_rsp_tag[i];
        assign l2_mem_rsp_ready[i]     = l2_mem_if[i].rsp_ready;
        `UNUSED_VAR (l2_mem_if[i].req_data.attr)
    end

    VX_cache_wrap #(
        .INSTANCE_ID    (`SFORMATF(("%s-l2cache", INSTANCE_ID))),
        .CACHE_SIZE     (`VX_CFG_L2_SIZE),
        .LINE_SIZE      (`VX_CFG_L2_LINE_SIZE),
        .SECTOR_SIZE    (L2_SECTOR_SIZE),
        .NUM_BANKS      (L2_NUM_BANKS),
        .NUM_WAYS       (`VX_CFG_L2_NUM_WAYS),
        .WORD_SIZE      (L2_WORD_SIZE),
        .NUM_REQS       (L2_NUM_REQS),
        .MEM_PORTS      (L2_MEM_PORTS),
        .CRSQ_SIZE      (`VX_CFG_L2_CRSQ_SIZE),
        .MSHR_SIZE      (`VX_CFG_L2_MSHR_SIZE),
        .MRSQ_SIZE      (`VX_CFG_L2_MRSQ_SIZE),
        .MREQ_SIZE      (`VX_CFG_L2_MREQ_SIZE),
        .LATENCY        (`VX_CFG_L2_LATENCY),
        .TAG_WIDTH      (L2_TAG_WIDTH),
        .WRITE_ENABLE   (1),
        .WRITEBACK      (`VX_CFG_L2_WRITEBACK),
        .DIRTY_BYTES    (`VX_CFG_L2_DIRTYBYTES),
        .REPL_POLICY    (`VX_CFG_L2_REPL_POLICY),
        .CORE_OUT_BUF   (3),
        .MEM_OUT_BUF    (3),
        .NC_ENABLE      (1),
        .PASSTHRU       (!`VX_CFG_L2_ENABLED),
        .IS_LLC         (L2_IS_LLC),
        .AMO_ENABLE     (`VX_CFG_EXT_A_ENABLED)
    ) l2cache (
        .clk            (clk),
        .reset          (reset),
        .core_bus_if    (l2_core_if),
        .mem_bus_if     (l2_mem_if)
    );

    assign empty = (& dmmu_empty) && (& immu_empty) && l2tlb_empty && ptw_empty;

endmodule
