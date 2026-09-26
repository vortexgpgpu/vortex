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

// VX_socket with plain ports, for FireSim model multithreading (VX_CFG_FAME5).
//
// FAME-5 time-multiplexes identical instances of one module through a single
// physical copy, and it works on the design as FIRRTL. Vortex reaches FIRRTL
// through sv2v, which inlines every module with interface ports into its
// parent, so the unit to thread needs a boundary made of wires: this wrapper
// owns the socket's interfaces and exposes each field as a port. It takes no
// parameters and its index arrives on socket_id, so every instance elaborates
// to the same module.
//
// Covers the compute configuration: no VM, graphics, or scope.

module VX_socket_top import VX_gpu_pkg::*; #(
    parameter MREQ_W = 1 + (`VX_CFG_MEM_ADDR_WIDTH - `CLOG2(`VX_CFG_L1_LINE_SIZE)) + `VX_CFG_L1_LINE_SIZE * 8
                     + `VX_CFG_L1_LINE_SIZE + `UP(MEM_ATTR_WIDTH) + L2_TAG_WIDTH,
    parameter MRSP_W = `VX_CFG_L1_LINE_SIZE * 8 + L2_TAG_WIDTH
) (
    input  wire                                clk,
    input  wire                                reset,
    input  wire [31:0]                         socket_id,

`ifdef PERF_ENABLE
    input  sysmem_perf_t                       sysmem_perf,
`endif

    input  wire                                dcr_req_valid,
    input  dcr_req_t                           dcr_req_data,
    output wire                                dcr_rsp_valid,
    output dcr_rsp_t                           dcr_rsp_data,

    output wire [L1_MEM_PORTS-1:0]             mem_req_valid,
    output wire [L1_MEM_PORTS-1:0][MREQ_W-1:0] mem_req_data,
    input  wire [L1_MEM_PORTS-1:0]             mem_req_ready,
    input  wire [L1_MEM_PORTS-1:0]             mem_rsp_valid,
    input  wire [L1_MEM_PORTS-1:0][MRSP_W-1:0] mem_rsp_data,
    output wire [L1_MEM_PORTS-1:0]             mem_rsp_ready,

    input  wire                                kmu_valid,
    input  wire                                kmu_kind,
    input  wire                                kmu_eop,
    input  wire [KMU_DEST_W-1:0]               kmu_dest,
    input  wire [KMU_DATAW-1:0]                kmu_data,
    output wire                                kmu_ready,

    output wire                                gbar_req_valid,
    output gbar_req_t                          gbar_req_data,
    input  wire                                gbar_req_ready,
    input  wire                                gbar_rsp_valid,
    input  gbar_rsp_t                          gbar_rsp_data,
    output wire                                gbar_rsp_ready,

    output wire                                busy
);
`ifdef VX_CFG_VM_ENABLE
    `STATIC_ASSERT(0, ("VX_socket_top does not carry the VM ports yet"))
`endif
`ifdef EXT_GFX_ANY_ENABLE
    `STATIC_ASSERT(0, ("VX_socket_top does not carry the graphics ports yet"))
`endif

    VX_dcr_bus_if dcr_bus_if();
    assign dcr_bus_if.req_valid = dcr_req_valid;
    assign dcr_bus_if.req_data  = dcr_req_data;
    assign dcr_rsp_valid = dcr_bus_if.rsp_valid;
    assign dcr_rsp_data  = dcr_bus_if.rsp_data;

    VX_mem_bus_if #(
        .DATA_SIZE (`VX_CFG_L1_LINE_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) mem_bus_if[L1_MEM_PORTS]();

    for (genvar i = 0; i < L1_MEM_PORTS; ++i) begin : g_mem
        assign mem_req_valid[i] = mem_bus_if[i].req_valid;
        assign mem_req_data[i]  = mem_bus_if[i].req_data;
        assign mem_bus_if[i].req_ready = mem_req_ready[i];
        assign mem_bus_if[i].rsp_valid = mem_rsp_valid[i];
        assign mem_bus_if[i].rsp_data  = mem_rsp_data[i];
        assign mem_rsp_ready[i] = mem_bus_if[i].rsp_ready;
    end

    VX_kmu_bus_if kmu_bus_if[1]();
    assign kmu_bus_if[0].valid = kmu_valid;
    assign kmu_bus_if[0].kind  = kmu_kind;
    assign kmu_bus_if[0].eop   = kmu_eop;
    assign kmu_bus_if[0].dest  = kmu_dest;
    assign kmu_bus_if[0].data  = kmu_data;
    assign kmu_ready = kmu_bus_if[0].ready;

    VX_gbar_bus_if gbar_bus_if();
    assign gbar_req_valid = gbar_bus_if.req_valid;
    assign gbar_req_data  = gbar_bus_if.req_data;
    assign gbar_bus_if.req_ready = gbar_req_ready;
    assign gbar_bus_if.rsp_valid = gbar_rsp_valid;
    assign gbar_bus_if.rsp_data  = gbar_rsp_data;
    assign gbar_rsp_ready = gbar_bus_if.rsp_ready;

    VX_socket socket (
        .clk        (clk),
        .reset      (reset),
        .socket_id  (socket_id),
    `ifdef PERF_ENABLE
        .sysmem_perf(sysmem_perf),
    `endif
        .dcr_bus_if (dcr_bus_if),
        .mem_bus_if (mem_bus_if),
        .kmu_bus_if (kmu_bus_if),
        .gbar_bus_if(gbar_bus_if),
        .busy       (busy)
    );

endmodule
