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

module VX_dxa_cluster import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter DXA_NUM_SOCKETS = 1,
    parameter NUM_DXA_UNITS = 1,
    parameter L2_MEM_PORTS = 1,
    parameter CORE_LOCAL_BITS = 0,
    parameter ENABLE = 0
) (
    input wire clk,
    input wire reset,

    VX_dcr_bus_if.slave dcr_bus_if,

    VX_dxa_req_bus_if.slave per_socket_dxa_bus_if[DXA_NUM_SOCKETS],
    VX_mem_bus_if.master per_socket_dxa_smem_bus_if[DXA_NUM_SOCKETS],

    VX_mem_bus_if.slave l2_mem_bus_if[L2_MEM_PORTS],
    VX_mem_bus_if.master mem_bus_if[L2_MEM_PORTS]
);
    VX_dxa_req_bus_if cluster_dxa_bus_if[NUM_DXA_UNITS]();

    VX_mem_bus_if #(
        .DATA_SIZE (`L2_LINE_SIZE),
        .TAG_WIDTH (L2_MEM_TAG_WIDTH)
    ) dxa_gmem_bus_if[NUM_DXA_UNITS]();

    VX_mem_bus_if #(
        .DATA_SIZE (DXA_SMEM_WORD_SIZE),
        .TAG_WIDTH (LMEM_TAG_WIDTH)
    ) dxa_smem_bus_if[NUM_DXA_UNITS]();

    VX_dxa_cluster_ctrl #(
        .DXA_NUM_SOCKETS(DXA_NUM_SOCKETS),
        .NUM_DXA_UNITS  (NUM_DXA_UNITS),
        .CORE_LOCAL_BITS(CORE_LOCAL_BITS),
        .ENABLE         (ENABLE)
    ) dxa_ctrl (
        .clk                  (clk),
        .reset                (reset),
        .per_socket_dxa_bus_if(per_socket_dxa_bus_if),
        .cluster_dxa_bus_if   (cluster_dxa_bus_if)
    );

    VX_dxa_cluster_engine_array #(
        .INSTANCE_ID  (`SFORMATF(("%s-engines", INSTANCE_ID))),
        .NUM_DXA_UNITS(NUM_DXA_UNITS),
        .ENABLE       (ENABLE)
    ) dxa_engine_array (
        .clk               (clk),
        .reset             (reset),
        .dcr_bus_if        (dcr_bus_if),
        .cluster_dxa_bus_if(cluster_dxa_bus_if),
        .dxa_gmem_bus_if   (dxa_gmem_bus_if),
        .dxa_smem_bus_if   (dxa_smem_bus_if)
    );

    VX_dxa_cluster_smem_xbar #(
        .DXA_NUM_SOCKETS(DXA_NUM_SOCKETS),
        .NUM_DXA_UNITS  (NUM_DXA_UNITS),
        .CORE_LOCAL_BITS(CORE_LOCAL_BITS),
        .ENABLE         (ENABLE)
    ) dxa_smem_xbar (
        .clk                    (clk),
        .reset                  (reset),
        .dxa_smem_bus_if        (dxa_smem_bus_if),
        .per_socket_dxa_smem_bus_if(per_socket_dxa_smem_bus_if)
    );

    VX_dxa_cluster_gmem_path #(
        .NUM_DXA_UNITS(NUM_DXA_UNITS),
        .L2_MEM_PORTS (L2_MEM_PORTS),
        .OUT_TAG_WIDTH(L3_TAG_WIDTH),
        .ENABLE       (ENABLE)
    ) dxa_gmem_path (
        .clk          (clk),
        .reset        (reset),
        .dxa_gmem_bus_if(dxa_gmem_bus_if),
        .l2_mem_bus_if(l2_mem_bus_if),
        .mem_bus_if   (mem_bus_if)
    );

endmodule
