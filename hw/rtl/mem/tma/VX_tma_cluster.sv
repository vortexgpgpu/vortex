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

module VX_tma_cluster import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter TMA_NUM_SOCKETS = 1,
    parameter NUM_TMA_UNITS = 1,
    parameter L2_MEM_PORTS = 1,
    parameter CORE_LOCAL_BITS = 0,
    parameter ENABLE = 0
) (
    input wire clk,
    input wire reset,

    VX_dcr_bus_if.slave dcr_bus_if,

    VX_tma_bus_if.slave per_socket_tma_bus_if[TMA_NUM_SOCKETS],
    VX_mem_bus_if.master per_socket_tma_smem_bus_if[TMA_NUM_SOCKETS],

    VX_mem_bus_if.slave l2_mem_bus_if[L2_MEM_PORTS],
    VX_mem_bus_if.master mem_bus_if[L2_MEM_PORTS]
);
    VX_tma_bus_if cluster_tma_bus_if[NUM_TMA_UNITS]();

    VX_mem_bus_if #(
        .DATA_SIZE (`L2_LINE_SIZE),
        .TAG_WIDTH (L2_MEM_TAG_WIDTH)
    ) tma_gmem_bus_if[NUM_TMA_UNITS]();

    VX_mem_bus_if #(
        .DATA_SIZE (TMA_SMEM_WORD_SIZE),
        .TAG_WIDTH (LMEM_TAG_WIDTH)
    ) tma_smem_bus_if[NUM_TMA_UNITS]();

    VX_tma_cluster_ctrl #(
        .TMA_NUM_SOCKETS(TMA_NUM_SOCKETS),
        .NUM_TMA_UNITS  (NUM_TMA_UNITS),
        .CORE_LOCAL_BITS(CORE_LOCAL_BITS),
        .ENABLE         (ENABLE)
    ) tma_ctrl (
        .clk                  (clk),
        .reset                (reset),
        .per_socket_tma_bus_if(per_socket_tma_bus_if),
        .cluster_tma_bus_if   (cluster_tma_bus_if)
    );

    VX_tma_cluster_engine_array #(
        .INSTANCE_ID  (`SFORMATF(("%s-engines", INSTANCE_ID))),
        .NUM_TMA_UNITS(NUM_TMA_UNITS),
        .ENABLE       (ENABLE)
    ) tma_engine_array (
        .clk               (clk),
        .reset             (reset),
        .dcr_bus_if        (dcr_bus_if),
        .cluster_tma_bus_if(cluster_tma_bus_if),
        .tma_gmem_bus_if   (tma_gmem_bus_if),
        .tma_smem_bus_if   (tma_smem_bus_if)
    );

    VX_tma_cluster_smem_xbar #(
        .TMA_NUM_SOCKETS(TMA_NUM_SOCKETS),
        .NUM_TMA_UNITS  (NUM_TMA_UNITS),
        .CORE_LOCAL_BITS(CORE_LOCAL_BITS),
        .ENABLE         (ENABLE)
    ) tma_smem_xbar (
        .clk                    (clk),
        .reset                  (reset),
        .tma_smem_bus_if        (tma_smem_bus_if),
        .per_socket_tma_smem_bus_if(per_socket_tma_smem_bus_if)
    );

    VX_tma_cluster_gmem_path #(
        .NUM_TMA_UNITS(NUM_TMA_UNITS),
        .L2_MEM_PORTS (L2_MEM_PORTS),
        .OUT_TAG_WIDTH(L3_TAG_WIDTH),
        .ENABLE       (ENABLE)
    ) tma_gmem_path (
        .clk          (clk),
        .reset        (reset),
        .tma_gmem_bus_if(tma_gmem_bus_if),
        .l2_mem_bus_if(l2_mem_bus_if),
        .mem_bus_if   (mem_bus_if)
    );

endmodule
