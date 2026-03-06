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
    parameter CORE_LOCAL_BITS = 0,
    parameter ENABLE = 0
) (
    input wire clk,
    input wire reset,

    VX_dcr_bus_if.slave dcr_bus_if,

    VX_dxa_req_bus_if.slave req_bus_if[DXA_NUM_SOCKETS],
    VX_dxa_bank_wr_if.master smem_bus_if[DXA_NUM_SOCKETS * DXA_SMEM_PORTS_PER_SOCKET],
    output wire [DXA_NUM_SOCKETS * DXA_SMEM_PORTS_PER_SOCKET-1:0][DXA_SMEM_LOCAL_CORE_W-1:0] smem_local_core_id,
    VX_mem_bus_if.master gmem_bus_if[GMEM_OUT_PORTS]
);

    localparam NUM_SMEM_OUTPUTS  = DXA_NUM_SOCKETS * DXA_SMEM_PORTS_PER_SOCKET;
    localparam NEED_SOCKET_ARB   = (DXA_SMEM_PORTS_PER_SOCKET < `SOCKET_SIZE);
    localparam ROUTER_SEL_W      = `UP(`CLOG2(NUM_SMEM_OUTPUTS));
    localparam ROUTER_CORE_ID_W  = NEED_SOCKET_ARB ? DXA_SMEM_LOCAL_CORE_BITS : 0;

    VX_dxa_req_bus_if cluster_dxa_bus_if[NUM_DXA_UNITS]();

    VX_dxa_bank_wr_if #(
        .NUM_BANKS       (`LMEM_NUM_BANKS),
        .BANK_ADDR_WIDTH (DXA_SMEM_BANK_ADDR_WIDTH),
        .WORD_SIZE       (`XLEN / 8),
        .TAG_WIDTH       (DXA_BANK_WR_TAG_WIDTH)
    ) worker_bank_wr_if[NUM_DXA_UNITS]();

    wire [NUM_DXA_UNITS-1:0][NC_WIDTH-1:0] worker_smem_core_id;

    VX_dxa_core_ctrl #(
        .DXA_NUM_SOCKETS(DXA_NUM_SOCKETS),
        .NUM_DXA_UNITS  (NUM_DXA_UNITS),
        .CORE_LOCAL_BITS(CORE_LOCAL_BITS),
        .ENABLE         (ENABLE)
    ) dxa_ctrl (
        .clk                  (clk),
        .reset                (reset),
        .per_socket_dxa_bus_if(req_bus_if),
        .cluster_dxa_bus_if   (cluster_dxa_bus_if)
    );

    // Internal worker gmem buses (pre-distribution)
    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
    ) worker_gmem_bus_if[NUM_DXA_UNITS]();

    VX_dxa_unified_engine #(
        .INSTANCE_ID  (`SFORMATF(("%s-unified", INSTANCE_ID))),
        .NUM_DXA_UNITS(NUM_DXA_UNITS),
        .ENABLE       (ENABLE)
    ) dxa_unified_engine (
        .clk                (clk),
        .reset              (reset),
        .dcr_bus_if         (dcr_bus_if),
        .cluster_dxa_bus_if (cluster_dxa_bus_if),
        .dxa_gmem_bus_if    (worker_gmem_bus_if),
        .dxa_smem_bank_wr_if(worker_bank_wr_if),
        .dxa_smem_core_id   (worker_smem_core_id)
    );

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

    // Compute routing sel and local_core_id sideband for each worker.
    wire [NUM_DXA_UNITS-1:0][ROUTER_SEL_W-1:0] worker_output_sel;
    wire [NUM_DXA_UNITS-1:0][`UP(ROUTER_CORE_ID_W)-1:0] worker_local_core_id;

    for (genvar w = 0; w < NUM_DXA_UNITS; ++w) begin : g_worker_sel
        if (NEED_SOCKET_ARB) begin : g_reduced
            localparam PORTS_BITS = `CLOG2(DXA_SMEM_PORTS_PER_SOCKET);
            wire [`UP(CORE_LOCAL_BITS)-1:0] local_cid = worker_smem_core_id[w][`UP(CORE_LOCAL_BITS)-1:0];
            wire [NC_WIDTH-1:0] socket_id = worker_smem_core_id[w] >> CORE_LOCAL_BITS;
            assign worker_output_sel[w] = ROUTER_SEL_W'(32'(socket_id) * DXA_SMEM_PORTS_PER_SOCKET + 32'(local_cid[`UP(PORTS_BITS)-1:0]));
            assign worker_local_core_id[w] = `UP(ROUTER_CORE_ID_W)'(local_cid);
        end else begin : g_direct
            assign worker_output_sel[w] = ROUTER_SEL_W'(worker_smem_core_id[w]);
            assign worker_local_core_id[w] = '0;
        end
    end

    // Single router with local_core_id sideband.
    // Socket-level fan-out (if needed) is handled downstream in VX_socket.
    VX_dxa_smem_core_router #(
        .NUM_INPUTS    (NUM_DXA_UNITS),
        .NUM_OUTPUTS   (NUM_SMEM_OUTPUTS),
        .CORE_ID_WIDTH (ROUTER_CORE_ID_W),
        .ENABLE        (ENABLE)
    ) dxa_smem_router (
        .clk                  (clk),
        .reset                (reset),
        .worker_bank_wr_if    (worker_bank_wr_if),
        .worker_output_sel    (worker_output_sel),
        .worker_local_core_id (worker_local_core_id),
        .out_bank_wr_if       (smem_bus_if),
        .out_local_core_id    (smem_local_core_id)
    );

endmodule
