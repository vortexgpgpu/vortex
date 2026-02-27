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
    parameter NUM_CORE_OUTPUTS = DXA_NUM_SOCKETS * (1 << CORE_LOCAL_BITS),
    parameter ENABLE = 0
) (
    input wire clk,
    input wire reset,

    VX_dcr_bus_if.slave dcr_bus_if,

    VX_dxa_req_bus_if.slave per_socket_dxa_bus_if[DXA_NUM_SOCKETS],
    VX_dxa_bank_wr_if.master per_core_bank_wr_if[NUM_CORE_OUTPUTS],
    VX_mem_bus_if.master dxa_gmem_bus_if[GMEM_OUT_PORTS]
);

    // Whether we need a socket-level arb (fewer router outputs than total cores).
    localparam NEED_SOCKET_ARB  = (DXA_SMEM_PORTS_PER_SOCKET < `SOCKET_SIZE);
    localparam ROUTER_OUTPUTS   = NEED_SOCKET_ARB ? (DXA_NUM_SOCKETS * DXA_SMEM_PORTS_PER_SOCKET) : NUM_CORE_OUTPUTS;
    localparam ROUTER_SEL_W     = `UP(`CLOG2(ROUTER_OUTPUTS));
    localparam ROUTER_CORE_ID_W = NEED_SOCKET_ARB ? DXA_SMEM_LOCAL_CORE_BITS : 0;

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
        .per_socket_dxa_bus_if(per_socket_dxa_bus_if),
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
        .bus_out_if (dxa_gmem_bus_if)
    );

    // Compute routing sel and local_core_id sideband for each worker.
    wire [NUM_DXA_UNITS-1:0][ROUTER_SEL_W-1:0] worker_output_sel;
    wire [NUM_DXA_UNITS-1:0][`UP(ROUTER_CORE_ID_W)-1:0] worker_local_core_id;

    for (genvar w = 0; w < NUM_DXA_UNITS; ++w) begin : g_worker_sel
        if (NEED_SOCKET_ARB) begin : g_reduced
            // socket_id = core_id >> CORE_LOCAL_BITS
            // local_core_id = core_id[CORE_LOCAL_BITS-1:0]
            // port_within_socket = local_core_id % PORTS_PER_SOCKET
            localparam PORTS_BITS = `CLOG2(DXA_SMEM_PORTS_PER_SOCKET);
            wire [`UP(CORE_LOCAL_BITS)-1:0] local_cid = worker_smem_core_id[w][`UP(CORE_LOCAL_BITS)-1:0];
            wire [NC_WIDTH-1:0] socket_id = worker_smem_core_id[w] >> CORE_LOCAL_BITS;
            assign worker_output_sel[w] = ROUTER_SEL_W'(socket_id * DXA_SMEM_PORTS_PER_SOCKET + local_cid[`UP(PORTS_BITS)-1:0]);
            assign worker_local_core_id[w] = `UP(ROUTER_CORE_ID_W)'(local_cid);
        end else begin : g_passthrough
            assign worker_output_sel[w] = ROUTER_SEL_W'(worker_smem_core_id[w]);
            assign worker_local_core_id[w] = '0;
        end
    end

    if (NEED_SOCKET_ARB) begin : g_socket_arb
        // Router outputs fewer ports; socket_arb expands to per-core.
        VX_dxa_bank_wr_if #(
            .NUM_BANKS       (`LMEM_NUM_BANKS),
            .BANK_ADDR_WIDTH (DXA_SMEM_BANK_ADDR_WIDTH),
            .WORD_SIZE       (`XLEN / 8),
            .TAG_WIDTH       (DXA_BANK_WR_TAG_WIDTH)
        ) router_out_bank_wr_if[ROUTER_OUTPUTS]();

        wire [ROUTER_OUTPUTS-1:0][DXA_SMEM_LOCAL_CORE_W-1:0] router_out_local_core_id;

        VX_dxa_smem_core_router #(
            .NUM_INPUTS    (NUM_DXA_UNITS),
            .NUM_OUTPUTS   (ROUTER_OUTPUTS),
            .CORE_ID_WIDTH (ROUTER_CORE_ID_W),
            .ENABLE        (ENABLE)
        ) dxa_smem_router (
            .clk                  (clk),
            .reset                (reset),
            .worker_bank_wr_if    (worker_bank_wr_if),
            .worker_output_sel    (worker_output_sel),
            .worker_local_core_id (worker_local_core_id),
            .out_bank_wr_if       (router_out_bank_wr_if),
            .out_local_core_id    (router_out_local_core_id)
        );

        // Per-socket arb: PORTS_PER_SOCKET → SOCKET_SIZE
        for (genvar s = 0; s < DXA_NUM_SOCKETS; ++s) begin : g_per_socket_arb
            /* verilator lint_off UNUSEDSIGNAL */
            wire [`SOCKET_SIZE-1:0][DXA_SMEM_LOCAL_CORE_W-1:0] socket_arb_lcid_out;
            /* verilator lint_on UNUSEDSIGNAL */

            VX_dxa_smem_socket_arb #(
                .NUM_INPUTS  (DXA_SMEM_PORTS_PER_SOCKET),
                .NUM_OUTPUTS (`SOCKET_SIZE)
            ) socket_arb (
                .clk              (clk),
                .reset            (reset),
                .bank_wr_in       (router_out_bank_wr_if[s * DXA_SMEM_PORTS_PER_SOCKET +: DXA_SMEM_PORTS_PER_SOCKET]),
                .local_core_id_in (router_out_local_core_id[s * DXA_SMEM_PORTS_PER_SOCKET +: DXA_SMEM_PORTS_PER_SOCKET]),
                .bank_wr_out      (per_core_bank_wr_if[s * `SOCKET_SIZE +: `SOCKET_SIZE]),
                .local_core_id_out(socket_arb_lcid_out)
            );
        end

    end else begin : g_direct
        // Direct 1:1 routing (PORTS_PER_SOCKET == SOCKET_SIZE).
        /* verilator lint_off UNUSEDSIGNAL */
        wire [NUM_CORE_OUTPUTS-1:0][`UP(ROUTER_CORE_ID_W)-1:0] unused_router_local_core_id;
        /* verilator lint_on UNUSEDSIGNAL */

        VX_dxa_smem_core_router #(
            .NUM_INPUTS    (NUM_DXA_UNITS),
            .NUM_OUTPUTS   (NUM_CORE_OUTPUTS),
            .CORE_ID_WIDTH (0),
            .ENABLE        (ENABLE)
        ) dxa_smem_router (
            .clk                  (clk),
            .reset                (reset),
            .worker_bank_wr_if    (worker_bank_wr_if),
            .worker_output_sel    (worker_output_sel),
            .worker_local_core_id (worker_local_core_id),
            .out_bank_wr_if       (per_core_bank_wr_if),
            .out_local_core_id    (unused_router_local_core_id)
        );
    end

endmodule
