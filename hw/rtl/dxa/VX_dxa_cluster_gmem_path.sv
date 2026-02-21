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

module VX_dxa_cluster_gmem_path import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter NUM_DXA_UNITS = 1,
    parameter L2_MEM_PORTS = 1,
    parameter OUT_TAG_WIDTH = L3_TAG_WIDTH,
    parameter ENABLE = 1
) (
    input wire clk,
    input wire reset,

    VX_mem_bus_if.slave dxa_gmem_bus_if[NUM_DXA_UNITS],
    VX_mem_bus_if.slave l2_mem_bus_if[L2_MEM_PORTS],
    VX_mem_bus_if.master mem_bus_if[L2_MEM_PORTS]
);
    localparam DXA_ARB1_SEL_BITS   = `ARB_SEL_BITS(NUM_DXA_UNITS, 1);
    localparam DXA_ARB2_SEL_BITS   = `ARB_SEL_BITS(2, 1);
    localparam DXA_ARB1_TAG_WIDTH  = L2_MEM_TAG_WIDTH + DXA_ARB1_SEL_BITS;
    localparam DXA_ARB2_TAG_WIDTH  = DXA_ARB1_TAG_WIDTH + DXA_ARB2_SEL_BITS;

    if (ENABLE) begin : g_dxa_gmem_mempath
        `STATIC_ASSERT(OUT_TAG_WIDTH >= DXA_ARB2_TAG_WIDTH, ("invalid parameter"))

        VX_mem_bus_if #(
            .DATA_SIZE (`L2_LINE_SIZE),
            .TAG_WIDTH (DXA_ARB1_TAG_WIDTH)
        ) dxa_gmem_arb_bus_if[1]();

        VX_mem_arb #(
            .NUM_INPUTS (NUM_DXA_UNITS),
            .NUM_OUTPUTS(1),
            .DATA_SIZE  (`L2_LINE_SIZE),
            .TAG_WIDTH  (L2_MEM_TAG_WIDTH),
            .TAG_SEL_IDX(0),
            .ARBITER    ("R"),
            .REQ_OUT_BUF(2),
            .RSP_OUT_BUF(2)
        ) dxa_gmem_req_arb (
            .clk        (clk),
            .reset      (reset),
            .bus_in_if  (dxa_gmem_bus_if),
            .bus_out_if (dxa_gmem_arb_bus_if)
        );

        VX_mem_bus_if #(
            .DATA_SIZE (`L2_LINE_SIZE),
            .TAG_WIDTH (DXA_ARB1_TAG_WIDTH)
        ) mem0_arb_in_if[2]();

        VX_mem_bus_if #(
            .DATA_SIZE (`L2_LINE_SIZE),
            .TAG_WIDTH (DXA_ARB2_TAG_WIDTH)
        ) mem0_arb_out_if[1]();

        `ASSIGN_VX_MEM_BUS_IF_EX (mem0_arb_in_if[0], l2_mem_bus_if[0], DXA_ARB1_TAG_WIDTH, L2_MEM_TAG_WIDTH, UUID_WIDTH);
        `ASSIGN_VX_MEM_BUS_IF (mem0_arb_in_if[1], dxa_gmem_arb_bus_if[0]);

        VX_mem_arb #(
            .NUM_INPUTS (2),
            .NUM_OUTPUTS(1),
            .DATA_SIZE  (`L2_LINE_SIZE),
            .TAG_WIDTH  (DXA_ARB1_TAG_WIDTH),
            .TAG_SEL_IDX(0),
            .ARBITER    ("R"),
            .REQ_OUT_BUF(2),
            .RSP_OUT_BUF(2)
        ) dxa_mem_port0_arb (
            .clk        (clk),
            .reset      (reset),
            .bus_in_if  (mem0_arb_in_if),
            .bus_out_if (mem0_arb_out_if)
        );

        for (genvar i = 0; i < L2_MEM_PORTS; ++i) begin : g_l2_mem_out
            if (i == 0) begin : g_l2_mem_out_0
                `ASSIGN_VX_MEM_BUS_IF_EX (mem_bus_if[i], mem0_arb_out_if[0], OUT_TAG_WIDTH, DXA_ARB2_TAG_WIDTH, UUID_WIDTH);
            end else begin : g_l2_mem_out_i
                `ASSIGN_VX_MEM_BUS_IF_EX (mem_bus_if[i], l2_mem_bus_if[i], OUT_TAG_WIDTH, L2_MEM_TAG_WIDTH, UUID_WIDTH);
            end
        end
    end else begin : g_dxa_gmem_mempath_off
        for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_dxa_gmem_stub_i
            assign dxa_gmem_bus_if[i].req_ready = 1'b1;
            assign dxa_gmem_bus_if[i].rsp_valid = 1'b0;
            assign dxa_gmem_bus_if[i].rsp_data  = '0;
        end

        for (genvar i = 0; i < L2_MEM_PORTS; ++i) begin : g_l2_mem_out
            `ASSIGN_VX_MEM_BUS_IF_EX (mem_bus_if[i], l2_mem_bus_if[i], OUT_TAG_WIDTH, L2_MEM_TAG_WIDTH, UUID_WIDTH);
        end
    end

endmodule
