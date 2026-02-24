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

module VX_dxa_cluster_engine_array import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_DXA_UNITS = 1,
    parameter ENABLE = 1
) (
    input wire clk,
    input wire reset,

    VX_dcr_bus_if.slave dcr_bus_if,
    VX_dxa_req_bus_if.slave cluster_dxa_bus_if[NUM_DXA_UNITS],
    VX_mem_bus_if.master dxa_gmem_bus_if[NUM_DXA_UNITS],
    VX_mem_bus_if.master dxa_smem_bus_if[NUM_DXA_UNITS]
);
    VX_mem_bus_if #(
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LMEM_TAG_WIDTH)
    ) dxa_smem_narrow_bus_if[NUM_DXA_UNITS]();

    if (ENABLE) begin : g_dxa_engine
        for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_units
            VX_dxa_engine #(
                .INSTANCE_ID (`SFORMATF(("%s-dxa%0d", INSTANCE_ID, i))),
                .ENGINE_ID   (i)
            ) dxa_engine (
                .clk        (clk),
                .reset      (reset),
                .dcr_bus_if (dcr_bus_if),
                .dxa_req_bus_if (cluster_dxa_bus_if[i]),
                .gmem_bus_if(dxa_gmem_bus_if[i]),
                .smem_bus_if(dxa_smem_narrow_bus_if[i])
            );

            VX_dxa_smem_upsize #(
                .SRC_DATA_SIZE (LSU_WORD_SIZE),
                .DST_DATA_SIZE (DXA_SMEM_WORD_SIZE),
                .TAG_WIDTH     (LMEM_TAG_WIDTH),
                .SRC_ADDR_WIDTH(LSU_ADDR_WIDTH),
                .DST_ADDR_WIDTH(DXA_SMEM_ADDR_WIDTH)
            ) smem_upsize (
                .clk       (clk),
                .reset     (reset),
                .src_bus_if(dxa_smem_narrow_bus_if[i]),
                .dst_bus_if(dxa_smem_bus_if[i])
            );
        end
    end else begin : g_dxa_engine_off
        for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_dxa_engine_off_i
            assign cluster_dxa_bus_if[i].req_ready = 1'b1;
            assign cluster_dxa_bus_if[i].rsp_valid = 1'b0;
            assign cluster_dxa_bus_if[i].rsp_data  = '0;
            `UNUSED_VAR (cluster_dxa_bus_if[i].req_valid)
            `UNUSED_VAR (cluster_dxa_bus_if[i].req_data)

            assign dxa_gmem_bus_if[i].req_valid = 1'b0;
            assign dxa_gmem_bus_if[i].req_data  = '0;
            assign dxa_gmem_bus_if[i].rsp_ready = 1'b1;

            assign dxa_smem_narrow_bus_if[i].req_valid = 1'b0;
            assign dxa_smem_narrow_bus_if[i].req_data  = '0;
            assign dxa_smem_narrow_bus_if[i].req_ready = 1'b1;
            assign dxa_smem_narrow_bus_if[i].rsp_valid = 1'b0;
            assign dxa_smem_narrow_bus_if[i].rsp_data  = '0;
            assign dxa_smem_narrow_bus_if[i].rsp_ready = 1'b1;

            assign dxa_smem_bus_if[i].req_valid = 1'b0;
            assign dxa_smem_bus_if[i].req_data  = '0;
            assign dxa_smem_bus_if[i].rsp_ready = 1'b1;
        end
    end
endmodule
