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

module VX_dxa_cluster_smem_xbar import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter DXA_NUM_SOCKETS = 1,
    parameter NUM_DXA_UNITS = 1,
    parameter CORE_LOCAL_BITS = 0,
    parameter ENABLE = 1
) (
    input wire clk,
    input wire reset,

    VX_mem_bus_if.slave dxa_smem_bus_if[NUM_DXA_UNITS],
    VX_mem_bus_if.master per_socket_dxa_smem_bus_if[DXA_NUM_SOCKETS]
);
    localparam DXA_SMEM_TAG_WIDTH = LMEM_TAG_WIDTH;
    localparam DXA_SMEM_ENGINE_BITS = `CLOG2(NUM_DXA_UNITS);
    localparam DXA_SMEM_ENGINE_W = `UP(DXA_SMEM_ENGINE_BITS);
    localparam DXA_SOCKET_SEL_BITS = `CLOG2(DXA_NUM_SOCKETS);
    localparam DXA_SOCKET_SEL_W = `UP(DXA_SOCKET_SEL_BITS);
    localparam DXA_SMEM_CORE_LSB = DXA_SMEM_ENGINE_W;
    localparam DXA_SMEM_REQ_DATAW = 1
                                  + DXA_SMEM_ADDR_WIDTH
                                  + (DXA_SMEM_WORD_SIZE * 8)
                                  + DXA_SMEM_WORD_SIZE
                                  + MEM_FLAGS_WIDTH
                                  + LMEM_TAG_WIDTH;
    localparam DXA_SMEM_RSP_DATAW = (DXA_SMEM_WORD_SIZE * 8) + LMEM_TAG_WIDTH;

    if (ENABLE) begin : g_smem_route
        wire [NUM_DXA_UNITS-1:0] smem_req_valid_in;
        wire [NUM_DXA_UNITS-1:0][DXA_SMEM_REQ_DATAW-1:0] smem_req_data_in;
        wire [NUM_DXA_UNITS-1:0][DXA_SOCKET_SEL_W-1:0] smem_req_sel_in;
        wire [NUM_DXA_UNITS-1:0] smem_req_ready_in;

        wire [DXA_NUM_SOCKETS-1:0] smem_req_valid_out;
        wire [DXA_NUM_SOCKETS-1:0][DXA_SMEM_REQ_DATAW-1:0] smem_req_data_out;
        wire [DXA_NUM_SOCKETS-1:0][`UP(`CLOG2(NUM_DXA_UNITS))-1:0] smem_req_sel_out;
        wire [DXA_NUM_SOCKETS-1:0] smem_req_ready_out;

        for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_dxa_smem_req_in
            /* verilator lint_off UNUSEDSIGNAL */
            wire [DXA_SMEM_TAG_WIDTH-1:0] smem_req_tag_route = {
                dxa_smem_bus_if[i].req_data.tag.uuid,
                dxa_smem_bus_if[i].req_data.tag.value
            };
            /* verilator lint_on UNUSEDSIGNAL */
            wire [NC_WIDTH-1:0] smem_req_core_id =
                NC_WIDTH'(smem_req_tag_route[DXA_SMEM_CORE_LSB +: NC_WIDTH]);

            assign smem_req_valid_in[i] = dxa_smem_bus_if[i].req_valid;
            assign smem_req_data_in[i] = dxa_smem_bus_if[i].req_data;
            assign smem_req_sel_in[i] = DXA_SOCKET_SEL_W'(smem_req_core_id >> CORE_LOCAL_BITS);
            assign dxa_smem_bus_if[i].req_ready = smem_req_ready_in[i];
        end

        VX_stream_xbar #(
            .NUM_INPUTS  (NUM_DXA_UNITS),
            .NUM_OUTPUTS (DXA_NUM_SOCKETS),
            .DATAW       (DXA_SMEM_REQ_DATAW),
            .ARBITER     ("R"),
            .OUT_BUF     ((NUM_DXA_UNITS != DXA_NUM_SOCKETS) ? 2 : 0)
        ) dxa_smem_req_xbar (
            .clk       (clk),
            .reset     (reset),
            `UNUSED_PIN (collisions),
            .valid_in  (smem_req_valid_in),
            .data_in   (smem_req_data_in),
            .sel_in    (smem_req_sel_in),
            .ready_in  (smem_req_ready_in),
            .valid_out (smem_req_valid_out),
            .data_out  (smem_req_data_out),
            .sel_out   (smem_req_sel_out),
            .ready_out (smem_req_ready_out)
        );
        `UNUSED_VAR (smem_req_sel_out)

        for (genvar i = 0; i < DXA_NUM_SOCKETS; ++i) begin : g_dxa_smem_req_out
            assign per_socket_dxa_smem_bus_if[i].req_valid = smem_req_valid_out[i];
            assign per_socket_dxa_smem_bus_if[i].req_data = smem_req_data_out[i];
            assign smem_req_ready_out[i] = per_socket_dxa_smem_bus_if[i].req_ready;
        end

        wire [DXA_NUM_SOCKETS-1:0] smem_rsp_valid_in;
        wire [DXA_NUM_SOCKETS-1:0][DXA_SMEM_RSP_DATAW-1:0] smem_rsp_data_in;
        wire [DXA_NUM_SOCKETS-1:0][DXA_SMEM_ENGINE_W-1:0] smem_rsp_sel_in;
        wire [DXA_NUM_SOCKETS-1:0] smem_rsp_ready_in;

        wire [NUM_DXA_UNITS-1:0] smem_rsp_valid_out;
        wire [NUM_DXA_UNITS-1:0][DXA_SMEM_RSP_DATAW-1:0] smem_rsp_data_out;
        wire [NUM_DXA_UNITS-1:0][`UP(`CLOG2(DXA_NUM_SOCKETS))-1:0] smem_rsp_sel_out;
        wire [NUM_DXA_UNITS-1:0] smem_rsp_ready_out;

        for (genvar i = 0; i < DXA_NUM_SOCKETS; ++i) begin : g_dxa_smem_rsp_in
            /* verilator lint_off UNUSEDSIGNAL */
            wire [DXA_SMEM_TAG_WIDTH-1:0] smem_rsp_tag_route = {
                per_socket_dxa_smem_bus_if[i].rsp_data.tag.uuid,
                per_socket_dxa_smem_bus_if[i].rsp_data.tag.value
            };
            /* verilator lint_on UNUSEDSIGNAL */
            assign smem_rsp_valid_in[i] = per_socket_dxa_smem_bus_if[i].rsp_valid;
            assign smem_rsp_data_in[i] = per_socket_dxa_smem_bus_if[i].rsp_data;
            assign smem_rsp_sel_in[i] = DXA_SMEM_ENGINE_W'(smem_rsp_tag_route[DXA_SMEM_ENGINE_W-1:0]);
            assign per_socket_dxa_smem_bus_if[i].rsp_ready = smem_rsp_ready_in[i];
        end

        VX_stream_xbar #(
            .NUM_INPUTS  (DXA_NUM_SOCKETS),
            .NUM_OUTPUTS (NUM_DXA_UNITS),
            .DATAW       (DXA_SMEM_RSP_DATAW),
            .ARBITER     ("R"),
            .OUT_BUF     ((DXA_NUM_SOCKETS != NUM_DXA_UNITS) ? 2 : 0)
        ) dxa_smem_rsp_xbar (
            .clk       (clk),
            .reset     (reset),
            `UNUSED_PIN (collisions),
            .valid_in  (smem_rsp_valid_in),
            .data_in   (smem_rsp_data_in),
            .sel_in    (smem_rsp_sel_in),
            .ready_in  (smem_rsp_ready_in),
            .valid_out (smem_rsp_valid_out),
            .data_out  (smem_rsp_data_out),
            .sel_out   (smem_rsp_sel_out),
            .ready_out (smem_rsp_ready_out)
        );
        `UNUSED_VAR (smem_rsp_sel_out)

        for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_dxa_smem_rsp_out
            assign dxa_smem_bus_if[i].rsp_valid = smem_rsp_valid_out[i];
            assign dxa_smem_bus_if[i].rsp_data  = smem_rsp_data_out[i];
            assign smem_rsp_ready_out[i] = dxa_smem_bus_if[i].rsp_ready;
        end
    end else begin : g_smem_route_off
        for (genvar i = 0; i < DXA_NUM_SOCKETS; ++i) begin : g_dxa_smem_socket_off
            assign per_socket_dxa_smem_bus_if[i].req_valid = 1'b0;
            assign per_socket_dxa_smem_bus_if[i].req_data  = '0;
            assign per_socket_dxa_smem_bus_if[i].rsp_ready = 1'b1;
        end
        for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_dxa_smem_engine_off
            assign dxa_smem_bus_if[i].req_ready = 1'b1;
            assign dxa_smem_bus_if[i].rsp_valid = 1'b0;
            assign dxa_smem_bus_if[i].rsp_data  = '0;
        end
    end

endmodule
