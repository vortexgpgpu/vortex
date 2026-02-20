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

module VX_tma_cluster_ctrl import VX_gpu_pkg::*; #(
    parameter TMA_NUM_SOCKETS = 1,
    parameter NUM_TMA_UNITS = 1,
    parameter CORE_LOCAL_BITS = 0,
    parameter ENABLE = 1
) (
    input wire clk,
    input wire reset,

    VX_tma_bus_if.slave per_socket_tma_bus_if[TMA_NUM_SOCKETS],
    VX_tma_bus_if.master cluster_tma_bus_if[NUM_TMA_UNITS]
);
    localparam TMA_REQ_DATAW = NC_WIDTH + UUID_WIDTH + NW_WIDTH + 3 + (2 * `XLEN);
    localparam TMA_RSP_DATAW = NC_WIDTH + UUID_WIDTH + NW_WIDTH + BAR_ADDR_W + 1 + 1;
    localparam TMA_REQ_SEL_COUNT = `MIN(TMA_NUM_SOCKETS, NUM_TMA_UNITS);
    localparam TMA_REQ_NUM_REQS = (TMA_NUM_SOCKETS > NUM_TMA_UNITS)
                                ? `CDIV(TMA_NUM_SOCKETS, NUM_TMA_UNITS)
                                : `CDIV(NUM_TMA_UNITS, TMA_NUM_SOCKETS);
    localparam TMA_REQ_SEL_W = `UP(`CLOG2(TMA_REQ_NUM_REQS));
    localparam TMA_SOCKET_SEL_BITS = `CLOG2(TMA_NUM_SOCKETS);
    localparam TMA_SOCKET_SEL_W = `UP(TMA_SOCKET_SEL_BITS);

    if (ENABLE) begin : g_ctrl
        wire [TMA_NUM_SOCKETS-1:0] tma_req_valid_in;
        wire [TMA_NUM_SOCKETS-1:0][TMA_REQ_DATAW-1:0] tma_req_data_in;
        wire [TMA_NUM_SOCKETS-1:0] tma_req_ready_in;
        wire [TMA_NUM_SOCKETS-1:0] tma_rsp_ready_socket_in;

        for (genvar i = 0; i < TMA_NUM_SOCKETS; ++i) begin : g_tma_req_in
            assign tma_req_valid_in[i] = per_socket_tma_bus_if[i].req_valid;
            assign tma_req_data_in[i] = per_socket_tma_bus_if[i].req_data;
            assign per_socket_tma_bus_if[i].req_ready = tma_req_ready_in[i];
            assign tma_rsp_ready_socket_in[i] = per_socket_tma_bus_if[i].rsp_ready;
        end

        wire [NUM_TMA_UNITS-1:0] tma_req_valid_out;
        wire [NUM_TMA_UNITS-1:0][TMA_REQ_DATAW-1:0] tma_req_data_out;
        wire [NUM_TMA_UNITS-1:0] tma_req_ready_out;
        wire [TMA_REQ_SEL_COUNT-1:0][TMA_REQ_SEL_W-1:0] tma_req_sel_out;

        VX_stream_arb #(
            .NUM_INPUTS  (TMA_NUM_SOCKETS),
            .NUM_OUTPUTS (NUM_TMA_UNITS),
            .DATAW       (TMA_REQ_DATAW),
            .ARBITER     ("R"),
            .OUT_BUF     ((TMA_NUM_SOCKETS != NUM_TMA_UNITS) ? 2 : 0)
        ) tma_req_arb (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (tma_req_valid_in),
            .data_in    (tma_req_data_in),
            .ready_in   (tma_req_ready_in),
            .valid_out  (tma_req_valid_out),
            .data_out   (tma_req_data_out),
            .ready_out  (tma_req_ready_out),
            .sel_out    (tma_req_sel_out)
        );

        for (genvar i = 0; i < NUM_TMA_UNITS; ++i) begin : g_tma_req_out
            assign cluster_tma_bus_if[i].req_valid = tma_req_valid_out[i];
            assign cluster_tma_bus_if[i].req_data  = tma_req_data_out[i];
            assign tma_req_ready_out[i] = cluster_tma_bus_if[i].req_ready;
        end
        `UNUSED_VAR (tma_req_sel_out)

        wire [NUM_TMA_UNITS-1:0] tma_rsp_valid_in;
        wire [NUM_TMA_UNITS-1:0][TMA_RSP_DATAW-1:0] tma_rsp_data_in;
        wire [NUM_TMA_UNITS-1:0] tma_rsp_ready_in;

        for (genvar i = 0; i < NUM_TMA_UNITS; ++i) begin : g_tma_rsp_in
            assign tma_rsp_valid_in[i] = cluster_tma_bus_if[i].rsp_valid;
            assign tma_rsp_data_in[i] = cluster_tma_bus_if[i].rsp_data;
            assign cluster_tma_bus_if[i].rsp_ready = tma_rsp_ready_in[i];
        end

        wire [0:0] tma_rsp_valid_out;
        wire [0:0][TMA_RSP_DATAW-1:0] tma_rsp_data_out;
        wire [0:0] tma_rsp_ready_out;
        wire [0:0][`UP(`CLOG2(NUM_TMA_UNITS))-1:0] tma_rsp_sel_out;

        VX_stream_arb #(
            .NUM_INPUTS  (NUM_TMA_UNITS),
            .NUM_OUTPUTS (1),
            .DATAW       (TMA_RSP_DATAW),
            .ARBITER     ("R"),
            .OUT_BUF     (2)
        ) tma_rsp_arb (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (tma_rsp_valid_in),
            .data_in    (tma_rsp_data_in),
            .ready_in   (tma_rsp_ready_in),
            .valid_out  (tma_rsp_valid_out),
            .data_out   (tma_rsp_data_out),
            .ready_out  (tma_rsp_ready_out),
            .sel_out    (tma_rsp_sel_out)
        );
        `UNUSED_VAR (tma_rsp_sel_out)

        wire [NC_WIDTH-1:0] tma_rsp_core_id = tma_rsp_data_out[0][TMA_RSP_DATAW-1 -: NC_WIDTH];
        wire [TMA_SOCKET_SEL_W-1:0] tma_rsp_socket_sel =
            TMA_SOCKET_SEL_W'(tma_rsp_core_id >> CORE_LOCAL_BITS);

        reg tma_rsp_ready_out_r;
        always @(*) begin
            tma_rsp_ready_out_r = 1'b0;
            for (integer i = 0; i < TMA_NUM_SOCKETS; ++i) begin
                if (tma_rsp_socket_sel == TMA_SOCKET_SEL_W'(i)) begin
                    tma_rsp_ready_out_r = tma_rsp_ready_socket_in[i];
                end
            end
        end

        assign tma_rsp_ready_out[0] = tma_rsp_ready_out_r;

        for (genvar i = 0; i < TMA_NUM_SOCKETS; ++i) begin : g_tma_rsp_out
            assign per_socket_tma_bus_if[i].rsp_valid = tma_rsp_valid_out[0]
                                                      && (tma_rsp_socket_sel == TMA_SOCKET_SEL_W'(i));
            assign per_socket_tma_bus_if[i].rsp_data  = tma_rsp_data_out[0];
        end
    end else begin : g_ctrl_off
        for (genvar i = 0; i < TMA_NUM_SOCKETS; ++i) begin : g_socket_off
            assign per_socket_tma_bus_if[i].req_ready = 1'b1;
            assign per_socket_tma_bus_if[i].rsp_valid = 1'b0;
            assign per_socket_tma_bus_if[i].rsp_data  = '0;
            `UNUSED_VAR (per_socket_tma_bus_if[i].req_valid)
            `UNUSED_VAR (per_socket_tma_bus_if[i].req_data)
            `UNUSED_VAR (per_socket_tma_bus_if[i].rsp_ready)
        end

        for (genvar i = 0; i < NUM_TMA_UNITS; ++i) begin : g_unit_off
            assign cluster_tma_bus_if[i].req_valid = 1'b0;
            assign cluster_tma_bus_if[i].req_data  = '0;
            assign cluster_tma_bus_if[i].rsp_ready = 1'b1;
            `UNUSED_VAR (cluster_tma_bus_if[i].req_ready)
            `UNUSED_VAR (cluster_tma_bus_if[i].rsp_valid)
            `UNUSED_VAR (cluster_tma_bus_if[i].rsp_data)
        end
    end

endmodule
