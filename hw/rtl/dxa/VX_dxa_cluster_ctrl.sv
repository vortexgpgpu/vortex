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

module VX_dxa_cluster_ctrl import VX_gpu_pkg::*; #(
    parameter DXA_NUM_SOCKETS = 1,
    parameter NUM_DXA_UNITS = 1,
    parameter CORE_LOCAL_BITS = 0,
    parameter ENABLE = 1
) (
    input wire clk,
    input wire reset,

    VX_dxa_req_bus_if.slave per_socket_dxa_bus_if[DXA_NUM_SOCKETS],
    VX_dxa_req_bus_if.master cluster_dxa_bus_if[NUM_DXA_UNITS]
);
    localparam DXA_REQ_DATAW = NC_WIDTH + UUID_WIDTH + NW_WIDTH + 3 + (2 * `XLEN);
    localparam DXA_RSP_DATAW = NC_WIDTH + UUID_WIDTH + NW_WIDTH + BAR_ADDR_W + 1 + 1;
    localparam DXA_REQ_SEL_COUNT = `MIN(DXA_NUM_SOCKETS, NUM_DXA_UNITS);
    localparam DXA_REQ_NUM_REQS = (DXA_NUM_SOCKETS > NUM_DXA_UNITS)
                                ? `CDIV(DXA_NUM_SOCKETS, NUM_DXA_UNITS)
                                : `CDIV(NUM_DXA_UNITS, DXA_NUM_SOCKETS);
    localparam DXA_REQ_SEL_W = `UP(`CLOG2(DXA_REQ_NUM_REQS));
    localparam DXA_SOCKET_SEL_BITS = `CLOG2(DXA_NUM_SOCKETS);
    localparam DXA_SOCKET_SEL_W = `UP(DXA_SOCKET_SEL_BITS);

    if (ENABLE) begin : g_ctrl
        wire [DXA_NUM_SOCKETS-1:0] dxa_req_valid_in;
        wire [DXA_NUM_SOCKETS-1:0][DXA_REQ_DATAW-1:0] dxa_req_data_in;
        wire [DXA_NUM_SOCKETS-1:0] dxa_req_ready_in;
        wire [DXA_NUM_SOCKETS-1:0] dxa_rsp_ready_socket_in;

        for (genvar i = 0; i < DXA_NUM_SOCKETS; ++i) begin : g_dxa_req_in
            assign dxa_req_valid_in[i] = per_socket_dxa_bus_if[i].req_valid;
            assign dxa_req_data_in[i] = per_socket_dxa_bus_if[i].req_data;
            assign per_socket_dxa_bus_if[i].req_ready = dxa_req_ready_in[i];
            assign dxa_rsp_ready_socket_in[i] = per_socket_dxa_bus_if[i].rsp_ready;
        end

        wire [NUM_DXA_UNITS-1:0] dxa_req_valid_out;
        wire [NUM_DXA_UNITS-1:0][DXA_REQ_DATAW-1:0] dxa_req_data_out;
        wire [NUM_DXA_UNITS-1:0] dxa_req_ready_out;
        wire [DXA_REQ_SEL_COUNT-1:0][DXA_REQ_SEL_W-1:0] dxa_req_sel_out;

        VX_stream_arb #(
            .NUM_INPUTS  (DXA_NUM_SOCKETS),
            .NUM_OUTPUTS (NUM_DXA_UNITS),
            .DATAW       (DXA_REQ_DATAW),
            .ARBITER     ("R"),
            .OUT_BUF     ((DXA_NUM_SOCKETS != NUM_DXA_UNITS) ? 2 : 0)
        ) dxa_req_arb (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (dxa_req_valid_in),
            .data_in    (dxa_req_data_in),
            .ready_in   (dxa_req_ready_in),
            .valid_out  (dxa_req_valid_out),
            .data_out   (dxa_req_data_out),
            .ready_out  (dxa_req_ready_out),
            .sel_out    (dxa_req_sel_out)
        );

        for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_dxa_req_out
            assign cluster_dxa_bus_if[i].req_valid = dxa_req_valid_out[i];
            assign cluster_dxa_bus_if[i].req_data  = dxa_req_data_out[i];
            assign dxa_req_ready_out[i] = cluster_dxa_bus_if[i].req_ready;
        end
        `UNUSED_VAR (dxa_req_sel_out)

        wire [NUM_DXA_UNITS-1:0] dxa_rsp_valid_in;
        wire [NUM_DXA_UNITS-1:0][DXA_RSP_DATAW-1:0] dxa_rsp_data_in;
        wire [NUM_DXA_UNITS-1:0] dxa_rsp_ready_in;

        for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_dxa_rsp_in
            assign dxa_rsp_valid_in[i] = cluster_dxa_bus_if[i].rsp_valid;
            assign dxa_rsp_data_in[i] = cluster_dxa_bus_if[i].rsp_data;
            assign cluster_dxa_bus_if[i].rsp_ready = dxa_rsp_ready_in[i];
        end

        wire [0:0] dxa_rsp_valid_out;
        wire [0:0][DXA_RSP_DATAW-1:0] dxa_rsp_data_out;
        wire [0:0] dxa_rsp_ready_out;
        wire [0:0][`UP(`CLOG2(NUM_DXA_UNITS))-1:0] dxa_rsp_sel_out;

        VX_stream_arb #(
            .NUM_INPUTS  (NUM_DXA_UNITS),
            .NUM_OUTPUTS (1),
            .DATAW       (DXA_RSP_DATAW),
            .ARBITER     ("R"),
            .OUT_BUF     (2)
        ) dxa_rsp_arb (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (dxa_rsp_valid_in),
            .data_in    (dxa_rsp_data_in),
            .ready_in   (dxa_rsp_ready_in),
            .valid_out  (dxa_rsp_valid_out),
            .data_out   (dxa_rsp_data_out),
            .ready_out  (dxa_rsp_ready_out),
            .sel_out    (dxa_rsp_sel_out)
        );
        `UNUSED_VAR (dxa_rsp_sel_out)

        wire [NC_WIDTH-1:0] dxa_rsp_core_id = dxa_rsp_data_out[0][DXA_RSP_DATAW-1 -: NC_WIDTH];
        wire [DXA_SOCKET_SEL_W-1:0] dxa_rsp_socket_sel =
            DXA_SOCKET_SEL_W'(dxa_rsp_core_id >> CORE_LOCAL_BITS);

        reg dxa_rsp_ready_out_r;
        always @(*) begin
            dxa_rsp_ready_out_r = 1'b0;
            for (integer i = 0; i < DXA_NUM_SOCKETS; ++i) begin
                if (dxa_rsp_socket_sel == DXA_SOCKET_SEL_W'(i)) begin
                    dxa_rsp_ready_out_r = dxa_rsp_ready_socket_in[i];
                end
            end
        end

        assign dxa_rsp_ready_out[0] = dxa_rsp_ready_out_r;

        for (genvar i = 0; i < DXA_NUM_SOCKETS; ++i) begin : g_dxa_rsp_out
            assign per_socket_dxa_bus_if[i].rsp_valid = dxa_rsp_valid_out[0]
                                                      && (dxa_rsp_socket_sel == DXA_SOCKET_SEL_W'(i));
            assign per_socket_dxa_bus_if[i].rsp_data  = dxa_rsp_data_out[0];
        end
    end else begin : g_ctrl_off
        for (genvar i = 0; i < DXA_NUM_SOCKETS; ++i) begin : g_socket_off
            assign per_socket_dxa_bus_if[i].req_ready = 1'b1;
            assign per_socket_dxa_bus_if[i].rsp_valid = 1'b0;
            assign per_socket_dxa_bus_if[i].rsp_data  = '0;
            `UNUSED_VAR (per_socket_dxa_bus_if[i].req_valid)
            `UNUSED_VAR (per_socket_dxa_bus_if[i].req_data)
            `UNUSED_VAR (per_socket_dxa_bus_if[i].rsp_ready)
        end

        for (genvar i = 0; i < NUM_DXA_UNITS; ++i) begin : g_unit_off
            assign cluster_dxa_bus_if[i].req_valid = 1'b0;
            assign cluster_dxa_bus_if[i].req_data  = '0;
            assign cluster_dxa_bus_if[i].rsp_ready = 1'b1;
            `UNUSED_VAR (cluster_dxa_bus_if[i].req_ready)
            `UNUSED_VAR (cluster_dxa_bus_if[i].rsp_valid)
            `UNUSED_VAR (cluster_dxa_bus_if[i].rsp_data)
        end
    end

endmodule
