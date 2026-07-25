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

// N:1 arbiter for the narrow TLB miss/fill fabric. Requests round-robin
// down to one output with the grant index prepended into the id; responses
// route back by peeling those bits. One elastic slice per hierarchy level.
module VX_tlb_arb import VX_tlb_pkg::*; #(
    parameter NUM_INPUTS   = 2,
    parameter ID_WIDTH_IN  = 4,
    parameter `STRING ARBITER = "R",
    parameter OUT_BUF      = 1,
    parameter SEL_BITS     = `ARB_SEL_BITS(NUM_INPUTS, 1),
    parameter ID_WIDTH_OUT = ID_WIDTH_IN + SEL_BITS
) (
    input wire clk,
    input wire reset,
    VX_tlb_bus_if.slave  bus_in_if [NUM_INPUTS],
    VX_tlb_bus_if.master bus_out_if
);
    localparam ID_IN       = `UP(ID_WIDTH_IN);
    localparam REQ_REST_W  = $bits(tlb_access_e) + 1 + TLB_VPN_WIDTH;
    localparam RSP_REST_W  = 1 + TLB_LEVEL_WIDTH + TLB_PPN_WIDTH + TLB_FLAGS_WIDTH;
    localparam REQ_DATAW_IN  = ID_IN + REQ_REST_W;
    localparam RSP_DATAW_OUT = ID_WIDTH_OUT + RSP_REST_W;
    localparam RSP_DATAW_IN  = ID_IN + RSP_REST_W;

    if (NUM_INPUTS == 1) begin : g_passthru
        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        assign bus_out_if.req_valid = bus_in_if[0].req_valid;
        assign bus_out_if.req_data  = bus_in_if[0].req_data;
        assign bus_in_if[0].req_ready = bus_out_if.req_ready;
        assign bus_in_if[0].rsp_valid = bus_out_if.rsp_valid;
        assign bus_in_if[0].rsp_data  = bus_out_if.rsp_data;
        assign bus_out_if.rsp_ready = bus_in_if[0].rsp_ready;
    end else begin : g_arb

        // ---- Requests: NUM_INPUTS -> 1, prepend grant index into the id ----
        wire [NUM_INPUTS-1:0]                  req_valid_in;
        wire [NUM_INPUTS-1:0][REQ_DATAW_IN-1:0] req_data_in;
        wire [NUM_INPUTS-1:0]                  req_ready_in;

        for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_req_in
            assign req_valid_in[i] = bus_in_if[i].req_valid;
            assign req_data_in[i]  = bus_in_if[i].req_data;
            assign bus_in_if[i].req_ready = req_ready_in[i];
        end

        wire                     ser_req_valid;
        wire [REQ_DATAW_IN-1:0]   ser_req_data;
        wire [`UP(SEL_BITS)-1:0]  ser_req_sel;
        wire                     ser_req_ready;

        VX_stream_arb #(
            .NUM_INPUTS  (NUM_INPUTS),
            .NUM_OUTPUTS (1),
            .DATAW       (REQ_DATAW_IN),
            .ARBITER     (ARBITER),
            .OUT_BUF     (OUT_BUF)
        ) req_arb (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (req_valid_in),
            .data_in   (req_data_in),
            .ready_in  (req_ready_in),
            .valid_out (ser_req_valid),
            .data_out  (ser_req_data),
            .sel_out   (ser_req_sel),
            .ready_out (ser_req_ready)
        );

        // id is the top field of req_data, so prepending sel widens the id.
        assign bus_out_if.req_valid = ser_req_valid;
        assign bus_out_if.req_data  = {ser_req_sel, ser_req_data};
        assign ser_req_ready = bus_out_if.req_ready;

        // ---- Responses: 1 -> NUM_INPUTS, route by the prepended sel bits ----
        wire [SEL_BITS-1:0]        rsp_sel = bus_out_if.rsp_data[RSP_DATAW_OUT-1 -: SEL_BITS];
        wire [RSP_DATAW_IN-1:0]    rsp_data_restored = bus_out_if.rsp_data[RSP_DATAW_IN-1:0];

        wire [NUM_INPUTS-1:0]                  rsp_valid_out;
        wire [NUM_INPUTS-1:0][RSP_DATAW_IN-1:0] rsp_data_out;
        wire [NUM_INPUTS-1:0]                  rsp_ready_out;

        VX_stream_switch #(
            .NUM_INPUTS  (1),
            .NUM_OUTPUTS (NUM_INPUTS),
            .DATAW       (RSP_DATAW_IN),
            .OUT_BUF     (OUT_BUF)
        ) rsp_switch (
            .clk       (clk),
            .reset     (reset),
            .sel_in    (rsp_sel),
            .valid_in  (bus_out_if.rsp_valid),
            .data_in   (rsp_data_restored),
            .ready_in  (bus_out_if.rsp_ready),
            .valid_out (rsp_valid_out),
            .data_out  (rsp_data_out),
            .ready_out (rsp_ready_out)
        );

        for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_rsp_out
            assign bus_in_if[i].rsp_valid = rsp_valid_out[i];
            assign bus_in_if[i].rsp_data  = rsp_data_out[i];
            assign rsp_ready_out[i] = bus_in_if[i].rsp_ready;
        end
    end

endmodule
