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

// ============================================================================
// VX_rtu_bus_slice — point-to-point register slice for a VX_rtu_bus_if.
//
// Lets a producer register the RTU bus it drives at its own output boundary,
// terminating an SLR-crossing route at a flop (the registered module-boundary
// seam the SLR floorplan relies on). Following the request/response ownership
// split, a master registers its forward request (REQ_OUT_BUF) and a slave its
// forward response (RSP_OUT_BUF); the opposite direction stays passthrough and
// is registered by the far endpoint. REQ_OUT_BUF / RSP_OUT_BUF use the standard
// TO_OUT_BUF encoding (0 = passthrough).
// ============================================================================

`TRACING_OFF
module VX_rtu_bus_slice import VX_gpu_pkg::*, VX_rtu_pkg::*; #(
    parameter NUM_LANES   = 1,
    parameter TAG_WIDTH   = 1,
    parameter REQ_OUT_BUF = 0,
    parameter RSP_OUT_BUF = 0
) (
    input wire              clk,
    input wire              reset,
    VX_rtu_bus_if.slave     bus_in_if,
    VX_rtu_bus_if.master    bus_out_if
);
    // Packed widths of the beat-serial req/rsp payload structs (mirror
    // VX_rtu_bus_if / VX_rtu_arb). A parameter cannot read $bits of a
    // hierarchical interface member, so size the elastic buffers from the field
    // formula instead. req: {tag, kind, eop, mask, data, cb_action, scene_base};
    // rsp: {tag, kind, eop, data, cb_active_mask}.
    localparam REQ_DATAW = TAG_WIDTH + 1 + 1 + `VX_CFG_MEM_ADDR_WIDTH + NUM_LANES * (1 + 32)
                         + NUM_LANES * RTU_CB_ACTION_BITS;
    localparam RSP_DATAW = TAG_WIDTH + 1 + 1 + NUM_LANES * (32 + 1);

    // ---- Request : bus_in -> bus_out ----
    wire [REQ_DATAW-1:0] req_data_in = bus_in_if.req_data;
    wire [REQ_DATAW-1:0] req_data_out;

    VX_elastic_buffer #(
        .DATAW   (REQ_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(REQ_OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(REQ_OUT_BUF)),
        .LUTRAM  (`TO_OUT_BUF_LUTRAM(REQ_OUT_BUF))
    ) req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (bus_in_if.req_valid),
        .ready_in  (bus_in_if.req_ready),
        .data_in   (req_data_in),
        .data_out  (req_data_out),
        .valid_out (bus_out_if.req_valid),
        .ready_out (bus_out_if.req_ready)
    );
    assign bus_out_if.req_data = req_data_out;

    // ---- Response : bus_out -> bus_in ----
    wire [RSP_DATAW-1:0] rsp_data_in = bus_out_if.rsp_data;
    wire [RSP_DATAW-1:0] rsp_data_out;

    VX_elastic_buffer #(
        .DATAW   (RSP_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(RSP_OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(RSP_OUT_BUF)),
        .LUTRAM  (`TO_OUT_BUF_LUTRAM(RSP_OUT_BUF))
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (bus_out_if.rsp_valid),
        .ready_in  (bus_out_if.rsp_ready),
        .data_in   (rsp_data_in),
        .data_out  (rsp_data_out),
        .valid_out (bus_in_if.rsp_valid),
        .ready_out (bus_in_if.rsp_ready)
    );
    assign bus_in_if.rsp_data = rsp_data_out;

endmodule
`TRACING_ON
