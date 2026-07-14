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
// seam the SLR floorplan relies on). Each endpoint registers the channels it
// SOURCES and leaves the rest passthrough for the far endpoint to register:
// the unit sources arm/req, the RTU core sources win (SLV_OUT_BUF). All use the
// standard TO_OUT_BUF encoding (0 = passthrough).
//
// `arm` MUST STAY UNBUFFERED (ARM_OUT_BUF = 0). Its ready is the RTU's "I am
// idle and can take a ray", and the TRACE burst's first uop rides it: buffer the
// arm and that uop retires while the RTU is still busy, so the ray beats behind
// it stream into a traversal that is not theirs — corrupting it, and leaving the
// real arm waiting forever for beats that were already consumed. A skid buffer
// here does not add latency, it breaks the protocol.
//
// A buffered channel adds latency but never reorders, so the win channel's
// write ordering — status slot last — survives the slice.
// ============================================================================

`TRACING_OFF
module VX_rtu_bus_slice import VX_gpu_pkg::*, VX_rtu_pkg::*; #(
    parameter NUM_LANES   = 1,
    parameter TAG_WIDTH   = 1,
    parameter ARM_OUT_BUF = 0,  // arm  (sourced by the unit)
    parameter REQ_OUT_BUF = 0,  // req  (sourced by the unit)
    parameter SLV_OUT_BUF = 0   // win  (sourced by the RTU core)
) (
    input wire              clk,
    input wire              reset,
    VX_rtu_bus_if.slave     bus_in_if,
    VX_rtu_bus_if.master    bus_out_if
);
    // Packed widths of the channel payload structs (mirror VX_rtu_bus_if). A
    // parameter cannot read $bits of a hierarchical interface member, so size
    // the elastic buffers from the field formula instead.
    localparam ARM_DATAW = NW_WIDTH + NUM_LANES
                         + `VX_CFG_MEM_ADDR_WIDTH + 16 + 16 + 32 + TAG_WIDTH;
    localparam REQ_DATAW = 1 + NUM_LANES * 32 + NUM_LANES * RTU_CB_ACTION_BITS;
    localparam WIN_DATAW = 1 + NW_WIDTH + RTU_SLOT_BITS
                         + NUM_LANES + NUM_LANES * 32 + TAG_WIDTH;

    // The formulas above are hand-mirrored from VX_rtu_bus_if, and a stale one
    // undersizes an elastic buffer and TRUNCATES the payload silently. Pin each
    // to the struct it mirrors: $bits is illegal in a parameter's initializer but
    // fine in an elaboration check.
    `STATIC_ASSERT((ARM_DATAW == $bits(bus_in_if.arm_data)),
        ("VX_rtu_bus_if arm_data_t width changed: fix ARM_DATAW"))
    `STATIC_ASSERT((REQ_DATAW == $bits(bus_in_if.req_data)),
        ("VX_rtu_bus_if req_data_t width changed: fix REQ_DATAW"))
    `STATIC_ASSERT((WIN_DATAW == $bits(bus_in_if.win_data)),
        ("VX_rtu_bus_if win_data_t width changed: fix WIN_DATAW"))

    // See the header: buffering the arm breaks the ray-delivery protocol.
    `STATIC_ASSERT((ARM_OUT_BUF == 0),
        ("the RTU arm channel must not be buffered"))

    // ---- arm : bus_in -> bus_out ----
    wire [ARM_DATAW-1:0] arm_data_in = bus_in_if.arm_data;
    wire [ARM_DATAW-1:0] arm_data_out;

    VX_elastic_buffer #(
        .DATAW   (ARM_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(ARM_OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(ARM_OUT_BUF)),
        .LUTRAM  (`TO_OUT_BUF_LUTRAM(ARM_OUT_BUF))
    ) arm_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (bus_in_if.arm_valid),
        .ready_in  (bus_in_if.arm_ready),
        .data_in   (arm_data_in),
        .data_out  (arm_data_out),
        .valid_out (bus_out_if.arm_valid),
        .ready_out (bus_out_if.arm_ready)
    );
    assign bus_out_if.arm_data = arm_data_out;

    // ---- req : bus_in -> bus_out ----
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

    // ---- win : bus_out -> bus_in ----
    wire [WIN_DATAW-1:0] win_data_in = bus_out_if.win_data;
    wire [WIN_DATAW-1:0] win_data_out;

    VX_elastic_buffer #(
        .DATAW   (WIN_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(SLV_OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(SLV_OUT_BUF)),
        .LUTRAM  (`TO_OUT_BUF_LUTRAM(SLV_OUT_BUF))
    ) win_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (bus_out_if.win_valid),
        .ready_in  (bus_out_if.win_ready),
        .data_in   (win_data_in),
        .data_out  (win_data_out),
        .valid_out (bus_in_if.win_valid),
        .ready_out (bus_in_if.win_ready)
    );
    assign bus_in_if.win_data = win_data_out;

endmodule
`TRACING_ON
