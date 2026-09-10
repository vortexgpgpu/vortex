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

// VX_rtu_bus_arb — connects the socket's core windows to its RTU cores.
//
// Cores are bound to an RTU STATICALLY, in contiguous groups of NUM_INPUTS /
// NUM_OUTPUTS: core i is always served by RTU i / CORES_PER_RTU. A dynamic
// arbiter would be free to send a warp's TRACE to one RTU and its later
// CONTINUE to another, orphaning the traversal that is parked waiting for it —
// the binding has to outlive a single transfer, and the transfers of one trace
// are separated by the whole traversal.
//
// Within a group the three channels are independent (see VX_rtu_bus_if):
//   arm  cores -> RTU, arbitrated; the winner's index is folded into the tag
//   req  cores -> RTU, arbitrated; only the core with the RTU's active trace
//                 ever drives it, so it never actually contends
//   win  RTU -> cores, switched back by the core index recovered from the tag
//
// `arm` may block (the RTU accepts one only when idle) without blocking `req`,
// which is what keeps a parked traversal reachable while another warp queues.

`include "VX_define.vh"

module VX_rtu_bus_arb import VX_gpu_pkg::*, VX_rtu_pkg::*; #(
    parameter NUM_INPUTS  = 1,
    parameter NUM_OUTPUTS = 1,
    parameter NUM_LANES   = 1,
    parameter TAG_WIDTH   = 1,
    parameter TAG_SEL_IDX = 0,
    parameter OUT_BUF_ARM = 0,
    parameter OUT_BUF_REQ = 0,
    parameter OUT_BUF_WIN = 0,
    parameter `STRING ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,

    VX_rtu_bus_if.slave     bus_in_if [NUM_INPUTS],
    VX_rtu_bus_if.master    bus_out_if [NUM_OUTPUTS]
);
    localparam CORES_PER_RTU = NUM_INPUTS / NUM_OUTPUTS;
    localparam LOG_NUM_REQS  = `ARB_SEL_BITS(CORES_PER_RTU, 1);
    localparam SEL_W         = `UP(LOG_NUM_REQS);

    // The arm MAY be buffered: a beat names its owner ({src, wid}) and the RTU
    // stages a ray per owner, so a TRACE retiring before its ray is taken cannot
    // let the beats behind it land in somebody else's traversal. See VX_rtu_bus_slice.

    `STATIC_ASSERT((NUM_INPUTS >= NUM_OUTPUTS),
        ("RTU cores must not outnumber the cores they serve"))
    `STATIC_ASSERT(((CORES_PER_RTU * NUM_OUTPUTS) == NUM_INPUTS),
        ("socket cores must divide evenly among the socket's RTU cores"))

    // The `src` field is not transported: it is MEANINGLESS on the way in (a unit
    // does not know its own index) and is re-derived here from the arbiter's own
    // grant. Only the RTU-core side carries a real one, which is why an input bus
    // is always instanced SRC_WIDTH=1 and an output bus SRC_WIDTH=SEL_W.
    localparam SRC_IN_W = 1;

    localparam ARM_DATAW = NW_WIDTH + NUM_LANES
                         + `VX_CFG_MEM_ADDR_WIDTH + 16 + 16 + 32 + TAG_WIDTH;
    localparam REQ_DATAW = 1 + NW_WIDTH + NUM_LANES * 32 + NUM_LANES * RTU_CB_ACTION_BITS;
    // the win payload carries the sel-stripped tag on the core side
    localparam WIN_DATAW = 1 + NW_WIDTH + RTU_SLOT_BITS
                         + NUM_LANES + NUM_LANES * 32 + TAG_WIDTH;

    // Pin the hand-mirrored widths to the structs they mirror (see VX_rtu_bus_slice).
    `STATIC_ASSERT((ARM_DATAW + SRC_IN_W == $bits(bus_in_if[0].arm_data)),
        ("VX_rtu_bus_if arm_data_t width changed: fix ARM_DATAW"))
    `STATIC_ASSERT((REQ_DATAW + SRC_IN_W == $bits(bus_in_if[0].req_data)),
        ("VX_rtu_bus_if req_data_t width changed: fix REQ_DATAW"))
    `STATIC_ASSERT((WIN_DATAW == $bits(bus_in_if[0].win_data)),
        ("VX_rtu_bus_if win_data_t width changed: fix WIN_DATAW"))
    `STATIC_ASSERT((SEL_W == $bits(bus_out_if[0].req_data.src)),
        ("an RTU-core-side bus must be instanced SRC_WIDTH = SEL_W"))


    for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_group

        // ── arm : the group's cores -> this RTU ────────────────────────
        wire [CORES_PER_RTU-1:0]                 arm_valid_in;
        wire [CORES_PER_RTU-1:0][ARM_DATAW-1:0]  arm_data_in;
        wire [CORES_PER_RTU-1:0]                 arm_ready_in;

        for (genvar j = 0; j < CORES_PER_RTU; ++j) begin : g_arm_in
            assign arm_valid_in[j] = bus_in_if[o * CORES_PER_RTU + j].arm_valid;
            `UNUSED_VAR (bus_in_if[o * CORES_PER_RTU + j].arm_data.src)
            assign arm_data_in[j]  = {bus_in_if[o * CORES_PER_RTU + j].arm_data.tag,
                                      bus_in_if[o * CORES_PER_RTU + j].arm_data.wid,
                                      bus_in_if[o * CORES_PER_RTU + j].arm_data.mask,
                                      bus_in_if[o * CORES_PER_RTU + j].arm_data.scene_base,
                                      bus_in_if[o * CORES_PER_RTU + j].arm_data.flags,
                                      bus_in_if[o * CORES_PER_RTU + j].arm_data.cull_mask,
                                      bus_in_if[o * CORES_PER_RTU + j].arm_data.payload_ptr};
            assign bus_in_if[o * CORES_PER_RTU + j].arm_ready = arm_ready_in[j];
        end

        wire [0:0]                arm_valid_out;
        wire [0:0][ARM_DATAW-1:0] arm_data_out;
        wire [0:0][SEL_W-1:0]     arm_sel_out;
        wire [0:0]                arm_ready_out;

        VX_stream_arb #(
            .NUM_INPUTS  (CORES_PER_RTU),
            .NUM_OUTPUTS (1),
            .DATAW       (ARM_DATAW),
            .ARBITER     (ARBITER),
            .OUT_BUF     (OUT_BUF_ARM)
        ) arm_arb (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (arm_valid_in),
            .ready_in  (arm_ready_in),
            .data_in   (arm_data_in),
            .data_out  (arm_data_out),
            .sel_out   (arm_sel_out),
            .valid_out (arm_valid_out),
            .ready_out (arm_ready_out)
        );

        // Fold the winning core's index into the tag: every win beat the RTU
        // raises for this trace carries it back, and the win switch below
        // recovers it to pick the core.
        wire [TAG_WIDTH-1:0] arm_tag_out;
        VX_bits_insert #(
            .N   (TAG_WIDTH),
            .S   (LOG_NUM_REQS),
            .POS (TAG_SEL_IDX)
        ) arm_tag_insert (
            .data_in  (arm_tag_out),
            .ins_in   (arm_sel_out[0]),
            .data_out (bus_out_if[o].arm_data.tag)
        );

        assign bus_out_if[o].arm_valid = arm_valid_out[0];
        assign bus_out_if[o].arm_data.src = arm_sel_out[0];
        assign {arm_tag_out,
                bus_out_if[o].arm_data.wid,
                bus_out_if[o].arm_data.mask,
                bus_out_if[o].arm_data.scene_base,
                bus_out_if[o].arm_data.flags,
                bus_out_if[o].arm_data.cull_mask,
                bus_out_if[o].arm_data.payload_ptr} = arm_data_out[0];
        assign arm_ready_out[0] = bus_out_if[o].arm_ready;

        // ── req : the group's cores -> this RTU ────────────────────────
        // No tag: a req is never answered, and only the core whose trace this
        // RTU is running can raise one.
        wire [CORES_PER_RTU-1:0]                 req_valid_in;
        wire [CORES_PER_RTU-1:0][REQ_DATAW-1:0]  req_data_in;
        wire [CORES_PER_RTU-1:0]                 req_ready_in;

        for (genvar j = 0; j < CORES_PER_RTU; ++j) begin : g_req_in
            assign req_valid_in[j] = bus_in_if[o * CORES_PER_RTU + j].req_valid;
            `UNUSED_VAR (bus_in_if[o * CORES_PER_RTU + j].req_data.src)
            assign req_data_in[j]  = {bus_in_if[o * CORES_PER_RTU + j].req_data.kind,
                                      bus_in_if[o * CORES_PER_RTU + j].req_data.wid,
                                      bus_in_if[o * CORES_PER_RTU + j].req_data.data,
                                      bus_in_if[o * CORES_PER_RTU + j].req_data.cb_action};
            assign bus_in_if[o * CORES_PER_RTU + j].req_ready = req_ready_in[j];
        end

        wire [0:0]                req_valid_out;
        wire [0:0][REQ_DATAW-1:0] req_data_out;
        wire [0:0][SEL_W-1:0]     req_sel_out;
        wire [0:0]                req_ready_out;

        VX_stream_arb #(
            .NUM_INPUTS  (CORES_PER_RTU),
            .NUM_OUTPUTS (1),
            .DATAW       (REQ_DATAW),
            .ARBITER     (ARBITER),
            .OUT_BUF     (OUT_BUF_REQ)
        ) req_arb (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (req_valid_in),
            .ready_in  (req_ready_in),
            .data_in   (req_data_in),
            .data_out  (req_data_out),
            .sel_out   (req_sel_out),
            .valid_out (req_valid_out),
            .ready_out (req_ready_out)
        );

        assign bus_out_if[o].req_valid = req_valid_out[0];
        // The grant IS the beat's core id. Dropping it (as this arbiter used to)
        // aliases two cores' identically numbered warps onto one staging entry.
        assign bus_out_if[o].req_data.src = req_sel_out[0];
        assign {bus_out_if[o].req_data.kind,
                bus_out_if[o].req_data.wid,
                bus_out_if[o].req_data.data,
                bus_out_if[o].req_data.cb_action} = req_data_out[0];
        assign req_ready_out[0] = bus_out_if[o].req_ready;

        // ── win : this RTU -> the group's cores ────────────────────────
        wire [0:0]                win_valid_in;
        wire [0:0][WIN_DATAW-1:0] win_data_in;
        wire [0:0]                win_ready_in;
        wire [0:0][SEL_W-1:0]     win_sel_in;

        wire [TAG_WIDTH-1:0] win_tag_out;
        VX_bits_remove #(
            .N   (TAG_WIDTH + LOG_NUM_REQS),
            .S   (LOG_NUM_REQS),
            .POS (TAG_SEL_IDX)
        ) win_tag_remove (
            .data_in  (bus_out_if[o].win_data.tag),
            `UNUSED_PIN (sel_out),
            .data_out (win_tag_out)
        );

        assign win_valid_in[0] = bus_out_if[o].win_valid;
        assign win_data_in[0]  = {win_tag_out,
                                  bus_out_if[o].win_data.is_cand,
                                  bus_out_if[o].win_data.wid,
                                  bus_out_if[o].win_data.slot,
                                  bus_out_if[o].win_data.mask,
                                  bus_out_if[o].win_data.data};
        assign bus_out_if[o].win_ready = win_ready_in[0];

        if (CORES_PER_RTU > 1) begin : g_win_sel
            assign win_sel_in[0] = bus_out_if[o].win_data.tag[TAG_SEL_IDX +: LOG_NUM_REQS];
        end else begin : g_win_sel_0
            assign win_sel_in[0] = '0;
        end

        wire [CORES_PER_RTU-1:0]                win_valid_out;
        wire [CORES_PER_RTU-1:0][WIN_DATAW-1:0] win_data_out;
        wire [CORES_PER_RTU-1:0]                win_ready_out;

        VX_stream_switch #(
            .NUM_INPUTS  (1),
            .NUM_OUTPUTS (CORES_PER_RTU),
            .DATAW       (WIN_DATAW),
            .OUT_BUF     (OUT_BUF_WIN)
        ) win_switch (
            .clk       (clk),
            .reset     (reset),
            .sel_in    (win_sel_in),
            .valid_in  (win_valid_in),
            .ready_in  (win_ready_in),
            .data_in   (win_data_in),
            .data_out  (win_data_out),
            .valid_out (win_valid_out),
            .ready_out (win_ready_out)
        );

        for (genvar j = 0; j < CORES_PER_RTU; ++j) begin : g_win_out
            assign bus_in_if[o * CORES_PER_RTU + j].win_valid = win_valid_out[j];
            assign {bus_in_if[o * CORES_PER_RTU + j].win_data.tag,
                    bus_in_if[o * CORES_PER_RTU + j].win_data.is_cand,
                    bus_in_if[o * CORES_PER_RTU + j].win_data.wid,
                    bus_in_if[o * CORES_PER_RTU + j].win_data.slot,
                    bus_in_if[o * CORES_PER_RTU + j].win_data.mask,
                    bus_in_if[o * CORES_PER_RTU + j].win_data.data} = win_data_out[j];
            assign win_ready_out[j] = bus_in_if[o * CORES_PER_RTU + j].win_ready;
        end

    end

endmodule
