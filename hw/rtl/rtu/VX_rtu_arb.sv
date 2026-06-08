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

// VX_rtu_arb — arbitrates RTU trace requests from NUM_INPUTS senders onto
// NUM_OUTPUTS cluster cores and routes the per-lane results back. The
// arbiter select index is folded into the tag so the response stream can be
// switched back to its originating input. Mirrors VX_tex_arb.

`include "VX_define.vh"

module VX_rtu_arb import VX_gpu_pkg::*, VX_rtu_pkg::*; #(
    parameter NUM_INPUTS  = 1,
    parameter NUM_OUTPUTS = 1,
    parameter NUM_LANES   = 1,
    parameter TAG_WIDTH   = 1,
    parameter TAG_SEL_IDX = 0,
    parameter OUT_BUF_REQ = 0,
    parameter OUT_BUF_RSP = 0,
    parameter `STRING ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,

    VX_rtu_bus_if.slave     bus_in_if [NUM_INPUTS],
    VX_rtu_bus_if.master    bus_out_if [NUM_OUTPUTS]
);
    localparam LOG_NUM_REQS = `ARB_SEL_BITS(NUM_INPUTS, NUM_OUTPUTS);
    localparam RAY_BITS     = $bits(rtu_ray_t);
    // Payload also carries the Phase-2 callback fields: req {kind, cb_action};
    // rsp {kind, cb_active_mask, cb_type, cb_sbt_idx}. Packed alongside the
    // base fields below — order must match between pack and unpack.
    localparam REQ_DATAW    = TAG_WIDTH + 1 + NUM_LANES * (1 + RAY_BITS)
                            + NUM_LANES * RTU_CB_ACTION_BITS;
    localparam RSP_DATAW    = TAG_WIDTH + 1 + NUM_LANES * (6 * 32)
                            + NUM_LANES * (1 + RTU_CB_TYPE_BITS + RTU_CB_SBT_BITS);

    // ── request path ──────────────────────────────────────────────────
    wire [NUM_INPUTS-1:0]                 req_valid_in;
    wire [NUM_INPUTS-1:0][REQ_DATAW-1:0]  req_data_in;
    wire [NUM_INPUTS-1:0]                 req_ready_in;

    wire [NUM_OUTPUTS-1:0]                req_valid_out;
    wire [NUM_OUTPUTS-1:0][REQ_DATAW-1:0] req_data_out;
    wire [NUM_OUTPUTS-1:0][`UP(LOG_NUM_REQS)-1:0] req_sel_out;
    wire [NUM_OUTPUTS-1:0]                req_ready_out;

    for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_req_data_in
        assign req_valid_in[i] = bus_in_if[i].req_valid;
        assign req_data_in[i]  = {bus_in_if[i].req_data.tag,
                                  bus_in_if[i].req_data.kind,
                                  bus_in_if[i].req_data.mask,
                                  bus_in_if[i].req_data.rays,
                                  bus_in_if[i].req_data.cb_action};
        assign bus_in_if[i].req_ready = req_ready_in[i];
    end

    VX_stream_arb #(
        .NUM_INPUTS  (NUM_INPUTS),
        .NUM_OUTPUTS (NUM_OUTPUTS),
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

    for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin : g_bus_out_if
        wire [TAG_WIDTH-1:0] req_tag_out;
        VX_bits_insert #(
            .N   (TAG_WIDTH),
            .S   (LOG_NUM_REQS),
            .POS (TAG_SEL_IDX)
        ) bits_insert (
            .data_in  (req_tag_out),
            .ins_in   (req_sel_out[i]),
            .data_out (bus_out_if[i].req_data.tag)
        );
        assign bus_out_if[i].req_valid = req_valid_out[i];
        assign {req_tag_out,
                bus_out_if[i].req_data.kind,
                bus_out_if[i].req_data.mask,
                bus_out_if[i].req_data.rays,
                bus_out_if[i].req_data.cb_action} = req_data_out[i];
        assign req_ready_out[i] = bus_out_if[i].req_ready;
    end

    // ── response path ─────────────────────────────────────────────────
    wire [NUM_INPUTS-1:0]                 rsp_valid_out;
    wire [NUM_INPUTS-1:0][RSP_DATAW-1:0]  rsp_data_out;
    wire [NUM_INPUTS-1:0]                 rsp_ready_out;

    wire [NUM_OUTPUTS-1:0]                rsp_valid_in;
    wire [NUM_OUTPUTS-1:0][RSP_DATAW-1:0] rsp_data_in;
    wire [NUM_OUTPUTS-1:0]                rsp_ready_in;

    if (NUM_INPUTS > NUM_OUTPUTS) begin : g_rsp_switch

        wire [NUM_OUTPUTS-1:0][LOG_NUM_REQS-1:0] rsp_sel_in;

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin : g_rsp_data_in
            wire [TAG_WIDTH-1:0] rsp_tag_out;
            VX_bits_remove #(
                .N   (TAG_WIDTH + LOG_NUM_REQS),
                .S   (LOG_NUM_REQS),
                .POS (TAG_SEL_IDX)
            ) bits_remove (
                .data_in  (bus_out_if[i].rsp_data.tag),
                `UNUSED_PIN (sel_out),
                .data_out (rsp_tag_out)
            );
            assign rsp_valid_in[i] = bus_out_if[i].rsp_valid;
            assign rsp_data_in[i]  = {rsp_tag_out,
                                      bus_out_if[i].rsp_data.kind,
                                      bus_out_if[i].rsp_data.status,
                                      bus_out_if[i].rsp_data.hit_t,
                                      bus_out_if[i].rsp_data.hit_u,
                                      bus_out_if[i].rsp_data.hit_v,
                                      bus_out_if[i].rsp_data.hit_prim_id,
                                      bus_out_if[i].rsp_data.hit_geometry,
                                      bus_out_if[i].rsp_data.cb_active_mask,
                                      bus_out_if[i].rsp_data.cb_type,
                                      bus_out_if[i].rsp_data.cb_sbt_idx};
            assign bus_out_if[i].rsp_ready = rsp_ready_in[i];

            if (NUM_INPUTS > 1) begin : g_rsp_sel_in
                assign rsp_sel_in[i] = bus_out_if[i].rsp_data.tag[TAG_SEL_IDX +: LOG_NUM_REQS];
            end else begin : g_rsp_sel_in_0
                assign rsp_sel_in[i] = '0;
            end
        end

        VX_stream_switch #(
            .NUM_INPUTS  (NUM_OUTPUTS),
            .NUM_OUTPUTS (NUM_INPUTS),
            .DATAW       (RSP_DATAW),
            .OUT_BUF     (OUT_BUF_RSP)
        ) rsp_switch (
            .clk       (clk),
            .reset     (reset),
            .sel_in    (rsp_sel_in),
            .valid_in  (rsp_valid_in),
            .ready_in  (rsp_ready_in),
            .data_in   (rsp_data_in),
            .data_out  (rsp_data_out),
            .valid_out (rsp_valid_out),
            .ready_out (rsp_ready_out)
        );

    end else begin : g_rsp_arb

        for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin : g_rsp_data_in
            assign rsp_valid_in[i] = bus_out_if[i].rsp_valid;
            assign rsp_data_in[i]  = {bus_out_if[i].rsp_data.tag,
                                      bus_out_if[i].rsp_data.kind,
                                      bus_out_if[i].rsp_data.status,
                                      bus_out_if[i].rsp_data.hit_t,
                                      bus_out_if[i].rsp_data.hit_u,
                                      bus_out_if[i].rsp_data.hit_v,
                                      bus_out_if[i].rsp_data.hit_prim_id,
                                      bus_out_if[i].rsp_data.hit_geometry,
                                      bus_out_if[i].rsp_data.cb_active_mask,
                                      bus_out_if[i].rsp_data.cb_type,
                                      bus_out_if[i].rsp_data.cb_sbt_idx};
            assign bus_out_if[i].rsp_ready = rsp_ready_in[i];
        end

        VX_stream_arb #(
            .NUM_INPUTS  (NUM_OUTPUTS),
            .NUM_OUTPUTS (NUM_INPUTS),
            .DATAW       (RSP_DATAW),
            .ARBITER     (ARBITER),
            .OUT_BUF     (OUT_BUF_RSP)
        ) rsp_arb (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (rsp_valid_in),
            .ready_in  (rsp_ready_in),
            .data_in   (rsp_data_in),
            .data_out  (rsp_data_out),
            .valid_out (rsp_valid_out),
            .ready_out (rsp_ready_out),
            `UNUSED_PIN (sel_out)
        );

    end

    for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_bus_in_if
        assign bus_in_if[i].rsp_valid = rsp_valid_out[i];
        assign {bus_in_if[i].rsp_data.tag,
                bus_in_if[i].rsp_data.kind,
                bus_in_if[i].rsp_data.status,
                bus_in_if[i].rsp_data.hit_t,
                bus_in_if[i].rsp_data.hit_u,
                bus_in_if[i].rsp_data.hit_v,
                bus_in_if[i].rsp_data.hit_prim_id,
                bus_in_if[i].rsp_data.hit_geometry,
                bus_in_if[i].rsp_data.cb_active_mask,
                bus_in_if[i].rsp_data.cb_type,
                bus_in_if[i].rsp_data.cb_sbt_idx} = rsp_data_out[i];
        assign rsp_ready_out[i] = bus_in_if[i].rsp_ready;
    end

endmodule
