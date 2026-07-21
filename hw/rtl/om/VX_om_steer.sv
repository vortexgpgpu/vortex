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

// VX_om_steer — peels OM fragment-export writes off a socket->L2 trunk input.
//
// One instance per physical trunk input, placed at the combinational socket->L2
// join: the last point at which a request has left the socket and has not yet
// touched anything L2-owned. That placement is the deadlock argument:
// a full OM ingress back-pressures this switch, which blocks that trunk input
// BEFORE L2, holding no L2 resource; the OM drains through the ocache, which
// owns a dedicated, disjoint L2 input port. The two never contend, so there is
// no cycle.
//
// The select is the is_addr_om attr bit alone. No cache level decodes it -- the
// aperture store is presented to the caches as an ordinary IO store (they carry
// attr through verbatim), so this bit arrives untouched. See VX_lsu_slice.
//
// Buffering: only the OM leg is registered. The L2 pass-through stays
// combinational -- it is not a new driver (the socket already registers its
// outgoing bus), and a skid here would sit on the critical L1->L2 path and
// double the switch's cost for nothing.

`include "VX_define.vh"

module VX_om_steer import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter DATA_SIZE  = 1,
    parameter TAG_WIDTH  = 1,
    parameter OUT_BUF    = 0
) (
    input wire  clk,
    input wire  reset,

    VX_mem_bus_if.slave  bus_in_if,     // from the socket
    VX_mem_bus_if.master l2_out_if,     // everything else -> L2
    VX_mem_bus_if.master om_out_if      // aperture writes -> OM ingress (posted)
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(DATA_SIZE);
    localparam REQ_DATAW  = 1 + ADDR_WIDTH + DATA_SIZE * 8 + DATA_SIZE
                          + `UP(MEM_ATTR_WIDTH) + TAG_WIDTH;

    wire is_om = bus_in_if.req_data.attr[MEM_ATTR_OM_OFFS];

    // ── request: demux by is_addr_om ───────────────────────────────────────
    wire l2_ready;
    wire om_ready;

    assign bus_in_if.req_ready = is_om ? om_ready : l2_ready;

    // Both legs go through an OUT_BUF elastic buffer so the socket->L2 trunk and
    // the OM aperture path each present a fully-registered master interface.
    wire [REQ_DATAW-1:0] l2_req_data_in = bus_in_if.req_data;
    wire [REQ_DATAW-1:0] l2_req_data_out;

    VX_elastic_buffer #(
        .DATAW   (REQ_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF))
    ) l2_req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (bus_in_if.req_valid && ~is_om),
        .ready_in  (l2_ready),
        .data_in   (l2_req_data_in),
        .data_out  (l2_req_data_out),
        .valid_out (l2_out_if.req_valid),
        .ready_out (l2_out_if.req_ready)
    );
    assign l2_out_if.req_data = l2_req_data_out;

    wire [REQ_DATAW-1:0] om_req_data_in = bus_in_if.req_data;
    wire [REQ_DATAW-1:0] om_req_data_out;

    VX_elastic_buffer #(
        .DATAW   (REQ_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF))
    ) om_req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (bus_in_if.req_valid && is_om),
        .ready_in  (om_ready),
        .data_in   (om_req_data_in),
        .data_out  (om_req_data_out),
        .valid_out (om_out_if.req_valid),
        .ready_out (om_out_if.req_ready)
    );
    assign om_out_if.req_data = om_req_data_out;

    // ── response: only the L2 leg answers ─────────────────────────────────
    // Aperture writes are posted; the OM never responds. Reading the aperture is
    // illegal -- it is a virtual range with nothing behind it, so a read would
    // hang waiting for a response that cannot come. Catch it loudly.
    assign bus_in_if.rsp_valid = l2_out_if.rsp_valid;
    assign bus_in_if.rsp_data  = l2_out_if.rsp_data;
    assign l2_out_if.rsp_ready = bus_in_if.rsp_ready;

    assign om_out_if.rsp_ready = 1'b0;
    `UNUSED_VAR (om_out_if.rsp_valid)
    `UNUSED_VAR (om_out_if.rsp_data)

    `RUNTIME_ASSERT(~(bus_in_if.req_valid && is_om && ~bus_in_if.req_data.rw),
        ("%t: *** %s: read of the OM aperture — it is a virtual range, nothing answers", $time, INSTANCE_ID))

endmodule
