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

`ifdef VX_CFG_EXT_DXA_GROUP_ENABLE

// Lossless worker-to-core completion transport. The core-id route header is
// consumed here; the tracker sees only {wid, group_id}.
module VX_dxa_group_completion_hub import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter NUM_SRCS  = 1,
    parameter NUM_CORES = 1,
    parameter CORE_W    = `UP(`CLOG2(NUM_CORES)),
    parameter OUT_BUF   = 2
) (
    input wire clk,
    input wire reset,
    VX_dxa_group_completion_if.slave  src_if [NUM_SRCS],
    VX_dxa_group_completion_if.master dst_if [NUM_CORES]
);
    localparam PAYLOAD_W = $bits(dxa_group_completion_t);

    wire arb_valid;
    wire [CORE_W+PAYLOAD_W-1:0] arb_data;
    wire arb_ready;
    wire [NUM_SRCS-1:0] src_valid;
    wire [NUM_SRCS-1:0][CORE_W+PAYLOAD_W-1:0] src_bundle;
    wire [NUM_SRCS-1:0] src_ready;

    for (genvar i = 0; i < NUM_SRCS; ++i) begin : g_src
        assign src_valid[i] = src_if[i].valid;
        assign src_bundle[i] = {src_if[i].core_id[CORE_W-1:0], src_if[i].data};
        assign src_if[i].ready = src_ready[i];
    end

    VX_stream_arb #(
        .NUM_INPUTS  (NUM_SRCS),
        .NUM_OUTPUTS (1),
        .DATAW       (CORE_W + PAYLOAD_W),
        .ARBITER     ("R"),
        .OUT_BUF     (OUT_BUF)
    ) source_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (src_valid),
        .data_in   (src_bundle),
        .ready_in  (src_ready),
        .valid_out (arb_valid),
        .data_out  (arb_data),
        .ready_out (arb_ready),
        `UNUSED_PIN (sel_out)
    );

    wire [CORE_W-1:0] arb_core = arb_data[PAYLOAD_W +: CORE_W];
    wire [PAYLOAD_W-1:0] arb_payload = arb_data[0 +: PAYLOAD_W];
    wire [NUM_CORES-1:0] dst_valid;
    wire [NUM_CORES-1:0][PAYLOAD_W-1:0] dst_data;
    wire [NUM_CORES-1:0] dst_ready;

    VX_stream_switch #(
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (NUM_CORES),
        .DATAW       (PAYLOAD_W),
        .OUT_BUF     (OUT_BUF)
    ) core_switch (
        .clk       (clk),
        .reset     (reset),
        .sel_in    (arb_core),
        .valid_in  (arb_valid),
        .data_in   (arb_payload),
        .ready_in  (arb_ready),
        .valid_out (dst_valid),
        .data_out  (dst_data),
        .ready_out (dst_ready)
    );

    for (genvar c = 0; c < NUM_CORES; ++c) begin : g_dst
        assign dst_if[c].valid = dst_valid[c];
        assign dst_if[c].core_id = '0;
        assign dst_if[c].data = dxa_group_completion_t'(dst_data[c]);
        assign dst_ready[c] = dst_if[c].ready;
    end

endmodule

`endif
