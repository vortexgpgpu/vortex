// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

`ifdef VX_CFG_EXT_DXA_GROUP_ENABLE

// Group-completion transport: fair-arbitrates typed completion events from
// all worker-side sources into one stream, then routes each event to its
// issuing core's tracker by the core-id header (consumed here — trackers
// receive the bare dxa_group_completion_t payload). Elastic buffering on
// both stages; when a destination stalls, backpressure propagates losslessly
// to every source — events are never dropped, duplicated, or overwritten
// (payloads hold stable under valid && !ready). Reset empties the pipeline.
// Per-source ordering to any given core is preserved (single arbitration
// point, in-order switch).
module VX_dxa_group_completion_hub import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter NUM_SRCS  = 1,
    parameter NUM_CORES = 1,
    parameter CORE_W    = `UP(`CLOG2(NUM_CORES)),
    parameter OUT_BUF   = 2
) (
    input wire clk,
    input wire reset,

    input  wire [NUM_SRCS-1:0]                             src_valid,
    input  wire [NUM_SRCS-1:0][CORE_W-1:0]                 src_core,
    input  wire [NUM_SRCS-1:0][DXA_GROUP_COMPL_W-1:0]      src_data,
    output wire [NUM_SRCS-1:0]                             src_ready,

    output wire [NUM_CORES-1:0]                            dst_valid,
    output wire [NUM_CORES-1:0][DXA_GROUP_COMPL_W-1:0]     dst_data,
    input  wire [NUM_CORES-1:0]                            dst_ready
);
    // stage 1: arbitrate all sources into one event stream (the core-id
    // routing header rides along for the switch)
    wire                                   arb_valid;
    wire [CORE_W+DXA_GROUP_COMPL_W-1:0]    arb_data;
    wire                                   arb_ready;

    wire [NUM_SRCS-1:0][CORE_W+DXA_GROUP_COMPL_W-1:0] src_bundle;
    for (genvar i = 0; i < NUM_SRCS; ++i) begin : g_bundle
        assign src_bundle[i] = {src_core[i], src_data[i]};
    end

    VX_stream_arb #(
        .NUM_INPUTS  (NUM_SRCS),
        .NUM_OUTPUTS (1),
        .DATAW       (CORE_W + DXA_GROUP_COMPL_W),
        .ARBITER     ("R"),
        .OUT_BUF     (OUT_BUF)
    ) src_arb (
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

    // stage 2: route to the owning core, dropping the routing header
    wire [CORE_W-1:0]              arb_core    = arb_data[DXA_GROUP_COMPL_W +: CORE_W];
    wire [DXA_GROUP_COMPL_W-1:0]   arb_payload = arb_data[0 +: DXA_GROUP_COMPL_W];

    VX_stream_switch #(
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (NUM_CORES),
        .DATAW       (DXA_GROUP_COMPL_W),
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

endmodule

`endif // VX_CFG_EXT_DXA_GROUP_ENABLE
