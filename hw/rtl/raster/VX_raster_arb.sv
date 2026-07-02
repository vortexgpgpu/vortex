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

`include "VX_raster_define.vh"

// VX_raster_arb — cluster / socket raster bus arbiter (Fragment Dispatch v3)
//
// Routes covered-quad waves (raster_stamp_t[NUM_LANES]) from N producers to M
// consumers. The raster bus is a pure data stream: frame drain is signaled
// out-of-band via VX_raster_core.busy, so this arbiter carries no `done`
// token and needs none of the v2 sticky-done / frame-rearm / consumer-served
// machinery.
//
//   N >= M : VX_stream_arb (fan-in N>M, or 1:1 N==M).
//   N <  M : collapse N->1, then deterministic screen-bin -> owner-core demux.
//            Owner routing (bin % M) keeps every pixel on one core so per-pixel
//            raster/OM order is preserved by construction (blend-safe); a wave
//            to a busy owner back-pressures, it is never dropped.

module VX_raster_arb import VX_raster_pkg::*; #(
    parameter NUM_INPUTS     = 1,
    parameter NUM_OUTPUTS    = 1,
    parameter NUM_LANES      = 1,
    parameter OUT_BUF        = 0,
    parameter `STRING ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,

    VX_raster_bus_if.slave  bus_in_if  [NUM_INPUTS],
    VX_raster_bus_if.master bus_out_if [NUM_OUTPUTS]
);
    localparam REQ_DATAW   = NUM_LANES * $bits(raster_stamp_t);
    localparam IS_FANOUT   = (NUM_INPUTS < NUM_OUTPUTS);
    localparam LOG_OUTPUTS = `LOG2UP(NUM_OUTPUTS);

    wire [NUM_INPUTS-1:0]                in_valid;
    wire [NUM_INPUTS-1:0][REQ_DATAW-1:0] in_data;
    wire [NUM_INPUTS-1:0]                in_ready;
    for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_in
        assign in_valid[i] = bus_in_if[i].req_valid;
        assign in_data[i]  = bus_in_if[i].req_data.stamps;
        assign bus_in_if[i].req_ready = in_ready[i];
    end

    wire [NUM_OUTPUTS-1:0]                out_valid;
    wire [NUM_OUTPUTS-1:0][REQ_DATAW-1:0] out_data;
    wire [NUM_OUTPUTS-1:0]                out_ready;
    for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_out
        assign bus_out_if[o].req_valid       = out_valid[o];
        assign bus_out_if[o].req_data.stamps = out_data[o];
        assign out_ready[o]                  = bus_out_if[o].req_ready;
    end

    if (!IS_FANOUT) begin : g_fanin_or_pass
        // N >= M: VX_stream_arb handles fan-in (N>M) and 1:1 pairing (N==M).
        VX_stream_arb #(
            .NUM_INPUTS (NUM_INPUTS),
            .NUM_OUTPUTS(NUM_OUTPUTS),
            .DATAW      (REQ_DATAW),
            .ARBITER    (ARBITER),
            .OUT_BUF    (OUT_BUF)
        ) req_arb (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (in_valid),
            .ready_in   (in_ready),
            .data_in    (in_data),
            .data_out   (out_data),
            .valid_out  (out_valid),
            .ready_out  (out_ready),
            `UNUSED_PIN (sel_out)
        );
    end else begin : g_fanout
        // N < M: collapse N->1 then owner-route 1->M by screen bin.
        wire                 merged_valid;
        wire [REQ_DATAW-1:0] merged_data;
        wire                 merged_ready;

        VX_stream_arb #(
            .NUM_INPUTS (NUM_INPUTS),
            .NUM_OUTPUTS(1),
            .DATAW      (REQ_DATAW),
            .ARBITER    (ARBITER),
            .OUT_BUF    (0)
        ) collapse_arb (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (in_valid),
            .ready_in   (in_ready),
            .data_in    (in_data),
            .data_out   (merged_data),
            .valid_out  (merged_valid),
            .ready_out  (merged_ready),
            `UNUSED_PIN (sel_out)
        );

        // Owner core = screen bin of the wave's first quad, modulo M. The wave's
        // lanes are all from one block (one bin), so lane 0's position selects
        // the bin. quad pos -> bin: >> (BIN_LOGSIZE-1) (quad = 2 px).
        raster_stamp_t [NUM_LANES-1:0] m_stamps;
        assign m_stamps = merged_data;
        localparam BIN_Q = `VX_CFG_RASTER_BIN_LOGSIZE - 1;
        wire [31:0] bin_lin = (32'(m_stamps[0].pos_x) >> BIN_Q)
                            + (32'(m_stamps[0].pos_y) >> BIN_Q);
        wire [LOG_OUTPUTS-1:0] owner = LOG_OUTPUTS'(bin_lin % NUM_OUTPUTS);

        wire [NUM_OUTPUTS-1:0] obuf_ready;
        for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_obuf
            wire sel = (owner == LOG_OUTPUTS'(o));
            VX_elastic_buffer #(
                .DATAW  (REQ_DATAW),
                .SIZE   (`TO_OUT_BUF_SIZE(OUT_BUF)),
                .OUT_REG(`TO_OUT_BUF_REG(OUT_BUF)),
                .LUTRAM (`TO_OUT_BUF_LUTRAM(OUT_BUF))
            ) out_buf (
                .clk      (clk),
                .reset    (reset),
                .valid_in (merged_valid && sel),
                .data_in  (merged_data),
                .ready_in (obuf_ready[o]),
                .valid_out(out_valid[o]),
                .data_out (out_data[o]),
                .ready_out(out_ready[o])
            );
        end

        assign merged_ready = obuf_ready[owner];
    end

endmodule
