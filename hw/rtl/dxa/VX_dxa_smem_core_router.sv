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

// DXA SMEM core router: routes bank-native writes from workers to output ports.
// Write-only (no response path). Uses output_sel for routing selection.
// Optional CORE_ID_WIDTH sideband carried through the xbar payload for
// downstream socket-level arbitration.

`include "VX_define.vh"

module VX_dxa_smem_core_router import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter NUM_INPUTS     = 1,
    parameter NUM_OUTPUTS    = 1,
    parameter CORE_ID_WIDTH  = 0,   // sideband width (0 = no sideband)
    parameter ENABLE         = 1
) (
    input wire clk,
    input wire reset,

    // From workers: bank_wr + routing sel + optional local_core_id sideband
    VX_dxa_bank_wr_if.slave   worker_bank_wr_if [NUM_INPUTS],
    input wire [NUM_INPUTS-1:0][`UP(`CLOG2(NUM_OUTPUTS))-1:0] worker_output_sel,
    input wire [NUM_INPUTS-1:0][`UP(CORE_ID_WIDTH)-1:0] worker_local_core_id,

    // To output ports
    VX_dxa_bank_wr_if.master  out_bank_wr_if [NUM_OUTPUTS],
    output wire [NUM_OUTPUTS-1:0][`UP(CORE_ID_WIDTH)-1:0] out_local_core_id
);

    localparam NUM_BANKS       = `LMEM_NUM_BANKS;
    localparam BANK_ADDR_WIDTH = DXA_SMEM_BANK_ADDR_WIDTH;
    localparam WORD_SIZE       = `XLEN / 8;
    localparam WORD_WIDTH      = WORD_SIZE * 8;
    localparam TAG_WIDTH       = DXA_BANK_WR_TAG_WIDTH;
    localparam SEL_WIDTH       = `UP(`CLOG2(NUM_OUTPUTS));
    localparam CORE_ID_W       = `UP(CORE_ID_WIDTH);

    // Flatten payload: {[core_id], tag, per_bank(byteen, data, addr), per_bank(valid)}
    localparam BASE_PAYLOAD_W = TAG_WIDTH
                              + NUM_BANKS * (WORD_SIZE + WORD_WIDTH + BANK_ADDR_WIDTH)
                              + NUM_BANKS;
    localparam PAYLOAD_W = BASE_PAYLOAD_W + ((CORE_ID_WIDTH > 0) ? CORE_ID_WIDTH : 0);

    if (ENABLE) begin : g_route

        wire [NUM_INPUTS-1:0]                  req_valid_in;
        wire [NUM_INPUTS-1:0][PAYLOAD_W-1:0]   req_data_in;
        wire [NUM_INPUTS-1:0][SEL_WIDTH-1:0]   req_sel_in;
        wire [NUM_INPUTS-1:0]                  req_ready_in;

        wire [NUM_OUTPUTS-1:0]                 req_valid_out;
        wire [NUM_OUTPUTS-1:0][PAYLOAD_W-1:0]  req_data_out;
        wire [NUM_OUTPUTS-1:0]                 req_ready_out;

        // Flatten worker bank_wr into payload vectors
        for (genvar w = 0; w < NUM_INPUTS; ++w) begin : g_flatten_in
            assign req_valid_in[w] = |worker_bank_wr_if[w].wr_valid;
            if (CORE_ID_WIDTH > 0) begin : g_with_sideband
                assign req_data_in[w] = {
                    worker_local_core_id[w][CORE_ID_WIDTH-1:0],
                    worker_bank_wr_if[w].wr_tag,
                    worker_bank_wr_if[w].wr_byteen,
                    worker_bank_wr_if[w].wr_data,
                    worker_bank_wr_if[w].wr_addr,
                    worker_bank_wr_if[w].wr_valid
                };
            end else begin : g_no_sideband
                assign req_data_in[w] = {
                    worker_bank_wr_if[w].wr_tag,
                    worker_bank_wr_if[w].wr_byteen,
                    worker_bank_wr_if[w].wr_data,
                    worker_bank_wr_if[w].wr_addr,
                    worker_bank_wr_if[w].wr_valid
                };
                `UNUSED_VAR (worker_local_core_id[w])
            end
            assign req_sel_in[w] = worker_output_sel[w];
            assign worker_bank_wr_if[w].wr_ready = req_ready_in[w];
        end

        // Use Vortex stream xbar for routing
        /* verilator lint_off UNUSEDSIGNAL */
        wire [NUM_OUTPUTS-1:0][`UP(`CLOG2(NUM_INPUTS))-1:0] req_sel_out;
        /* verilator lint_on UNUSEDSIGNAL */

        VX_stream_xbar #(
            .NUM_INPUTS  (NUM_INPUTS),
            .NUM_OUTPUTS (NUM_OUTPUTS),
            .DATAW       (PAYLOAD_W),
            .ARBITER     ("R"),
            .OUT_BUF     ((NUM_INPUTS != NUM_OUTPUTS) ? 2 : 0)
        ) smem_core_xbar (
            .clk       (clk),
            .reset     (reset),
            `UNUSED_PIN (collisions),
            .valid_in  (req_valid_in),
            .data_in   (req_data_in),
            .sel_in    (req_sel_in),
            .ready_in  (req_ready_in),
            .valid_out (req_valid_out),
            .data_out  (req_data_out),
            .sel_out   (req_sel_out),
            .ready_out (req_ready_out)
        );

        // Unflatten output back to VX_dxa_bank_wr_if + optional sideband
        for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_unflatten_out
            wire [NUM_BANKS-1:0]                       out_wr_valid;
            wire [NUM_BANKS-1:0][BANK_ADDR_WIDTH-1:0]  out_wr_addr;
            wire [NUM_BANKS-1:0][WORD_WIDTH-1:0]       out_wr_data;
            wire [NUM_BANKS-1:0][WORD_SIZE-1:0]        out_wr_byteen;
            wire [TAG_WIDTH-1:0]                       out_wr_tag;

            if (CORE_ID_WIDTH > 0) begin : g_sideband_out
                wire [CORE_ID_WIDTH-1:0] out_core_id;
                assign {out_core_id, out_wr_tag, out_wr_byteen, out_wr_data, out_wr_addr, out_wr_valid}
                    = req_data_out[o];
                assign out_local_core_id[o] = CORE_ID_W'(out_core_id);
            end else begin : g_no_sideband_out
                assign {out_wr_tag, out_wr_byteen, out_wr_data, out_wr_addr, out_wr_valid}
                    = req_data_out[o];
                assign out_local_core_id[o] = '0;
            end

            // Gate per-bank valid with the xbar output valid
            for (genvar b = 0; b < NUM_BANKS; ++b) begin : g_bank_gate
                assign out_bank_wr_if[o].wr_valid[b]  = req_valid_out[o] && out_wr_valid[b];
                assign out_bank_wr_if[o].wr_addr[b]   = out_wr_addr[b];
                assign out_bank_wr_if[o].wr_data[b]   = out_wr_data[b];
                assign out_bank_wr_if[o].wr_byteen[b] = out_wr_byteen[b];
            end
            assign out_bank_wr_if[o].wr_tag = out_wr_tag;
            assign req_ready_out[o] = out_bank_wr_if[o].wr_ready;
        end

    end else begin : g_route_off
        for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_off_out
            assign out_bank_wr_if[o].wr_valid  = '0;
            assign out_bank_wr_if[o].wr_addr   = '0;
            assign out_bank_wr_if[o].wr_data   = '0;
            assign out_bank_wr_if[o].wr_byteen = '0;
            assign out_bank_wr_if[o].wr_tag    = '0;
            assign out_local_core_id[o]        = '0;
        end
        for (genvar w = 0; w < NUM_INPUTS; ++w) begin : g_off_in
            assign worker_bank_wr_if[w].wr_ready = 1'b1;
            `UNUSED_VAR (worker_bank_wr_if[w].wr_valid)
            `UNUSED_VAR (worker_bank_wr_if[w].wr_addr)
            `UNUSED_VAR (worker_bank_wr_if[w].wr_data)
            `UNUSED_VAR (worker_bank_wr_if[w].wr_byteen)
            `UNUSED_VAR (worker_bank_wr_if[w].wr_tag)
        end
        `UNUSED_VAR (worker_output_sel)
        `UNUSED_VAR (worker_local_core_id)
    end

endmodule
