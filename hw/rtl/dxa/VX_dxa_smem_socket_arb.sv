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

// Socket-level DXA bank_wr arbiter: routes M input bank_wr ports to N output
// bank_wr ports using local_core_id sideband for target selection.
// Used when DXA_SMEM_PORTS_PER_SOCKET < SOCKET_SIZE.

`include "VX_define.vh"

module VX_dxa_smem_socket_arb import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter NUM_INPUTS  = 1,
    parameter NUM_OUTPUTS = 1
) (
    input wire clk,
    input wire reset,

    VX_dxa_bank_wr_if.slave  bank_wr_in [NUM_INPUTS],
    input wire [NUM_INPUTS-1:0][DXA_SMEM_LOCAL_CORE_W-1:0] local_core_id_in,

    VX_dxa_bank_wr_if.master bank_wr_out [NUM_OUTPUTS],
    output wire [NUM_OUTPUTS-1:0][DXA_SMEM_LOCAL_CORE_W-1:0] local_core_id_out
);

    localparam NUM_BANKS       = `LMEM_NUM_BANKS;
    localparam BANK_ADDR_WIDTH = DXA_SMEM_BANK_ADDR_WIDTH;
    localparam WORD_SIZE       = `XLEN / 8;
    localparam WORD_WIDTH      = WORD_SIZE * 8;
    localparam TAG_WIDTH       = DXA_BANK_WR_TAG_WIDTH;
    localparam SEL_WIDTH       = `UP(`CLOG2(NUM_OUTPUTS));

    // Flatten payload: {tag, per_bank(byteen, data, addr), per_bank(valid)}
    localparam PAYLOAD_W = TAG_WIDTH
                         + NUM_BANKS * (WORD_SIZE + WORD_WIDTH + BANK_ADDR_WIDTH)
                         + NUM_BANKS;

    wire [NUM_INPUTS-1:0]                 req_valid_in;
    wire [NUM_INPUTS-1:0][PAYLOAD_W-1:0]  req_data_in;
    wire [NUM_INPUTS-1:0][SEL_WIDTH-1:0]  req_sel_in;
    wire [NUM_INPUTS-1:0]                 req_ready_in;

    wire [NUM_OUTPUTS-1:0]                req_valid_out;
    wire [NUM_OUTPUTS-1:0][PAYLOAD_W-1:0] req_data_out;
    wire [NUM_OUTPUTS-1:0]                req_ready_out;

    // Flatten inputs
    for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_flatten
        assign req_valid_in[i] = |bank_wr_in[i].wr_valid;
        assign req_data_in[i] = {
            bank_wr_in[i].wr_tag,
            bank_wr_in[i].wr_byteen,
            bank_wr_in[i].wr_data,
            bank_wr_in[i].wr_addr,
            bank_wr_in[i].wr_valid
        };
        assign req_sel_in[i] = SEL_WIDTH'(local_core_id_in[i]);
        assign bank_wr_in[i].wr_ready = req_ready_in[i];
    end

    // Use Vortex stream xbar for M:N routing
    /* verilator lint_off UNUSEDSIGNAL */
    wire [NUM_OUTPUTS-1:0][`UP(`CLOG2(NUM_INPUTS))-1:0] req_sel_out;
    /* verilator lint_on UNUSEDSIGNAL */

    VX_stream_xbar #(
        .NUM_INPUTS  (NUM_INPUTS),
        .NUM_OUTPUTS (NUM_OUTPUTS),
        .DATAW       (PAYLOAD_W),
        .ARBITER     ("R"),
        .OUT_BUF     (2)
    ) socket_xbar (
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

    // Unflatten outputs
    for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_unflatten
        wire [NUM_BANKS-1:0]                       out_wr_valid;
        wire [NUM_BANKS-1:0][BANK_ADDR_WIDTH-1:0]  out_wr_addr;
        wire [NUM_BANKS-1:0][WORD_WIDTH-1:0]       out_wr_data;
        wire [NUM_BANKS-1:0][WORD_SIZE-1:0]        out_wr_byteen;
        wire [TAG_WIDTH-1:0]                       out_wr_tag;

        assign {out_wr_tag, out_wr_byteen, out_wr_data, out_wr_addr, out_wr_valid}
            = req_data_out[o];

        for (genvar b = 0; b < NUM_BANKS; ++b) begin : g_bank
            assign bank_wr_out[o].wr_valid[b]  = req_valid_out[o] && out_wr_valid[b];
            assign bank_wr_out[o].wr_addr[b]   = out_wr_addr[b];
            assign bank_wr_out[o].wr_data[b]   = out_wr_data[b];
            assign bank_wr_out[o].wr_byteen[b] = out_wr_byteen[b];
        end
        assign bank_wr_out[o].wr_tag = out_wr_tag;
        assign req_ready_out[o] = bank_wr_out[o].wr_ready;

        // Output local_core_id is just the port index (1:1 mapping after xbar).
        assign local_core_id_out[o] = DXA_SMEM_LOCAL_CORE_W'(o);
    end

endmodule
