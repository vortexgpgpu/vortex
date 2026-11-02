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

// Commit for one issue slice: arbitrates its execution units' results onto the
// slice's writeback port and reports the warps whose instruction retired.
module VX_commit_slice import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

    // inputs
    VX_commit_if.slave      commit_if [NUM_EX_UNITS],

    // outputs
    VX_writeback_if.master  writeback_if,
    output wire [PER_ISSUE_WARPS-1:0] committed_wis   // registered
);
    `UNUSED_SPARAM (INSTANCE_ID)
    localparam OUT_DATAW = $bits(commit_t);

    VX_commit_if commit_arb_if();
    wire [PER_ISSUE_WARPS-1:0] commit_eop_wis;

    wire [NUM_EX_UNITS-1:0]            valid_in;
    wire [NUM_EX_UNITS-1:0][PER_ISSUE_WARPS+OUT_DATAW-1:0] data_in;
    wire [NUM_EX_UNITS-1:0]            ready_in;

    // Decoded ahead of the output register so each warp's release reads
    // its own flop.
    for (genvar j = 0; j < NUM_EX_UNITS; ++j) begin : g_data_in
        wire [PER_ISSUE_WARPS-1:0] eop_wis = PER_ISSUE_WARPS'(commit_if[j].data.eop)
                                          << wid_to_wis(commit_if[j].data.wid);
        assign valid_in[j] = commit_if[j].valid;
        assign data_in[j]  = {eop_wis, commit_if[j].data};
        assign commit_if[j].ready = ready_in[j];
    end

    VX_stream_arb #(
        .NUM_INPUTS (NUM_EX_UNITS),
        .DATAW      (PER_ISSUE_WARPS + OUT_DATAW),
        .ARBITER    ("P"),
        .OUT_BUF    (1)
    ) commit_arb (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (valid_in),
        .ready_in   (ready_in),
        .data_in    (data_in),
        .data_out   ({commit_eop_wis, commit_arb_if.data}),
        .valid_out  (commit_arb_if.valid),
        .ready_out  (commit_arb_if.ready),
        `UNUSED_PIN (sel_out)
    );

    // notify scheduler
    wire [PER_ISSUE_WARPS-1:0] committed_wis_n = commit_arb_if.valid ? commit_eop_wis : '0;
    `BUFFER(committed_wis, committed_wis_n);

    // Writeback

    wire [XLENB_W-1:0] bytesel_size = commit_arb_if.data.bytesel[BYTESEL_BITS-1 -: XLENB_W];
    wire [XLENB_W-1:0] bytesel_off  = commit_arb_if.data.bytesel[0 +: XLENB_W];
    wire [`VX_CFG_SIMD_WIDTH-1:0][`VX_CFG_XLEN-1:0] writeback_data;
    wire [`VX_CFG_SIMD_WIDTH-1:0][XLENB-1:0] writeback_byteen;

    wire [XLENB-1:0] size_mask = (bytesel_size == XLENB_W'(7)) ? XLENB'(255) :
                                 (bytesel_size == XLENB_W'(6)) ? XLENB'(127) :
                                 (bytesel_size == XLENB_W'(5)) ? XLENB'(63)  :
                                 (bytesel_size == XLENB_W'(4)) ? XLENB'(31)  :
                                 (bytesel_size == XLENB_W'(3)) ? XLENB'(15)  :
                                 (bytesel_size == XLENB_W'(2)) ? XLENB'(7)   :
                                 (bytesel_size == XLENB_W'(1)) ? XLENB'(3)   : XLENB'(1);
    wire [XLENB-1:0] base_byteen = size_mask << bytesel_off;

    for (genvar lane = 0; lane < `VX_CFG_SIMD_WIDTH; ++lane) begin : g_bytesel
        assign writeback_data[lane] = commit_arb_if.data.data[lane] << (8 * bytesel_off);
        assign writeback_byteen[lane] = commit_arb_if.data.tmask[lane] ? base_byteen : '0;
    end

    assign writeback_if.valid     = commit_arb_if.valid;
    assign writeback_if.data.uuid = commit_arb_if.data.uuid;
    assign writeback_if.data.wis  = wid_to_wis(commit_arb_if.data.wid);
    assign writeback_if.data.eop_wis = commit_eop_wis;
    assign writeback_if.data.cta_id = commit_arb_if.data.cta_id;
    assign writeback_if.data.sid  = commit_arb_if.data.sid;
    assign writeback_if.data.PC   = commit_arb_if.data.PC;
    assign writeback_if.data.tmask= commit_arb_if.data.tmask;
    assign writeback_if.data.wb   = commit_arb_if.data.wb;
    assign writeback_if.data.wr_xregs = commit_arb_if.data.wr_xregs;
    assign writeback_if.data.rd   = commit_arb_if.data.rd;
    assign writeback_if.data.byteen = writeback_byteen;
    assign writeback_if.data.data = writeback_data;
    assign writeback_if.data.sop  = commit_arb_if.data.sop;
    assign writeback_if.data.eop  = commit_arb_if.data.eop;
    assign commit_arb_if.ready    = 1;

`ifdef DBG_TRACE_PIPELINE
    for (genvar j = 0; j < NUM_EX_UNITS; ++j) begin : g_trace
        always @(posedge clk) begin
            if (commit_if[j].valid && commit_if[j].ready) begin
                `TRACE(1, ("%t: %s commit: wid=%0d, cta_id=%0d, sid=%0d, PC=0x%0h, ex=", $time, INSTANCE_ID, commit_if[j].data.wid, commit_if[j].data.cta_id, commit_if[j].data.sid, to_fullPC(commit_if[j].data.PC)))
                VX_trace_pkg::trace_ex_type(1, j);
                `TRACE(1, (", tmask=%b, wb=%b, wr_xregs=%b, rd=%0d, sop=%b, eop=%b, data=", commit_if[j].data.tmask, commit_if[j].data.wb, commit_if[j].data.wr_xregs, commit_if[j].data.rd, commit_if[j].data.sop, commit_if[j].data.eop))
                `TRACE_ARRAY1D(1, "0x%0h", commit_if[j].data.data, `VX_CFG_SIMD_WIDTH)
                `TRACE(1, (" (#%0d)\n", commit_if[j].data.uuid))
            end
        end
    end
`endif

endmodule
