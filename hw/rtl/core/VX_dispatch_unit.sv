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

module VX_dispatch_unit import VX_gpu_pkg::*; #(
    parameter BLOCK_SIZE = 1,
    parameter NUM_LANES  = 1,
    parameter OUT_BUF    = 0,
    parameter MAX_FANOUT = `MAX_FANOUT
) (
    input  wire             clk,
    input  wire             reset,

    // inputs
    VX_dispatch_if.slave    dispatch_if [`ISSUE_WIDTH],

    // outputs
    VX_execute_if.master    execute_if [BLOCK_SIZE]
);
    `STATIC_ASSERT (`IS_DIVISBLE(`ISSUE_WIDTH, BLOCK_SIZE), ("invalid parameter"))
    `STATIC_ASSERT (`IS_DIVISBLE(`SIMD_WIDTH, NUM_LANES), ("invalid parameter"))
    localparam BLOCK_SIZE_W = `LOG2UP(BLOCK_SIZE);
    localparam NUM_PACKETS  = `SIMD_WIDTH / NUM_LANES;
    localparam LPID_BITS    = `CLOG2(NUM_PACKETS);
    localparam LPID_WIDTH   = `UP(LPID_BITS);
    localparam GPID_BITS    = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam GPID_WIDTH   = `UP(GPID_BITS);
    localparam BATCH_COUNT  = `ISSUE_WIDTH / BLOCK_SIZE;
    localparam BATCH_COUNT_W= `LOG2UP(BATCH_COUNT);
    localparam ISSUE_W      = `LOG2UP(`ISSUE_WIDTH);
    localparam IN_DATAW     = UUID_WIDTH + ISSUE_WIS_W + SIMD_IDX_W + `SIMD_WIDTH + INST_OP_BITS + INST_ARGS_BITS + 1 + PC_BITS + NUM_REGS_BITS + (NUM_SRC_OPDS * `SIMD_WIDTH * `XLEN) + 1 + 1;
    localparam OUT_DATAW    = UUID_WIDTH + NW_WIDTH + NUM_LANES + INST_OP_BITS + INST_ARGS_BITS + 1 + PC_BITS + NUM_REGS_BITS + (NUM_SRC_OPDS * NUM_LANES * `XLEN) + GPID_WIDTH + 1 + 1;
    localparam FANOUT_ENABLE= (`SIMD_WIDTH > (MAX_FANOUT + MAX_FANOUT /2));

    localparam DATA_TMASK_OFF = IN_DATAW - (UUID_WIDTH + ISSUE_WIS_W + SIMD_IDX_W + `SIMD_WIDTH);
    localparam DATA_REGS_OFF = 1 + 1;

    typedef struct packed {
        logic [2:0][NUM_LANES-1:0][`XLEN-1:0] rsdata;
        logic [NUM_LANES-1:0] tmask;
    } packet_t;

    wire [`ISSUE_WIDTH-1:0] dispatch_valid;
    wire [`ISSUE_WIDTH-1:0][IN_DATAW-1:0] dispatch_data;
    wire [`ISSUE_WIDTH-1:0] dispatch_ready;

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin : g_dispatch_data
        assign dispatch_valid[i] = dispatch_if[i].valid;
        assign dispatch_data[i] = dispatch_if[i].data;
        assign dispatch_if[i].ready = dispatch_ready[i];
    end

    wire [BLOCK_SIZE-1:0] block_ready;
    wire [BLOCK_SIZE-1:0][NUM_LANES-1:0] block_tmask;
    wire [BLOCK_SIZE-1:0][2:0][NUM_LANES-1:0][`XLEN-1:0] block_rsdata;
    wire [BLOCK_SIZE-1:0][LPID_WIDTH-1:0] block_pid;
    wire [BLOCK_SIZE-1:0] block_sop;
    wire [BLOCK_SIZE-1:0] block_eop;
    wire [BLOCK_SIZE-1:0] block_done;

    wire batch_done = (& block_done);

    // batch select logic

    logic [BATCH_COUNT_W-1:0] batch_idx;

    if (BATCH_COUNT != 1) begin : g_batch_idx
        wire [BATCH_COUNT_W-1:0] batch_idx_n;
        wire [BATCH_COUNT-1:0] valid_batches;
        for (genvar i = 0; i < BATCH_COUNT; ++i) begin : g_valid_batches
            assign valid_batches[i] = | dispatch_valid[i * BLOCK_SIZE +: BLOCK_SIZE];
        end

        VX_generic_arbiter #(
            .NUM_REQS    (BATCH_COUNT),
            .TYPE        ("P")
        ) batch_sel (
            .clk          (clk),
            .reset        (reset),
            .requests     (valid_batches),
            .grant_index  (batch_idx_n),
            `UNUSED_PIN (grant_onehot),
            `UNUSED_PIN (grant_valid),
            .grant_ready  (batch_done)
        );

        always @(posedge clk) begin
            if (reset) begin
                batch_idx <= '0;
            end else if (batch_done) begin
                batch_idx <= batch_idx_n;
            end
        end
    end else begin : g_batch_idx_0
        assign batch_idx = 0;
        `UNUSED_VAR (batch_done)
    end

    wire [BLOCK_SIZE-1:0][ISSUE_W-1:0] issue_indices;
    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_issue_indices
        assign issue_indices[block_idx] = ISSUE_W'(batch_idx * BLOCK_SIZE) + ISSUE_W'(block_idx);
    end

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_blocks

        wire [ISSUE_W-1:0] issue_idx = issue_indices[block_idx];
        wire [ISSUE_WIS_W-1:0] dispatch_wis = dispatch_data[issue_idx][DATA_TMASK_OFF + `SIMD_WIDTH + SIMD_IDX_W +: ISSUE_WIS_W];
        wire [SIMD_IDX_W-1:0] dispatch_sid = dispatch_data[issue_idx][DATA_TMASK_OFF + `SIMD_WIDTH +: SIMD_IDX_W];
        wire dispatch_sop = dispatch_data[issue_idx][1];
        wire dispatch_eop = dispatch_data[issue_idx][0];

        wire [`SIMD_WIDTH-1:0] dispatch_tmask;
        wire [2:0][`SIMD_WIDTH-1:0][`XLEN-1:0] dispatch_rsdata;

        assign dispatch_tmask = dispatch_data[issue_idx][DATA_TMASK_OFF +: `SIMD_WIDTH];
        assign dispatch_rsdata[0] = dispatch_data[issue_idx][DATA_REGS_OFF + 2 * `SIMD_WIDTH * `XLEN +: `SIMD_WIDTH * `XLEN];
        assign dispatch_rsdata[1] = dispatch_data[issue_idx][DATA_REGS_OFF + 1 * `SIMD_WIDTH * `XLEN +: `SIMD_WIDTH * `XLEN];
        assign dispatch_rsdata[2] = dispatch_data[issue_idx][DATA_REGS_OFF + 0 * `SIMD_WIDTH * `XLEN +: `SIMD_WIDTH * `XLEN];

        wire valid_p, ready_p;

        if (`SIMD_WIDTH != NUM_LANES) begin : g_partial_simd

            packet_t [NUM_PACKETS-1:0] packets;

            for (genvar i = 0; i < NUM_PACKETS; ++i) begin : g_per_packet_data
                for (genvar j = 0; j < NUM_LANES; ++j) begin : g_j
                    localparam k = i * NUM_LANES + j;
                    assign packets[i].tmask[j]   = dispatch_tmask[k];
                    assign packets[i].rsdata[0][j] = dispatch_rsdata[0][k];
                    assign packets[i].rsdata[1][j] = dispatch_rsdata[1][k];
                    assign packets[i].rsdata[2][j] = dispatch_rsdata[2][k];
                end
            end

            wire [LPID_WIDTH-1:0] start_p;
            wire is_first_p, is_last_p;
            packet_t block_packet;

            wire fire_p = valid_p && ready_p;

            VX_nz_iterator #(
                .DATAW   ($bits(packet_t)),
                .KEYW    (NUM_LANES),
                .N       (NUM_PACKETS),
                .OUT_REG (FANOUT_ENABLE)
            ) packet_iter (
                .clk     (clk),
                .reset   (reset),
                .valid_in(dispatch_valid[issue_idx]),
                .data_in (packets),
                .next    (fire_p),
                .valid_out(valid_p),
                .data_out(block_packet),
                .pid     (start_p),
                .sop     (is_first_p),
                .eop     (is_last_p)
            );

            assign block_tmask[block_idx] = block_packet.tmask;
            assign block_rsdata[block_idx] = block_packet.rsdata;
            assign block_pid[block_idx]   = start_p;
            assign block_sop[block_idx]   = is_first_p;
            assign block_eop[block_idx]   = is_last_p;
            assign block_ready[block_idx] = ready_p;
            assign block_done[block_idx]  = (fire_p && is_last_p) || ~dispatch_valid[issue_idx];
        end else begin : g_full_simd
            assign valid_p = dispatch_valid[issue_idx];
            assign block_tmask[block_idx] = dispatch_tmask;
            assign block_rsdata[block_idx] = dispatch_rsdata;
            assign block_pid[block_idx]   = 0;
            assign block_sop[block_idx]   = 1;
            assign block_eop[block_idx]   = 1;
            assign block_ready[block_idx] = ready_p;
            assign block_done[block_idx]  = ready_p || ~valid_p;
        end

        wire [ISSUE_ISW_W-1:0] isw;
        if (BATCH_COUNT != 1) begin : g_isw_batch
            if (BLOCK_SIZE != 1) begin : g_block
                assign isw = {batch_idx, BLOCK_SIZE_W'(block_idx)};
            end else begin : g_no_block
                assign isw = batch_idx;
            end
        end else begin : g_isw
            assign isw = block_idx;
        end

        wire [NW_WIDTH-1:0] block_wid = wis_to_wid(dispatch_wis, isw);
        wire [GPID_WIDTH-1:0] warp_pid = GPID_WIDTH'(block_pid[block_idx]) + GPID_WIDTH'(dispatch_sid * NUM_PACKETS);

        wire warp_sop = block_sop[block_idx] && dispatch_sop;
        wire warp_eop = block_eop[block_idx] && dispatch_eop;

        VX_elastic_buffer #(
            .DATAW   (OUT_DATAW),
            .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
            .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF))
        ) buf_out (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (valid_p),
            .ready_in  (ready_p),
            .data_in   ({
                dispatch_data[issue_idx][IN_DATAW-1 -: UUID_WIDTH],
                block_wid,
                block_tmask[block_idx],
                dispatch_data[issue_idx][DATA_TMASK_OFF-1 : (DATA_REGS_OFF + NUM_SRC_OPDS * `SIMD_WIDTH * `XLEN)],
                block_rsdata[block_idx][0],
                block_rsdata[block_idx][1],
                block_rsdata[block_idx][2],
                warp_pid,
                warp_sop,
                warp_eop}),
            .data_out  (execute_if[block_idx].data),
            .valid_out (execute_if[block_idx].valid),
            .ready_out (execute_if[block_idx].ready)
        );
    end

    // release the dispatch interface when all packets are sent
    reg [`ISSUE_WIDTH-1:0] ready_in;
    always @(*) begin
        ready_in = 0;
        for (integer block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin
            ready_in[issue_indices[block_idx]] = block_ready[block_idx] && block_eop[block_idx];
        end
    end
    assign dispatch_ready = ready_in;

endmodule
