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

module VX_lane_gather import VX_gpu_pkg::*; #(
    parameter BLOCK_SIZE = 1,
    parameter NUM_LANES  = 1,
    parameter OUT_BUF    = 0,
    // commit ports served
    parameter NUM_PORTS  = `VX_CFG_ISSUE_WIDTH
) (
    input  wire         clk,
    input  wire         reset,

    // inputs
    VX_result_if.slave  result_if [BLOCK_SIZE],

    // outputs
    VX_commit_if.master commit_if [NUM_PORTS]
);
    `STATIC_ASSERT (`IS_DIVISBLE(NUM_PORTS, BLOCK_SIZE), ("invalid parameter"))
    `STATIC_ASSERT (`IS_DIVISBLE(`VX_CFG_SIMD_WIDTH, NUM_LANES), ("invalid parameter"))

    `DECL_EXECUTE_T (pe, NUM_LANES);

    localparam BLOCK_SIZE_W = `LOG2UP(BLOCK_SIZE);
    localparam NUM_PACKETS  = `VX_CFG_SIMD_WIDTH / NUM_LANES;
    localparam LPID_BITS    = `CLOG2(NUM_PACKETS);
    localparam LPID_WIDTH   = `UP(LPID_BITS);
    localparam DATAW        = $bits(pe_result_t);
    localparam DATA_WIS_OFF = DATAW - (UUID_WIDTH + NW_WIDTH);
    localparam PORT_W       = `LOG2UP(NUM_PORTS);

    // a partial gather recovers the issue slice from the warp id
    `STATIC_ASSERT ((BLOCK_SIZE == NUM_PORTS) || (NUM_PORTS == `VX_CFG_ISSUE_WIDTH), ("invalid parameter"))

    wire [BLOCK_SIZE-1:0] result_in_valid;
    wire [BLOCK_SIZE-1:0][DATAW-1:0] result_in_data;
    wire [BLOCK_SIZE-1:0] result_in_ready;
    wire [BLOCK_SIZE-1:0][PORT_W-1:0] result_in_isw;

    for (genvar i = 0; i < BLOCK_SIZE; ++i) begin : g_commit_in
        assign result_in_valid[i] = result_if[i].valid;
        assign result_in_data[i]  = result_if[i].data;
        assign result_if[i].ready = result_in_ready[i];
        if (BLOCK_SIZE != NUM_PORTS) begin : g_result_in_isw_partial
            if (BLOCK_SIZE != 1) begin : g_block
                assign result_in_isw[i] = PORT_W'({result_in_data[i][DATA_WIS_OFF+BLOCK_SIZE_W +: (ISSUE_ISW_W-BLOCK_SIZE_W)], BLOCK_SIZE_W'(i)});
            end else begin : g_no_block
                assign result_in_isw[i] = PORT_W'(result_in_data[i][DATA_WIS_OFF +: ISSUE_ISW_W]);
            end
        end else begin : g_result_in_isw_full
            assign result_in_isw[i] = PORT_W'(i);
        end
    end

    reg [NUM_PORTS-1:0] result_out_valid;
    reg [NUM_PORTS-1:0][DATAW-1:0] result_out_data;
    wire [NUM_PORTS-1:0] result_out_ready;

    always @(*) begin
        result_out_valid = '0;
        for (integer i = 0; i < NUM_PORTS; ++i) begin
            result_out_data[i] = 'x;
        end
        for (integer i = 0; i < BLOCK_SIZE; ++i) begin
            result_out_valid[result_in_isw[i]] = result_in_valid[i];
            result_out_data[result_in_isw[i]] = result_in_data[i];
        end
    end

    for (genvar i = 0; i < BLOCK_SIZE; ++i) begin : g_result_in_ready
        assign result_in_ready[i] = result_out_ready[result_in_isw[i]];
    end

    // The hold of a narrow unit: the passes of one instruction that wait for
    // the pass completing it, so that the instruction commits in one beat.
    typedef struct packed {
        logic [UUID_WIDTH-1:0]    uuid;
        logic [NW_WIDTH-1:0]      wid;
        logic [NCTA_WIDTH-1:0]    cta_id;
        logic [SIMD_IDX_W-1:0]    sid;
        logic [PC_BITS-1:0]       PC;
        logic                     wb;
        logic [NUM_XREGS-1:0]     wr_xregs;
        logic [NUM_REGS_BITS-1:0] rd;
        logic [BYTESEL_BITS-1:0]  bytesel;
    } hold_hdr_t;

    for (genvar i = 0; i < NUM_PORTS; ++i) begin: g_out_bufs
        VX_result_if #(
            .data_t (pe_result_t)
        ) result_tmp_if();

        // a narrow unit's results carry whether they complete their beat and
        // whether they continue the hold
        localparam TAG_W     = (LPID_BITS != 0) ? 2 : 0;
        localparam BUF_DATAW = DATAW + TAG_W;

        wire [BUF_DATAW-1:0] buf_in, buf_out;

        VX_elastic_buffer #(
            .DATAW   (BUF_DATAW),
            .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
            .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF))
        ) out_buf (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (result_out_valid[i]),
            .ready_in   (result_out_ready[i]),
            .data_in    (buf_in),
            .data_out   (buf_out),
            .valid_out  (result_tmp_if.valid),
            .ready_out  (result_tmp_if.ready)
        );

        assign result_tmp_if.data = buf_out[DATAW-1:0];

        if (LPID_BITS != 0) begin : g_lpid
            localparam HOLD_LANES = `VX_CFG_SIMD_WIDTH - NUM_LANES;
            localparam KEY_W      = NW_WIDTH + SIMD_IDX_W + PC_BITS + NUM_REGS_BITS + BYTESEL_BITS;

            pe_result_t in_res;
            assign in_res = result_out_data[i];
            `UNUSED_VAR ({in_res.header.uuid, in_res.header.cta_id, in_res.header.tmask, in_res.header.sop})
            `UNUSED_VAR ({in_res.header.wb, in_res.header.wr_xregs, in_res.data})

            logic [SIMD_IDX_W-1:0] in_sid, out_sid;
            logic [LPID_WIDTH-1:0] in_lpid, out_lpid;
            if (SIMD_IDX_BITS != 0) begin : g_simd
                assign {in_sid, in_lpid}   = (SIMD_IDX_W+LPID_WIDTH)'(in_res.header.pid);
                assign {out_sid, out_lpid} = (SIMD_IDX_W+LPID_WIDTH)'(result_tmp_if.data.header.pid);
            end else begin : g_no_simd
                assign in_sid   = '0;
                assign in_lpid  = in_res.header.pid;
                assign out_sid  = '0;
                assign out_lpid = result_tmp_if.data.header.pid;
            end

            // Tag each result on its way in, against the one before it: a
            // result continues the hold when that one waits in it and belongs
            // to the same instruction. The micro-ops of one instruction share
            // its PC and differ in destination.
            wire in_last = in_res.header.eop || (in_lpid == LPID_WIDTH'(NUM_PACKETS-1));
            wire [KEY_W-1:0] in_key = {in_res.header.wid, in_sid, in_res.header.PC, in_res.header.rd, in_res.header.bytesel};
            wire in_fire = result_out_valid[i] && result_out_ready[i];

            reg [KEY_W-1:0] prev_key;
            reg prev_held;
            always @(posedge clk) begin
                if (reset) begin
                    prev_held <= 1'b0;
                end else if (in_fire) begin
                    prev_held <= ~in_last;
                end
                if (in_fire) begin
                    prev_key <= in_key;
                end
            end

            wire in_cont = prev_held && (in_key == prev_key);
            assign buf_in = {in_cont, in_last, result_out_data[i]};

            wire out_cont = buf_out[DATAW+1];
            wire out_last = buf_out[DATAW];

            reg [`VX_CFG_SIMD_WIDTH-1:0] out_lanes;
            always @(*) begin
                out_lanes = '0;
                for (integer j = 0; j < NUM_LANES; ++j) begin
                    out_lanes[out_lpid * NUM_LANES + j] = result_tmp_if.data.header.tmask[j];
                end
            end

            hold_hdr_t out_hdr;
            assign out_hdr = {
                result_tmp_if.data.header.uuid,
                result_tmp_if.data.header.wid,
                result_tmp_if.data.header.cta_id,
                out_sid,
                result_tmp_if.data.header.PC,
                result_tmp_if.data.header.wb,
                result_tmp_if.data.header.wr_xregs,
                result_tmp_if.data.header.rd,
                result_tmp_if.data.header.bytesel
            };

            reg hold_valid;
            reg hold_sop;
            hold_hdr_t hold_hdr;
            reg [HOLD_LANES-1:0] hold_tmask;
            reg [HOLD_LANES-1:0][`VX_CFG_XLEN-1:0] hold_data;

            // A hold the head does not continue commits alone first, and a
            // head that does not complete its beat takes its place in that
            // beat. A head that completes its beat commits with the hold; any
            // other head joins the hold without a beat.
            wire flush_hold = hold_valid && ~out_cont;
            wire take_head  = out_last && ~flush_hold;
            wire out_fire   = commit_if[i].valid && commit_if[i].ready;
            wire absorb     = result_tmp_if.valid && ~out_last && (~flush_hold || commit_if[i].ready);

            assign result_tmp_if.ready = flush_hold ? (~out_last && commit_if[i].ready)
                                                    : (~out_last || commit_if[i].ready);

            always @(posedge clk) begin
                if (reset) begin
                    hold_valid <= 1'b0;
                end else if (absorb) begin
                    hold_valid <= 1'b1;
                end else if (out_fire) begin
                    hold_valid <= 1'b0;
                end
                if (absorb) begin
                    hold_tmask <= (out_cont ? hold_tmask : '0) | out_lanes[HOLD_LANES-1:0];
                    hold_hdr   <= out_hdr;
                    hold_sop   <= out_cont ? hold_sop : result_tmp_if.data.header.sop;
                end
                for (integer k = 0; k < HOLD_LANES; ++k) begin
                    if (absorb && out_lanes[k]) begin
                        hold_data[k] <= result_tmp_if.data.data[k % NUM_LANES];
                    end
                end
            end

            `RUNTIME_ASSERT(~(result_tmp_if.valid && out_cont && ~hold_valid),
                ("%t: lane gather: a result continues an empty hold", $time))

            hold_hdr_t commit_hdr;
            logic [`VX_CFG_SIMD_WIDTH-1:0] commit_tmask_w;
            logic [`VX_CFG_SIMD_WIDTH-1:0][`VX_CFG_XLEN-1:0] commit_data_w;

            // a hold the head continues names the head's instruction too
            assign commit_hdr = hold_valid ? hold_hdr : out_hdr;

            for (genvar k = 0; k < `VX_CFG_SIMD_WIDTH; ++k) begin : g_commit_lanes
                if (k < HOLD_LANES) begin : g_hold
                    wire held = hold_valid && hold_tmask[k];
                    assign commit_tmask_w[k] = held || (take_head && out_lanes[k]);
                    assign commit_data_w[k]  = held ? hold_data[k] : result_tmp_if.data.data[k % NUM_LANES];
                end else begin : g_head
                    assign commit_tmask_w[k] = take_head && out_lanes[k];
                    assign commit_data_w[k]  = result_tmp_if.data.data[k % NUM_LANES];
                end
            end

            assign commit_if[i].valid = result_tmp_if.valid && (out_last || flush_hold);
            assign commit_if[i].data = {
                commit_hdr.uuid,
                commit_hdr.wid,
                commit_hdr.cta_id,
                commit_hdr.sid,
                commit_tmask_w,
                commit_hdr.PC,
                commit_hdr.wb,
                commit_hdr.wr_xregs,
                commit_hdr.rd,
                commit_hdr.bytesel,
                commit_data_w,
                hold_valid ? hold_sop : result_tmp_if.data.header.sop,
                result_tmp_if.data.header.eop && ~flush_hold
            };
        end else begin : g_no_lpid
            assign buf_in = result_out_data[i];

            assign commit_if[i].valid = result_tmp_if.valid;
            assign commit_if[i].data = {
                result_tmp_if.data.header.uuid,
                result_tmp_if.data.header.wid,
                result_tmp_if.data.header.cta_id,
                SIMD_IDX_W'(result_tmp_if.data.header.pid),
                result_tmp_if.data.header.tmask,
                result_tmp_if.data.header.PC,
                result_tmp_if.data.header.wb,
                result_tmp_if.data.header.wr_xregs,
                result_tmp_if.data.header.rd,
                result_tmp_if.data.header.bytesel,
                result_tmp_if.data.data,
                result_tmp_if.data.header.sop,
                result_tmp_if.data.header.eop
            };
            assign result_tmp_if.ready = commit_if[i].ready;
        end
    end

endmodule
