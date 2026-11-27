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

module VX_scoreboard import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output reg [PERF_CTR_BITS-1:0] perf_stalls,
`endif

    input wire [NUM_EX_UNITS-1:0] fu_release,
    VX_writeback_if.slave   writeback_if,
    VX_ibuffer_if.slave     ibuffer_if [PER_ISSUE_WARPS],
    VX_scoreboard_if.master scoreboard_if
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR ({writeback_if.data.sop, writeback_if.data.eop, writeback_if.data.wis})

    localparam IN_DATAW  = $bits(ibuffer_t);
    localparam OUT_DATAW = $bits(scoreboard_t) - ISSUE_WIS_W;
    localparam OUT_BUF   = 3; // Use skid buffer (SIZE=2, OUT_REG=1)

    // Per-FU dispatch credits: spent at issue, reclaimed on FU accept, so a
    // credit covers ops still in operand collection (not yet at the queue).
    wire [NUM_EX_UNITS-1:0] fu_issue;
    wire [NUM_EX_UNITS-1:0] fu_goingfull;

    // going-full (not full): a 1-slot guard band keeps outstanding <= queue
    // depth despite the registered suppress lag, so an issued op never stalls
    // in the shared operand-collection path and HoL-blocks another FU.
    for (genvar e = 0; e < NUM_EX_UNITS; ++e) begin : g_fu_goingfull
        VX_pending_size #(
            .SIZE (DISPATCH_QSIZE)
        ) fu_pending (
            .clk        (clk),
            .reset      (reset),
            .incr       (fu_issue[e]),
            .decr       (fu_release[e]),
            `UNUSED_PIN (empty),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (full),
            .alm_full   (fu_goingfull[e]),
            `UNUSED_PIN (size)
        );
    end

    VX_ibuffer_if staging_if [PER_ISSUE_WARPS]();
    wire [PER_ISSUE_WARPS-1:0] operands_ready;

`ifdef PERF_ENABLE
    wire [PER_ISSUE_WARPS-1:0] stg_valid_in;
    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_stg_valid_in
        assign stg_valid_in[w] = staging_if[w].valid;
    end

    wire perf_stall_per_cycle = |(stg_valid_in & ~operands_ready);

    always @(posedge clk) begin : g_perf_stalls
        if (reset) begin
            perf_stalls <= '0;
        end else begin
            perf_stalls <= perf_stalls + PERF_CTR_BITS'(perf_stall_per_cycle);
        end
    end
`endif

    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_stanging_bufs
        VX_pipe_buffer #(
            .DATAW (IN_DATAW)
        ) stanging_buf (
            .clk      (clk),
            .reset    (reset),
            .valid_in (ibuffer_if[w].valid),
            .data_in  (ibuffer_if[w].data),
            .ready_in (ibuffer_if[w].ready),
            .valid_out(staging_if[w].valid),
            .data_out (staging_if[w].data),
            .ready_out(staging_if[w].ready)
        );
    end

    wire [NUM_REGS-1:0] wb_rd_onehot = NUM_REGS'(1) << writeback_if.data.rd;

    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_scoreboard
        VX_scoreboard_table #(
            .INSTANCE_ID (INSTANCE_ID)
        ) inuse_table (
            .clk            (clk),
            .reset          (reset),
            .wid            (ISSUE_WIS_W'(w)),
            .fu_goingfull   (fu_goingfull),
            .ibf_valid      (ibuffer_if[w].valid),
            .ibf_ready      (ibuffer_if[w].ready),
            .ibf_data       (ibuffer_if[w].data),
            .stg_valid      (staging_if[w].valid),
            .stg_ready      (staging_if[w].ready),
            .stg_data       (staging_if[w].data),
            .wb_valid       (writeback_if.valid),
            .wb_eop         (writeback_if.data.eop_wis[w]),
            .wb_wb          (writeback_if.data.wb),
            .wb_rd          (writeback_if.data.rd),
            .wb_rd_onehot   (wb_rd_onehot),
            .wb_wr_xregs    (writeback_if.data.wr_xregs),
            .wb_PC          (writeback_if.data.PC),
            .wb_tmask       (writeback_if.data.tmask),
            .wb_uuid        (writeback_if.data.uuid),
            .operands_ready (operands_ready[w])
        );
    end

    wire [PER_ISSUE_WARPS-1:0] arb_valid_in;
    wire [PER_ISSUE_WARPS-1:0][OUT_DATAW-1:0] arb_data_in;
    wire [PER_ISSUE_WARPS-1:0] arb_ready_in;

    // FU lock: a sequence must not interleave with another warp at the same FU.
    // fu_locked ('1 = open, one-hot = locked) gates arb_valid_in so only the lock
    // holder is requested while it holds the lock.
    reg [PER_ISSUE_WARPS-1:0] fu_locked;

`ifdef EXT_LDST_ANY_ENABLE
    // A warp that issues a yielding uop while another warp also requests with
    // one sits out the next cycle, so the greedy arbiter moves on as it does
    // between scalar instructions that arrive one per fetch.
    reg [PER_ISSUE_WARPS-1:0] yield_r;
    wire [PER_ISSUE_WARPS-1:0] staging_yield_vec;
    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_staging_yield
        assign staging_yield_vec[w] = staging_if[w].data.issue_yield;
    end
    wire [PER_ISSUE_WARPS-1:0] yield_reqs = arb_valid_in & staging_yield_vec;
    wire [PER_ISSUE_WARPS-1:0] req_below;
    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_req_below
        if (w == 0) begin : g_first
            assign req_below[w] = 1'b0;
        end else begin : g_next
            assign req_below[w] = req_below[w-1] || yield_reqs[w-1];
        end
    end
    wire multi_req = | (yield_reqs & req_below);
`else
    wire [PER_ISSUE_WARPS-1:0] yield_r = '0;
`endif

    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_arb_data_in
        // operands_ready carries data-hazard + FU-congestion; fu_locked adds the
        // FU-lock gate so only the lock holder is requested during a sequence.
        assign arb_valid_in[w] = staging_if[w].valid && operands_ready[w] && fu_locked[w] && ~yield_r[w];

        assign arb_data_in[w] = {
            staging_if[w].data.uuid,
            staging_if[w].data.cta_id,
            staging_if[w].data.tmask,
            staging_if[w].data.PC,
            staging_if[w].data.ex_type,
            staging_if[w].data.op_type,
            staging_if[w].data.op_args,
            staging_if[w].data.wb,
            staging_if[w].data.wr_xregs,
            staging_if[w].data.used_rs,
            staging_if[w].data.rd,
            staging_if[w].data.bytesel,
            staging_if[w].data.rs1,
            staging_if[w].data.rs2,
            staging_if[w].data.rs3
        };
        assign staging_if[w].ready = arb_ready_in[w] && operands_ready[w];
    end

    localparam LOG_NUM_REQS = `LOG2UP(PER_ISSUE_WARPS);

    wire                    arb_valid;
    wire [LOG_NUM_REQS-1:0] arb_index;
    wire [PER_ISSUE_WARPS-1:0] arb_onehot;
    wire                    arb_ready;

    // Matrix arbiter scales better past 8 requesters; RR is cheaper below that.
    VX_generic_arbiter #(
        .NUM_REQS (PER_ISSUE_WARPS),
        .TYPE     ((PER_ISSUE_WARPS > 8) ? "M" : "R"),
        .STICKY   (1) // Greedy
    ) out_arb (
        .clk          (clk),
        .reset        (reset),
        .requests     (arb_valid_in),
        .grant_valid  (arb_valid),
        .grant_index  (arb_index),
        .grant_onehot (arb_onehot),
        .grant_ready  (arb_ready)
    );

    wire valid_out_w;
    wire [OUT_DATAW-1:0] data_out_w;
    wire ready_out_w;

    assign valid_out_w = arb_valid;
    assign data_out_w  = arb_data_in[arb_index];

    for (genvar i = 0; i < PER_ISSUE_WARPS; ++i) begin : g_arb_ready_in
        assign arb_ready_in[i] = ready_out_w && arb_onehot[i];
    end

    assign arb_ready = ready_out_w;

    // A sequence carries fu_lock on its first uop (acquire) and fu_unlock on its
    // last (release); 11 = single-uop default.

    wire issue_fire = valid_out_w && ready_out_w;

    wire [PER_ISSUE_WARPS-1:0][EX_BITS-1:0] staging_ex_vec;
    wire [PER_ISSUE_WARPS-1:0] staging_fu_lock_vec;
    wire [PER_ISSUE_WARPS-1:0] staging_fu_unlock_vec;
    for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_staging_fu_lock
        assign staging_ex_vec[w]        = staging_if[w].data.ex_type;
        assign staging_fu_lock_vec[w]   = staging_if[w].data.fu_lock;
        assign staging_fu_unlock_vec[w] = staging_if[w].data.fu_unlock;
    end

    wire [EX_BITS-1:0] issue_ex = staging_ex_vec[arb_index];

    for (genvar e = 0; e < NUM_EX_UNITS; ++e) begin : g_fu_issue
        assign fu_issue[e] = issue_fire && (issue_ex == EX_BITS'(e));
    end

    // Lock to the granted warp on acquire, hold across its sequence, reopen on
    // release. arb_onehot is the granted warp, registered into fu_locked.
    wire issue_fu_lock   = staging_fu_lock_vec[arb_index];
    wire issue_fu_unlock = staging_fu_unlock_vec[arb_index];

    always @(posedge clk) begin
        if (reset) begin
            fu_locked <= '1;
        end else if (issue_fire) begin
            if (issue_fu_lock && ~issue_fu_unlock) begin
                fu_locked <= arb_onehot;
            end else if (issue_fu_unlock) begin
                fu_locked <= '1;
            end
        end
    end

`ifdef EXT_LDST_ANY_ENABLE
    always @(posedge clk) begin
        if (reset) begin
            yield_r <= '0;
        end else begin
            yield_r <= arb_onehot & staging_yield_vec & {PER_ISSUE_WARPS{issue_fire && multi_req}};
        end
    end
`endif

    VX_elastic_buffer #(
        .DATAW   (LOG_NUM_REQS + OUT_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF)),
        .LUTRAM  (`TO_OUT_BUF_LUTRAM(OUT_BUF))
    ) out_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (valid_out_w),
        .ready_in  (ready_out_w),
        .data_in   ({arb_index, data_out_w}),
        .data_out  ({
            scoreboard_if.data.wis,
            scoreboard_if.data.uuid,
            scoreboard_if.data.cta_id,
            scoreboard_if.data.tmask,
            scoreboard_if.data.PC,
            scoreboard_if.data.ex_type,
            scoreboard_if.data.op_type,
            scoreboard_if.data.op_args,
            scoreboard_if.data.wb,
            scoreboard_if.data.wr_xregs,
            scoreboard_if.data.used_rs,
            scoreboard_if.data.rd,
            scoreboard_if.data.bytesel,
            scoreboard_if.data.rs1,
            scoreboard_if.data.rs2,
            scoreboard_if.data.rs3
        }),
        .valid_out (scoreboard_if.valid),
        .ready_out (scoreboard_if.ready)
    );

endmodule
