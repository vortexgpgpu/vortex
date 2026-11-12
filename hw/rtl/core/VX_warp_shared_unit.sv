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

// The core-wide warp-control resources: barrier unit, wspawn and warp retirement.
// One request per cycle across the requesters, round-robin; a request that
// asks for it is completed back to its requester. Transaction-barrier
// completions use the barrier unit when no request does.
//
// Several requesters register their requests toward the arbiter, which acts on
// the grant and completes it registered. A sole requester is connected
// directly: it is completed in the cycle it asks, and its request is registered
// here instead, so every request takes effect at the same distance from its
// instruction in both cases.
module VX_warp_shared_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_REQS = 1
) (
    input wire              clk,
    input wire              reset,

    input wire [CORE_ID_WIDTH-1:0] core_id,

    VX_warp_shared_if.slave warp_shared_if [NUM_REQS],
`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_txbar_bus_if.slave   txbar_bus_if,
`endif
    input wire [`VX_CFG_NUM_WARPS-1:0] active_warps,

    // barrier release
    output wire             bar_unlock_valid,
    output wire [`VX_CFG_NUM_WARPS-1:0] bar_unlock_mask,
    VX_gbar_bus_if.master   gbar_bus_if,

    // wspawn, applied once a single warp remains
    output wire             wspawn_apply,
    output wire [`VX_CFG_NUM_WARPS-1:0] wspawn_wmask,
    output wire [PC_BITS-1:0] wspawn_pc,
    output wire [NW_WIDTH-1:0] wspawn_wid,
    output wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] wspawn_mscratch,

    // warp retirement
    output wire             warp_done,
    output wire [NW_WIDTH-1:0] warp_done_wid,

    output wire             wspawn_pending
);
    localparam REQ_DATAW = $bits(warp_shared_req_t);
    localparam SEL_W     = `LOG2UP(NUM_REQS);
    localparam DIRECT    = (NUM_REQS == 1);

    wire [NUM_REQS-1:0]                req_valid_in;
    wire [NUM_REQS-1:0][REQ_DATAW-1:0] req_data_in;
    wire [NUM_REQS-1:0]                req_ready_in;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_in
        assign req_valid_in[i] = warp_shared_if[i].req_valid;
        assign req_data_in[i]  = warp_shared_if[i].req_data;
        assign warp_shared_if[i].req_ready = req_ready_in[i];
    end

    wire              req_valid;
    warp_shared_req_t req;
    wire [SEL_W-1:0]  req_sel;

    VX_stream_arb #(
        .NUM_INPUTS  (NUM_REQS),
        .NUM_OUTPUTS (1),
        .DATAW       (REQ_DATAW),
        .ARBITER     ("R"),
        .OUT_BUF     (0)
    ) req_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (req_valid_in),
        .ready_in  (req_ready_in),
        .data_in   (req_data_in),
        .data_out  (req),
        .valid_out (req_valid),
        .ready_out (1'b1),
        .sel_out   (req_sel)
    );

    wire is_bar    = req_valid && (req.kind == WSH_BAR);
    wire is_wspawn = req_valid && (req.kind == WSH_WSPAWN);
    wire is_exit   = req_valid && (req.kind == WSH_EXIT);

    // wspawn and retirement act on the registered request
    localparam ACT_DATAW = NW_WIDTH + $bits(wspawn_t) + `VX_CFG_MEM_ADDR_WIDTH;
    wire act_wspawn, act_exit;
    wire [NW_WIDTH-1:0] act_wid;
    wspawn_t act_wspawn_data;
    wire [`VX_CFG_MEM_ADDR_WIDTH-1:0] act_mscratch;
    if (DIRECT) begin : g_act_reg
        VX_pipe_register #(
            .DATAW  (2 + ACT_DATAW),
            .RESETW (2)
        ) act_reg (
            .clk      (clk),
            .reset    (reset),
            .enable   (1'b1),
            .data_in  ({is_wspawn,  is_exit,  req.wid, req.wspawn,      req.mscratch}),
            .data_out ({act_wspawn, act_exit, act_wid, act_wspawn_data, act_mscratch})
        );
    end else begin : g_act
        assign {act_wspawn, act_exit, act_wid, act_wspawn_data, act_mscratch}
             = {is_wspawn,  is_exit,  req.wid, req.wspawn,      req.mscratch};
    end

    // barrier unit

    wire txbar_req_valid;
    barrier_t txbar_req;
    wire [BAR_ADDR_W-1:0] bar_read_addr;
`ifdef VX_CFG_EXT_DXA_ENABLE
    assign txbar_bus_if.ready = ~is_bar;
    wire txbar_fire = txbar_bus_if.valid && txbar_bus_if.ready;
    assign bar_read_addr = is_bar ? req.bar_addr : txbar_bus_if.data.addr;
    // a completion releases one pending transaction event
    barrier_t txbar_req_w;
    assign txbar_req_w.id        = '0;
    assign txbar_req_w.is_event  = 1'b1;
    assign txbar_req_w.is_sync   = 1'b0;
    assign txbar_req_w.is_global = 1'b0;
    assign txbar_req_w.is_arrive = 1'b0;
    assign txbar_req_w.phase     = ~txbar_bus_if.data.is_done;
    assign txbar_req_w.size_m1   = '0;
    VX_pipe_register #(
        .DATAW  (1 + $bits(barrier_t)),
        .RESETW (1)
    ) txbar_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({txbar_fire, txbar_req_w}),
        .data_out ({txbar_req_valid, txbar_req})
    );
`else
    assign bar_read_addr   = req.bar_addr;
    assign txbar_req_valid = 1'b0;
    assign txbar_req       = '0;
`endif

    // the barrier request applies the cycle after its phase is read, as the
    // read must see the barrier before this arrival
    reg              bar_valid_r;
    barrier_t        bar_r;
    reg [NW_WIDTH-1:0] bar_wid_r;
    always @(posedge clk) begin
        if (reset) begin
            bar_valid_r <= 1'b0;
        end else begin
            bar_valid_r <= is_bar;
        end
        bar_r     <= req.bar;
        bar_wid_r <= req.wid;
    end

    wire bar_read_phase;

    VX_bar_unit #(
        .INSTANCE_ID (`SFORMATF(("%s-barrier", INSTANCE_ID)))
    ) bar_unit (
        .clk          (clk),
        .reset        (reset),
        .core_id      (core_id),
        .req_valid    (bar_valid_r || txbar_req_valid),
        .req_wid      (bar_wid_r),
        .req_data     (bar_valid_r ? bar_r : txbar_req),
        .read_addr    (bar_read_addr),
        .read_phase   (bar_read_phase),
        .active_warps (active_warps),
        .gbar_bus_if  (gbar_bus_if),
        .unlock_valid (bar_unlock_valid),
        .unlock_mask  (bar_unlock_mask)
    );

    // wspawn

    reg wspawn_valid_r;
    wspawn_t wspawn_r;
    reg [NW_WIDTH-1:0] wspawn_wid_r;
    reg [`VX_CFG_MEM_ADDR_WIDTH-1:0] wspawn_mscratch_r;
    reg is_single_warp;

    wire [`CLOG2(`VX_CFG_NUM_WARPS+1)-1:0] active_warps_cnt;
    `POP_COUNT(active_warps_cnt, active_warps);

    assign wspawn_apply = wspawn_valid_r && is_single_warp;

    always @(posedge clk) begin
        if (reset) begin
            wspawn_valid_r <= 1'b0;
            is_single_warp <= 1'b0;
        end else begin
            is_single_warp <= (active_warps_cnt == $bits(active_warps_cnt)'(1));
            if (act_wspawn) begin
                wspawn_valid_r <= 1'b1;
            end
            if (wspawn_apply) begin
                wspawn_valid_r <= 1'b0;
            end
        end
        if (act_wspawn) begin
            wspawn_r          <= act_wspawn_data;
            wspawn_wid_r      <= act_wid;
            wspawn_mscratch_r <= act_mscratch;
        end
    end

    assign wspawn_wmask    = wspawn_r.wmask;
    assign wspawn_pc       = wspawn_r.pc;
    assign wspawn_wid      = wspawn_wid_r;
    assign wspawn_mscratch = wspawn_mscratch_r;
    assign wspawn_pending  = wspawn_valid_r;

    // warp retirement

    assign warp_done     = act_exit;
    assign warp_done_wid = act_wid;

    // completion

    wire [NUM_REQS-1:0] rsp_valid = NUM_REQS'(req_valid && req.has_rsp) << req_sel;
    wire [NUM_REQS-1:0] rsp_valid_out;
    wire                rsp_phase_out;
    if (DIRECT) begin : g_rsp_direct
        assign rsp_valid_out = rsp_valid;
        assign rsp_phase_out = bar_read_phase;
    end else begin : g_rsp_reg
        reg [NUM_REQS-1:0] rsp_valid_r;
        reg                rsp_phase_r;
        always @(posedge clk) begin
            if (reset) begin
                rsp_valid_r <= '0;
            end else begin
                rsp_valid_r <= rsp_valid;
            end
            rsp_phase_r <= bar_read_phase;
        end
        assign rsp_valid_out = rsp_valid_r;
        assign rsp_phase_out = rsp_phase_r;
    end

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_rsp
        assign warp_shared_if[i].rsp_valid = rsp_valid_out[i];
        assign warp_shared_if[i].rsp_phase = rsp_phase_out;
    end

`ifdef DBG_TRACE_PIPELINE
    always @(posedge clk) begin
        if (req_valid) begin
            `TRACE(1, ("%t: %s request: sel=%0d, kind=%0d, wid=%0d\n", $time, INSTANCE_ID, req_sel, req.kind, req.wid))
        end
    end
`endif

endmodule
