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

// Cluster-TLB miss station. Misses on the same VPN — from any client, any
// core — dedup onto one entry and one walk; every attached requester is
// answered when the fill returns. Each entry keeps a short requester FIFO
// (the id the fill routes back to), issues one walk, then drains its
// requesters one response per cycle.
module VX_tlb_l2_mshr import VX_tlb_pkg::*; #(
    parameter MSHR_SIZE  = `VX_CFG_L2_TLB_MSHR_SIZE,
    parameter REQR_DEPTH = 4,
    parameter ID_WIDTH   = 6
) (
    input wire clk,
    input wire reset,

    // Allocate / append (from a lookup miss).
    input  wire                       alloc_valid,
    input  wire [TLB_VPN_WIDTH-1:0]   alloc_vpn,
    input  tlb_access_e               alloc_acc,
    input  wire                       alloc_amo,
    input  wire [`UP(ID_WIDTH)-1:0]   alloc_id,
    output wire                       alloc_ready,

    // Walk issue.
    output wire                       ptw_req_valid,
    output wire [`CLOG2(MSHR_SIZE)-1:0] ptw_req_slot,
    output tlb_access_e               ptw_req_acc,
    output wire                       ptw_req_amo,
    output wire [TLB_VPN_WIDTH-1:0]   ptw_req_vpn,
    input  wire                       ptw_req_ready,

    // Walk fill.
    input  wire                       ptw_rsp_valid,
    input  wire [`CLOG2(MSHR_SIZE)-1:0] ptw_rsp_slot,
    input  wire                       ptw_rsp_fault,
    input  wire [TLB_LEVEL_WIDTH-1:0] ptw_rsp_level,
    input  wire [TLB_PPN_WIDTH-1:0]   ptw_rsp_ppn,
    input  wire [TLB_FLAGS_WIDTH-1:0] ptw_rsp_flags,
    output wire                       ptw_rsp_ready,

    // Array install (non-faulting fill).
    output wire                       install_valid,
    output tlb_entry_t                install_entry,

    // Client drain response.
    output wire                       rsp_valid,
    output wire [`UP(ID_WIDTH)-1:0]   rsp_id,
    output wire                       rsp_fault,
    output wire [TLB_LEVEL_WIDTH-1:0] rsp_level,
    output wire [TLB_PPN_WIDTH-1:0]   rsp_ppn,
    output wire [TLB_FLAGS_WIDTH-1:0] rsp_flags,
    input  wire                       rsp_ready,

    input  wire                       flush,
    output wire                       empty
);
    localparam SLOT_W = `CLOG2(MSHR_SIZE);
    localparam ID_W   = `UP(ID_WIDTH);
    localparam SIZE_W = `CLOG2(REQR_DEPTH + 1);

    reg [MSHR_SIZE-1:0]         valid_r;
    reg [MSHR_SIZE-1:0]         issued_r;
    reg [MSHR_SIZE-1:0]         filling_r;
    reg [TLB_VPN_WIDTH-1:0]     vpn_r    [MSHR_SIZE];
    tlb_access_e                acc_r    [MSHR_SIZE];
    reg [MSHR_SIZE-1:0]         amo_r;
    reg [MSHR_SIZE-1:0]         ffault_r;
    reg [TLB_LEVEL_WIDTH-1:0]   flevel_r [MSHR_SIZE];
    reg [TLB_PPN_WIDTH-1:0]     fppn_r   [MSHR_SIZE];
    reg [TLB_FLAGS_WIDTH-1:0]   fflags_r [MSHR_SIZE];

    // Per-entry requester FIFO (the fill routes back to each queued id).
    wire [MSHR_SIZE-1:0]           rq_push;
    wire [MSHR_SIZE-1:0]           rq_pop;
    wire [MSHR_SIZE-1:0][ID_W-1:0] rq_head;
    wire [MSHR_SIZE-1:0]           rq_empty;
    wire [MSHR_SIZE-1:0]           rq_full;
    wire [MSHR_SIZE-1:0][SIZE_W-1:0] rq_size;

    for (genvar i = 0; i < MSHR_SIZE; ++i) begin : g_reqfifo
        VX_fifo_queue #(
            .DATAW  (ID_W),
            .DEPTH  (REQR_DEPTH),
            .LUTRAM (1)
        ) reqfifo (
            .clk       (clk),
            .reset     (reset),
            .push      (rq_push[i]),
            .pop       (rq_pop[i]),
            .data_in   (alloc_id),
            .data_out  (rq_head[i]),
            .empty     (rq_empty[i]),
            `UNUSED_PIN (alm_empty),
            .full      (rq_full[i]),
            `UNUSED_PIN (alm_full),
            .size      (rq_size[i])
        );
    end

    // ---------------------------------------------------------------------
    // Allocate / append
    // ---------------------------------------------------------------------
    wire [MSHR_SIZE-1:0] vpn_match;
    wire [MSHR_SIZE-1:0] free_vec;
    for (genvar i = 0; i < MSHR_SIZE; ++i) begin : g_match
        assign vpn_match[i] = valid_r[i] && (vpn_r[i] == alloc_vpn);
        assign free_vec[i]  = ~valid_r[i];
    end
    wire has_match = (| vpn_match);
    wire has_free  = (| free_vec);

    reg [SLOT_W-1:0] match_idx, free_idx;
    always @(*) begin
        match_idx = '0;
        free_idx  = '0;
        for (int i = MSHR_SIZE-1; i >= 0; --i) begin
            if (vpn_match[i]) begin
                match_idx = SLOT_W'(i);
            end
            if (free_vec[i]) begin
                free_idx = SLOT_W'(i);
            end
        end
    end

    wire [SLOT_W-1:0] alloc_target = has_match ? match_idx : free_idx;
    assign alloc_ready = has_match ? ~rq_full[match_idx] : has_free;
    wire alloc_fire = alloc_valid && alloc_ready;

    // ---------------------------------------------------------------------
    // Walk issue
    // ---------------------------------------------------------------------
    wire [MSHR_SIZE-1:0] issue_cand = valid_r & ~issued_r;
    wire has_issue = (| issue_cand);
    reg [SLOT_W-1:0] issue_idx;
    always @(*) begin
        issue_idx = '0;
        for (int i = MSHR_SIZE-1; i >= 0; --i) begin
            if (issue_cand[i]) begin
                issue_idx = SLOT_W'(i);
            end
        end
    end
    assign ptw_req_valid = has_issue;
    assign ptw_req_slot  = issue_idx;
    assign ptw_req_acc   = acc_r[issue_idx];
    assign ptw_req_amo   = amo_r[issue_idx];
    assign ptw_req_vpn   = vpn_r[issue_idx];
    wire issue_fire = ptw_req_valid && ptw_req_ready;

    // ---------------------------------------------------------------------
    // Fill
    // ---------------------------------------------------------------------
    assign ptw_rsp_ready = 1'b1;
    wire fill_fire = ptw_rsp_valid && ptw_rsp_ready;

    assign install_valid = fill_fire && ~ptw_rsp_fault;
    assign install_entry = '{
        level: ptw_rsp_level,
        vpn:   vpn_r[ptw_rsp_slot],
        ppn:   ptw_rsp_ppn,
        flags: ptw_rsp_flags
    };

    // ---------------------------------------------------------------------
    // Drain requesters, one response per cycle
    // ---------------------------------------------------------------------
    wire has_drain = (| filling_r);
    reg [SLOT_W-1:0] drain_idx;
    always @(*) begin
        drain_idx = '0;
        for (int i = MSHR_SIZE-1; i >= 0; --i) begin
            if (filling_r[i]) begin
                drain_idx = SLOT_W'(i);
            end
        end
    end

    assign rsp_valid = has_drain && ~rq_empty[drain_idx];
    assign rsp_id    = rq_head[drain_idx];
    assign rsp_fault = ffault_r[drain_idx];
    assign rsp_level = flevel_r[drain_idx];
    assign rsp_ppn   = fppn_r[drain_idx];
    assign rsp_flags = fflags_r[drain_idx];
    wire rsp_fire = rsp_valid && rsp_ready;

    // FIFO push/pop per entry.
    for (genvar i = 0; i < MSHR_SIZE; ++i) begin : g_reqfifo_ctrl
        assign rq_push[i] = alloc_fire && (alloc_target == SLOT_W'(i));
        assign rq_pop[i]  = rsp_fire && (drain_idx == SLOT_W'(i));
    end

    // An entry frees once its last requester drains and no new one arrives.
    wire drain_last = rsp_fire && (rq_size[drain_idx] == SIZE_W'(1)) && ~rq_push[drain_idx];

    // ---------------------------------------------------------------------
    // State update
    // ---------------------------------------------------------------------
    always @(posedge clk) begin
        if (reset || flush) begin
            valid_r   <= '0;
            issued_r  <= '0;
            filling_r <= '0;
        end else begin
            if (alloc_fire && ~has_match) begin
                valid_r[free_idx]   <= 1'b1;
                issued_r[free_idx]  <= 1'b0;
                filling_r[free_idx] <= 1'b0;
            end
            if (issue_fire) begin
                issued_r[issue_idx] <= 1'b1;
            end
            if (fill_fire) begin
                filling_r[ptw_rsp_slot] <= 1'b1;
            end
            if (drain_last) begin
                valid_r[drain_idx]   <= 1'b0;
                issued_r[drain_idx]  <= 1'b0;
                filling_r[drain_idx] <= 1'b0;
            end
        end
    end

    always @(posedge clk) begin
        if (alloc_fire && ~has_match) begin
            vpn_r[free_idx] <= alloc_vpn;
            acc_r[free_idx] <= alloc_acc;
            amo_r[free_idx] <= alloc_amo;
        end
        if (fill_fire) begin
            ffault_r[ptw_rsp_slot] <= ptw_rsp_fault;
            flevel_r[ptw_rsp_slot] <= ptw_rsp_level;
            fppn_r[ptw_rsp_slot]   <= ptw_rsp_ppn;
            fflags_r[ptw_rsp_slot] <= ptw_rsp_flags;
        end
    end

    assign empty = ~(| valid_r);

endmodule
