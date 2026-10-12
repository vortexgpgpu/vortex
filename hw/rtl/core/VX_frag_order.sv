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

// Primitive-order merge. Each fragment warp takes a ticket at launch, and the
// serving counter admits one ticket at a time into the merge between FRAG_WAIT
// and FRAG_RELEASE, so overlapping fragments merge in the order the rasterizer
// launched them.
//
// A warp releases at most once and before it exits (a warp that never waits
// releases at exit), so the unreleased tickets never outnumber the warps: the
// ticket -> warp map holds one entry per warp and both counters wrap at it.
// A release at exit can be out of ticket order, which is harmless because a
// shader that skips the wait never has a warp waiting.

module VX_frag_order import VX_gpu_pkg::*; (
    input wire                  clk,
    input wire                  reset,

    // a fragment warp was launched
    input wire                  launch_valid,
    input wire [NW_WIDTH-1:0]   launch_wid,

    // warp-control events, at most one per cycle
    input wire                  wait_valid,
    input wire                  release_valid,
    input wire                  exit_valid,
    input wire [NW_WIDTH-1:0]   wid,

    // admit a parked warp to the merge
    output wire                 grant_valid,
    output wire [NW_WIDTH-1:0]  grant_wid
);
    localparam NW = `VX_CFG_NUM_WARPS;

    reg [NW-1:0][NW_WIDTH-1:0] ticket_wid_r;
    reg [NW_WIDTH-1:0] launch_ptr_r, serving_ptr_r;
    reg [NW-1:0] held_r, waiting_r;

    wire [NW_WIDTH-1:0] serving_ptr_n = (serving_ptr_r == NW_WIDTH'(NW-1)) ? '0 : (serving_ptr_r + NW_WIDTH'(1));
    wire [NW_WIDTH-1:0] launch_ptr_n  = (launch_ptr_r == NW_WIDTH'(NW-1)) ? '0 : (launch_ptr_r + NW_WIDTH'(1));

    wire do_release = release_valid || (exit_valid && held_r[wid]);
    wire turn_now   = wait_valid && (ticket_wid_r[serving_ptr_r] == wid);

    // The warp holding the next ticket, if it already parked in its wait. The
    // next slot may never have been written, so nothing is read from it unless
    // some warp waits.
    wire [NW_WIDTH-1:0] next_wid = ticket_wid_r[serving_ptr_n];
    wire wake = do_release && (| waiting_r) && waiting_r[next_wid];

    assign grant_valid = turn_now || wake;
    assign grant_wid   = turn_now ? wid : next_wid;

    always @(posedge clk) begin
        if (reset) begin
            launch_ptr_r  <= '0;
            serving_ptr_r <= '0;
            held_r        <= '0;
            waiting_r     <= '0;
        end else begin
            if (launch_valid) begin
                launch_ptr_r <= launch_ptr_n;
                held_r[launch_wid] <= 1'b1;
            end
            if (do_release) begin
                serving_ptr_r <= serving_ptr_n;
                held_r[wid] <= 1'b0;
            end
            if (wait_valid && !turn_now) begin
                waiting_r[wid] <= 1'b1;
            end
            if (wake) begin
                waiting_r[next_wid] <= 1'b0;
            end
        end
    end

    always @(posedge clk) begin
        if (launch_valid) begin
            ticket_wid_r[launch_ptr_r] <= launch_wid;
        end
    end

    `RUNTIME_ASSERT(~(wait_valid || release_valid) || held_r[wid],
        ("%t: *** frag order op from a warp holding no ticket: wid=%0d", $time, wid))
    `RUNTIME_ASSERT(~release_valid || (ticket_wid_r[serving_ptr_r] == wid),
        ("%t: *** FRAG_RELEASE out of turn: wid=%0d", $time, wid))
    `RUNTIME_ASSERT(~(exit_valid && held_r[wid]) || (waiting_r == '0),
        ("%t: *** fragment warp exited unreleased while a warp waits: wid=%0d", $time, wid))
    `RUNTIME_ASSERT(~launch_valid || ~held_r[launch_wid],
        ("%t: *** fragment launch into a warp still holding a ticket: wid=%0d", $time, launch_wid))

endmodule
