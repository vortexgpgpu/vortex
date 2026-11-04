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

`include "VX_platform.vh"

// The five VX_generic_arbiter policies with STICKY=1 (GTO is greedy by
// construction), driven by the same requests, for the sticky-hold checks in
// main.cpp.
module VX_sticky_arb_top #(
    parameter NUM_REQS     = 16,
    parameter LOG_NUM_REQS = `LOG2UP(NUM_REQS)
) (
    input  wire                     clk,
    input  wire                     reset,
    input  wire [NUM_REQS-1:0]      requests,
    input  wire                     grant_ready,
    output wire [4:0]               grant_valid,
    output wire [LOG_NUM_REQS-1:0]  grant_index_p,
    output wire [LOG_NUM_REQS-1:0]  grant_index_r,
    output wire [LOG_NUM_REQS-1:0]  grant_index_m,
    output wire [LOG_NUM_REQS-1:0]  grant_index_c,
    output wire [LOG_NUM_REQS-1:0]  grant_index_g
);
    wire [4:0][NUM_REQS-1:0] grant_onehot;
    `UNUSED_VAR (grant_onehot)

    VX_generic_arbiter #(.NUM_REQS(NUM_REQS), .TYPE("P"), .STICKY(1)) arb_p (
        .clk(clk), .reset(reset), .requests(requests), .grant_index(grant_index_p),
        .grant_onehot(grant_onehot[0]), .grant_valid(grant_valid[0]), .grant_ready(grant_ready));
    VX_generic_arbiter #(.NUM_REQS(NUM_REQS), .TYPE("R"), .STICKY(1)) arb_r (
        .clk(clk), .reset(reset), .requests(requests), .grant_index(grant_index_r),
        .grant_onehot(grant_onehot[1]), .grant_valid(grant_valid[1]), .grant_ready(grant_ready));
    VX_generic_arbiter #(.NUM_REQS(NUM_REQS), .TYPE("M"), .STICKY(1)) arb_m (
        .clk(clk), .reset(reset), .requests(requests), .grant_index(grant_index_m),
        .grant_onehot(grant_onehot[2]), .grant_valid(grant_valid[2]), .grant_ready(grant_ready));
    VX_generic_arbiter #(.NUM_REQS(NUM_REQS), .TYPE("C"), .STICKY(1)) arb_c (
        .clk(clk), .reset(reset), .requests(requests), .grant_index(grant_index_c),
        .grant_onehot(grant_onehot[3]), .grant_valid(grant_valid[3]), .grant_ready(grant_ready));
    VX_generic_arbiter #(.NUM_REQS(NUM_REQS), .TYPE("G")) arb_g (
        .clk(clk), .reset(reset), .requests(requests), .grant_index(grant_index_g),
        .grant_onehot(grant_onehot[4]), .grant_valid(grant_valid[4]), .grant_ready(grant_ready));

endmodule
