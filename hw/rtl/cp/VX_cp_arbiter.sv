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

// ============================================================================
// VX_cp_arbiter — round-robin arbiter over N bidders (KMU, DMA, DCR, event).
// Grants at most one bidder per cycle, advancing the pointer past the winner.
// Grant lasts one cycle; no in-flight tracking. `bid_priority` is reserved and
// currently unused. Thin wrapper over the library VX_rr_arbiter.
// ============================================================================

module VX_cp_arbiter
  import VX_cp_pkg::*;
#(
  parameter int N = 1
)(
  input  wire                  clk,
  input  wire                  reset,

  input  wire                  bid_valid    [N],
  input  wire [1:0]            bid_priority [N],
  output logic                 bid_grant    [N]
);
  wire [N-1:0] requests;
  wire [N-1:0] grant_onehot;

  for (genvar i = 0; i < N; ++i) begin : g_ports
    assign requests[i]  = bid_valid[i];
    assign bid_grant[i] = grant_onehot[i];
    `UNUSED_VAR (bid_priority[i])
  end

  // grant_ready tied high: consume every grant so the pointer advances each
  // cycle a bidder is served (one-cycle, non-sticky grant).
  VX_rr_arbiter #(
    .NUM_REQS (N)
  ) rr_arb (
    .clk          (clk),
    .reset        (reset),
    .requests     (requests),
    `UNUSED_PIN   (grant_index),
    .grant_onehot (grant_onehot),
    `UNUSED_PIN   (grant_valid),
    .grant_ready  (1'b1)
  );

endmodule
