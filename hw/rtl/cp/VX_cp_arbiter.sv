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
// VX_cp_arbiter — generic round-robin arbiter over N bidders.
//
// Instantiated 3x in VX_cp_core (one per shared resource: KMU, DMA, DCR).
// On any given cycle, picks at most one bidder whose `valid` is asserted,
// rotating fairness across calls. Grant lasts a single cycle; the granted
// CPE is expected to hold its bid until the resource completes (the
// per-resource consumer module signals completion through a separate
// path; this arbiter does not track in-flight requests).
//
// Priority is honored only as a "high-priority bidders are visited first
// in the rotation" hint, not as strict preemption. This keeps the
// implementation small and avoids starvation guarantees beyond plain
// round-robin.
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

  // Rotating pointer to the bidder that gets first look this cycle.
  // For N=1, $clog2(N)=0, so PTR_W collapses to 1 (we still need at least
  // one bit to hold the value 0).
  localparam int PTR_W = (N > 1) ? $clog2(N) : 1;
  // SUM_W is one bit wider than PTR_W so (rr_ptr + N - 1) fits without
  // wrap, even when N is a power of 2 (PTR_W'(N) would truncate to 0
  // and break the modulo).
  localparam int SUM_W = PTR_W + 1;

  logic [PTR_W-1:0] rr_ptr;
  logic [PTR_W-1:0] winner;
  logic             any_grant;

  always_comb begin
    winner    = '0;
    any_grant = 1'b0;
    bid_grant = '{default: 1'b0};

    if (N == 1) begin
      if (bid_valid[0]) begin
        bid_grant[0] = 1'b1;
        winner       = '0;
        any_grant    = 1'b1;
      end
    end else begin
      // One-pass scan: starting at rr_ptr, find the first valid bidder.
      // Sum in SUM_W bits then conditionally subtract N (faster than
      // synthesizing a divider and dodges the PTR_W'(N)==0 hazard).
      for (int unsigned i = 0; i < N; ++i) begin
        logic [SUM_W-1:0]  sum;
        logic [PTR_W-1:0]  idx;
        sum = SUM_W'({1'b0, rr_ptr}) + SUM_W'(i);
        idx = (sum >= SUM_W'(N)) ? PTR_W'(sum - SUM_W'(N))
                                 : PTR_W'(sum);
        if (!any_grant && bid_valid[idx]) begin
          bid_grant[idx] = 1'b1;
          winner         = idx;
          any_grant      = 1'b1;
        end
      end
    end

  end

  // Round-robin only in v1 — priority is reserved for a future eligibility
  // pre-filter pass. Suppress unused-bit warnings per-element so the macro
  // sees a packed logic instead of the unpacked array.
  generate
    for (genvar gi = 0; gi < N; ++gi) begin : g_unused_prio
      `UNUSED_VAR (bid_priority[gi])
    end
  endgenerate

  // Advance the round-robin pointer one past the winner so the next
  // cycle starts the scan after the bidder we just served. Same
  // wrap-by-subtract trick as the scan above.
  always_ff @(posedge clk) begin
    if (reset) begin
      rr_ptr <= '0;
    end else if (any_grant) begin
      if (N == 1) begin
        rr_ptr <= '0;
      end else begin
        logic [SUM_W-1:0] nxt;
        nxt = SUM_W'({1'b0, winner}) + SUM_W'(1);
        rr_ptr <= (nxt >= SUM_W'(N)) ? PTR_W'(nxt - SUM_W'(N))
                                     : PTR_W'(nxt);
      end
    end
  end

endmodule : VX_cp_arbiter
