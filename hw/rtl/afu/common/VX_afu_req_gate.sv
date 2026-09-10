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

`include "vortex_afu.vh"

// ============================================================================
// VX_afu_req_gate — withhold new AXI requests without violating the protocol.
//
// The reset sequencer quiesces the AFU by stopping new AW/AR from leaving it,
// so the outstanding-transaction counters can reach zero in bounded time. The
// obvious way to write that is
//
//     assign out_valid = in_valid && !stop_req;
//
// and it is wrong. AXI4 §A3.2.1: once VALID is asserted it must remain
// asserted until the rising clock edge at which VALID and READY are both
// high. A master may not withdraw an offer because it changed its mind. The
// naive gate does exactly that whenever `stop_req` rises while a request is
// already being offered, and an interconnect is entitled to have latched that
// offer -- the Versal NoC does. The port is then permanently out of step with
// the slave: the transaction is neither accepted nor retractable, later reads
// on the same port never complete, and the drain counters (which only ever see
// the handshake) report the master busy forever.
//
// So the block decision is registered, and the register may only change while
// no offer is outstanding. A request already presented stays presented until
// it is accepted; the very next one is held off. `stop_req` therefore takes
// effect within one transaction rather than instantly, which is exactly what
// quiescing means and is bounded by the slave's own acceptance latency.
//
// The ready path is gated combinationally alongside so the upstream master
// does not see an acceptance the shell never made.
// ============================================================================

module VX_afu_req_gate (
    input  wire clk,
    input  wire reset,

    input  wire stop_req,    // withhold new requests

    input  wire in_valid,    // from the AFU-internal master
    output wire in_ready,
    output wire out_valid,   // to the shell
    input  wire out_ready
);
    reg blocked;

    // `out_valid && !out_ready` is an offer the shell has not taken yet.
    // Holding `blocked` across it is what keeps VALID from being withdrawn.
    always @(posedge clk) begin
        if (reset) begin
            blocked <= 1'b0;
        end else if (!(out_valid && !out_ready)) begin
            blocked <= stop_req;
        end
    end

    assign out_valid = in_valid && !blocked;
    assign in_ready  = out_ready && !blocked;

endmodule
