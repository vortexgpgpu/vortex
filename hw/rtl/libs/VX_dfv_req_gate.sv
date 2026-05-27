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

//==============================================================================
// VX_dfv_req_gate - DFV Request Path Gating Module
//==============================================================================
// This module implements symmetric request/ready gating for DFV (Design for
// Verification) backpressure injection. It acts as a transparent middleman
// that can block both directions of a ready/valid handshake simultaneously.
//
// Operation:
// - Normal mode (dfv_stall=0): Transparent passthrough
// - Stall mode (dfv_stall=1):
//   * Master sees slave_ready=0 (thinks slave not ready)
//   * Slave sees master_valid=0 (thinks master has no request)
//
// This prevents protocol violations where internal logic (that may have
// signal fanout relationships between req/rsp paths) sees inconsistent state.
//==============================================================================

module VX_dfv_req_gate (
    input  logic clk,
    input  logic reset,

    // DFV control
    input  logic dfv_enable,        // Global DFV enable
    input  logic dfv_stall,         // Stall injection signal

    // Master side (upstream - e.g., elastic buffer output)
    input  logic master_valid,      // Valid from master
    output logic master_ready,      // Ready to master (gated)

    // Slave side (downstream - e.g., cache input)
    output logic slave_valid,       // Valid to slave (gated)
    input  logic slave_ready        // Ready from slave
);
    `UNUSED_VAR(clk)
    //==========================================================================
    // Stall Enable Logic with Reset Handling
    //==========================================================================
    logic stall_enable;
    assign stall_enable = reset ? 1'b0 : (dfv_enable && dfv_stall);

    //==========================================================================
    // Symmetric Gating
    //==========================================================================

    // Gate ready signal going back to master
    // When stalled: master sees ready=0 (slave appears not ready)
    assign master_ready = stall_enable ? 1'b0 : slave_ready;

    // Gate valid signal going to slave
    // When stalled: slave sees valid=0 (master appears to have no request)
    assign slave_valid = stall_enable ? 1'b0 : master_valid;

endmodule
