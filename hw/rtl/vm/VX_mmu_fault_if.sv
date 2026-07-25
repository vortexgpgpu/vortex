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

// First-fault report sideband. A translation fault (structural, from the
// walker; or permission, from an L1 stage) raises `valid` for one cycle with
// the faulting virtual address, access kind, and AMO intent. The fault-latch
// surface records the first such report and reads it back to the host.
interface VX_mmu_fault_if #(
    parameter VA_WIDTH = `VX_CFG_XLEN
) ();

    logic                valid;
    logic [VA_WIDTH-1:0] va;
    logic [1:0]          access;
    logic                amo;

    modport master (
        output valid,
        output va,
        output access,
        output amo
    );

    modport slave (
        input  valid,
        input  va,
        input  access,
        input  amo
    );

endinterface
