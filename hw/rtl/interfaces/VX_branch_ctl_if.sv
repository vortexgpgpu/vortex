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

interface VX_branch_ctl_if import VX_gpu_pkg::*; ();

    wire                valid;
    wire [NW_WIDTH-1:0] wid;
    wire                taken;
    wire [PC_BITS-1:0]  dest;
    // Synchronous trap signalling (ECALL/EBREAK/MRET). When is_trap is set
    // `dest` carries the faulting PC (saved into mepc) and trap_cause the
    // mcause code; the scheduler redirects the warp to mtvec. is_mret
    // restores the warp PC from mepc. Both are mutually exclusive and
    // override the normal taken/dest path in the scheduler.
    wire                is_trap;
    wire                is_mret;
    wire [3:0]          trap_cause;

    modport master (
        output valid,
        output wid,
        output taken,
        output dest,
        output is_trap,
        output is_mret,
        output trap_cause
    );

    modport slave (
        input valid,
        input wid,
        input taken,
        input dest,
        input is_trap,
        input is_mret,
        input trap_cause
    );

endinterface
