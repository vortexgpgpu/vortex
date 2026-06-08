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

// VX_async_trap_if — asynchronous M-mode trap raised by a fixed-function unit
// (the RTU on a shader-callback yield) on a warp that is NOT executing a trap
// instruction. The scheduler treats it like an ECALL trap entry — redirect the
// warp PC to mtvec, snapshot the resume PC into mepc and the cause into mcause —
// but additionally narrows the warp's tmask to `tmask` (the yielding lanes),
// saving the pre-trap tmask into mscratch_tmask for MRET to restore, and
// resumes the warp if it was suspended on the parked macro-op the trap takes
// over. Single raiser (the RTU unit), single consumer (the scheduler).

interface VX_async_trap_if import VX_gpu_pkg::*; ();

    logic                              valid;   // trap entry (redirect to mtvec)
    logic                              unlock;  // resume the wstall'd trace warp
    logic [NW_WIDTH-1:0]               wid;
    logic [`VX_CFG_XLEN-1:0]           cause;   // VX_TRAP_CAUSE_*
    logic [`VX_CFG_NUM_THREADS-1:0]    tmask;   // narrowed (yielding) lane mask

    modport master (
        output valid,
        output unlock,
        output wid,
        output cause,
        output tmask
    );

    modport slave (
        input  valid,
        input  unlock,
        input  wid,
        input  cause,
        input  tmask
    );

endinterface
