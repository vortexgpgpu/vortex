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

// VX_sched_unlock_if — the RTU releases the warps its ops wstall'd at decode.
//
//   valid/wid  — release the WARP. Both RTU ops that stall a warp are released by
//                the op itself retiring: a TRACE once its burst has handed the RTU
//                the ray, a WAIT once a record has landed and it returns the
//                status. Never by a traversal event, so the pulse cannot race the
//                stall it clears — decode set that stall strictly earlier.
//
//   trace_done — reopen the TRACE GATE. TRACE is not issued while this core's RTU
//                still holds a trace (VX_scoreboard), because the burst is atomic
//                and its arm would otherwise block inside the SFU while holding the
//                issue lock — freezing the very warp whose CONTINUE the RTU is
//                parked waiting for. The gate reopens on the TERMINAL record.
//
//                This is a RESOURCE CREDIT, not a warp unlock: it is consumed by
//                VX_issue, not the scheduler, and it only lives on this interface
//                because there was no other path back. It leaves when the RTU
//                exports the credit directly.
//
// Single raiser (the RTU unit), no trap, no PC redirect — plain front-end holds.

interface VX_sched_unlock_if import VX_gpu_pkg::*; ();

    logic                valid;      // pulse: release the wstall'd warp
    logic [NW_WIDTH-1:0] wid;
    logic                trace_done; // pulse: the RTU is free; reopen the trace gate

    modport master (
        output valid,
        output wid,
        output trace_done
    );

    modport slave (
        input  valid,
        input  wid,
        input  trace_done
    );

endinterface
