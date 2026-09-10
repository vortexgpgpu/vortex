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
// Single raiser (the RTU unit), no trap, no PC redirect — a plain warp unlock, and
// nothing else. The issue stage knows nothing about the RTU.

interface VX_sched_unlock_if import VX_gpu_pkg::*; ();

    logic                valid;      // pulse: release the wstall'd warp
    logic [NW_WIDTH-1:0] wid;

    modport master (
        output valid,
        output wid
    );

    modport slave (
        input  valid,
        input  wid
    );

endinterface
