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

// VX_sched_unlock_if — a fixed-function unit releases a warp it holds wstall'd.
// The RTU's TRACE macro-op suspends its warp at decode until the traversal's
// first response retires the arm; the gfx window pulses `valid` with the warp
// id at that retire so the scheduler clears the warp's stall bit and it can
// proceed to WAIT. Single raiser (the gfx window / RTU), single consumer (the
// scheduler). No trap, no PC redirect — a plain warp unlock.

interface VX_sched_unlock_if import VX_gpu_pkg::*; ();

    logic                valid;   // pulse: release the wstall'd warp
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
