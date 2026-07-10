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

// Fixed-function window read port. A consumer FF (TEX / OM) presents a warp +
// thread base and NUM_PORTS slot indices; VX_gfx_window returns the per-(port,
// lane) record words on the FOLLOWING cycle. The window's storage is a memory,
// so the slot is an address and the read is synchronous: a consumer drives its
// slots one cycle before it samples them. There is no back-pressure — each port
// is served by its own RAM mirror and reads every cycle.
interface VX_gfx_win_rd_if import VX_gpu_pkg::*, VX_gfx_window_pkg::*; #(
    parameter NUM_LANES = 1,
    parameter NUM_PORTS = 1
) ();
    typedef struct packed {
        logic [NW_WIDTH-1:0]                      wid;    // window warp index
        logic [`CLOG2(`VX_CFG_NUM_THREADS)-1:0]   tbase;  // thread base within the warp
        logic [NUM_PORTS-1:0][GFXW_SLOT_BITS-1:0] slot;   // per-port window slot
    } req_t;

    req_t                                      req;    // read request
    logic [NUM_PORTS-1:0][NUM_LANES-1:0][31:0] data;   // per-(port, lane) response word

    // master = the FF consumer (TEX/OM) that issues the read
    modport master (
        output req,
        input  data
    );

    // slave = VX_gfx_window (the register file that serves the read)
    modport slave (
        input  req,
        output data
    );

endinterface
