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

// Fragment window-seed write port. The per-core fragment dispatcher
// (VX_raster_dispatch) stages one per-(warp, lane) record word per cycle into the
// gfx register window (VX_gfx_window). The sink is a register-file write port that
// commits every cycle, so this is a pure valid + payload push — there is no
// ready/back-pressure by construction (the window always accepts a write).
interface VX_gfx_win_wr_if import VX_gpu_pkg::*, VX_gfx_window_pkg::*; #(
    parameter NUM_LANES = 1
) ();
    typedef struct packed {
        logic [NW_WIDTH-1:0]                     wid;    // window warp index (slot)
        logic [`CLOG2(`VX_CFG_NUM_THREADS)-1:0]  tbase;  // thread base within the warp
        logic [NUM_LANES-1:0]                    mask;   // per-lane write mask
        logic [GFXW_SLOT_BITS-1:0]               slot;   // window slot (record word)
        logic [NUM_LANES-1:0][31:0]              data;   // per-lane record word
    } wr_data_t;

    logic       valid;
    wr_data_t   data;

    modport master (
        output valid,
        output data
    );

    modport slave (
        input  valid,
        input  data
    );

endinterface
