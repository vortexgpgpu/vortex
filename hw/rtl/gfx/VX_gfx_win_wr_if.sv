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

// Graphics-window write port. A producer — the fragment dispatcher's per-lane
// record seed (VX_raster_dispatch), or a TEX texel writeback — pushes one
// per-(warp, lane) slot word per cycle into VX_gfx_window. The window's storage
// has a single write port shared by several producers, so a push may be held:
// `ready` reports the grant. The raster seed holds top write priority and
// therefore sees `ready` constantly asserted.
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
    logic       ready;
    wr_data_t   data;

    modport master (
        output valid,
        input  ready,
        output data
    );

    modport slave (
        input  valid,
        output ready,
        input  data
    );

endinterface
