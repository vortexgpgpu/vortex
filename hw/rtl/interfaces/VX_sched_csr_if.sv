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

interface VX_sched_csr_if ();

    wire [`PERF_CTR_BITS-1:0] cycles;
    wire [`NUM_WARPS-1:0] active_warps;
    wire [`NUM_WARPS-1:0][`NUM_THREADS-1:0] thread_masks;
    wire alm_empty;
    wire [`NW_WIDTH-1:0] alm_empty_wid;
    wire unlock_warp;
    wire [`NW_WIDTH-1:0] unlock_wid;

    modport master (
        output cycles,
        output active_warps,
        output thread_masks,
        input  alm_empty_wid,
        output alm_empty,
        input  unlock_wid,        
        input  unlock_warp
    );

    modport slave (
        input  cycles,
        input  active_warps,
        input  thread_masks,
        output alm_empty_wid,
        input  alm_empty,
        output unlock_wid,
        output unlock_warp
    );

endinterface
