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

interface VX_sched_csr_if import VX_gpu_pkg::*; ();

    wire [PERF_CTR_BITS-1:0]        cycles;
    wire [PERF_CTR_BITS-1:0]        instret;
    wire [`NUM_WARPS-1:0]           active_warps;
    wire [`NUM_WARPS-1:0][`NUM_THREADS-1:0] thread_masks;

    modport master (
        output cycles,
        output instret,
        output active_warps,
        output thread_masks
    );

    modport slave (
        input  cycles,
        input  instret,
        input  active_warps,
        input  thread_masks
    );

endinterface
