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

interface VX_warp_ctl_if import VX_gpu_pkg::*; ();

    wire        valid;
    wire [NW_WIDTH-1:0] wid;
    tmc_t       tmc;
    wspawn_t    wspawn;
    split_t     split;
    join_t      sjoin;
    barrier_t   barrier;

    wire [NW_WIDTH-1:0] dvstack_wid;
    wire [DV_STACK_SIZEW-1:0] dvstack_ptr;

    // Async barrier token loopup
    wire [NB_WIDTH-1:0] barrier_id_rd; // barrier ID to read token from
    wire [`XLEN-1:0] arrive_token; // current generation (token) returned

    modport master (
        output valid,
        output wid,
        output wspawn,
        output tmc,
        output split,
        output sjoin,
        output barrier,

        output barrier_id_rd,
        input  arrive_token,

        output dvstack_wid,
        input  dvstack_ptr
    );

    modport slave (
        input valid,
        input wid,
        input wspawn,
        input tmc,
        input split,
        input sjoin,
        input barrier,

        input barrier_id_rd,
        output arrive_token,

        input dvstack_wid,
        output dvstack_ptr
    );

endinterface
