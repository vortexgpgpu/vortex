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

    wire wspawn_valid;
    wire tmc_valid;
    wire split_valid;
    wire sjoin_valid;
    wire bar_valid;

    wire [NW_WIDTH-1:0] wid;

    tmc_t     tmc;
    wspawn_t  wspawn;
    split_t   split;
    join_t    sjoin;
    barrier_t bar;

    // Return IPDOM stack address
    wire [NW_WIDTH-1:0] dvstack_wid;
    wire [DV_STACK_SIZEW-1:0] dvstack_ptr;

    // Return barrier phase
    wire [BAR_ADDR_W-1:0] bar_addr;
    wire bar_phase;

    modport master (
        output wspawn_valid,
        output tmc_valid,
        output split_valid,
        output sjoin_valid,
        output bar_valid,

        output wid,

        output wspawn,
        output tmc,
        output split,
        output sjoin,
        output bar,

        output bar_addr,
        input  bar_phase,

        output dvstack_wid,
        input  dvstack_ptr
    );

    modport slave (
        input wspawn_valid,
        input tmc_valid,
        input split_valid,
        input sjoin_valid,
        input bar_valid,

        input wid,

        input wspawn,
        input tmc,
        input split,
        input sjoin,
        input bar,

        input  bar_addr,
        output bar_phase,

        input dvstack_wid,
        output dvstack_ptr
    );

endinterface
