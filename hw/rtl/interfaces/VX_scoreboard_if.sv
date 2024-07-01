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

interface VX_scoreboard_if import VX_gpu_pkg::*; ();

    typedef struct packed {
        logic [`UUID_WIDTH-1:0]     uuid;
        logic [ISSUE_WIS_W-1:0]     wis;
        logic [`NUM_THREADS-1:0]    tmask;
        logic [`PC_BITS-1:0]        PC;
        logic [`EX_BITS-1:0]        ex_type;
        logic [`INST_OP_BITS-1:0]   op_type;
        op_args_t                   op_args;
        logic                       wb;
        logic [`NR_BITS-1:0]        rd;
        logic [`NR_BITS-1:0]        rs1;
        logic [`NR_BITS-1:0]        rs2;
        logic [`NR_BITS-1:0]        rs3;
    } data_t;

    logic  valid;
    data_t data;
    logic  ready;

    modport master (
        output valid,
        output data,
        input  ready
    );

    modport slave (
        input  valid,
        input  data,
        output ready
    );

endinterface
