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

interface VX_result_if import VX_gpu_pkg::*; #(
    parameter NUM_LANES = 1
) ();

    typedef struct packed {
        logic [UUID_WIDTH-1:0]           uuid;
        logic [NW_WIDTH-1:0]             wid;
        logic [NCTA_WIDTH-1:0]           cta_id;
        logic [NUM_LANES-1:0]            tmask;
        logic [`LOG2UP(`VX_CFG_NUM_THREADS / NUM_LANES)-1:0] pid;
        logic                            sop;
        logic                            eop;
        logic [PC_BITS-1:0]              PC;
        logic                            wb;
        logic [NUM_XREGS-1:0]            wr_xregs;
        logic [NUM_REGS_BITS-1:0]        rd;
        logic [BYTESEL_BITS-1:0]         bytesel;
    } header_t;

    typedef struct packed {
        header_t                         header;
        logic [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] data;
    } data_t;

    `STATIC_ASSERT ($bits(header_t) == exec_header_bits(NUM_LANES),
        ("exec_header_bits() disagrees with header_t"))
    `STATIC_ASSERT ($bits(data_t) == result_data_bits(NUM_LANES),
        ("result_data_bits() disagrees with data_t"))

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
