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

interface VX_decode_if import VX_gpu_pkg::*; ();
    logic  valid;
    decode_t data;
    logic  ready;
`ifndef L1_ENABLE
    wire [`NUM_WARPS-1:0] ibuf_pop;
`endif

    modport master (
        output valid,
        output data,
        input  ready
    `ifndef L1_ENABLE
        , input ibuf_pop
    `endif
    );

    modport slave (
        input  valid,
        input  data,
        output ready
    `ifndef L1_ENABLE
        , output ibuf_pop
    `endif
    );

endinterface
