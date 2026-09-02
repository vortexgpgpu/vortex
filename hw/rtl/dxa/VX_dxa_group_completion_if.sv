// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

`ifdef VX_CFG_EXT_DXA_GROUP_ENABLE

// Typed ready/valid link carrying one source-read completion event. The
// core-id header is consumed by the completion transport.
interface VX_dxa_group_completion_if import VX_gpu_pkg::*, VX_dxa_pkg::*; ();

    logic                        valid;
    logic [`UP(NC_BITS)-1:0]     core_id;
    dxa_group_completion_t       data;
    logic                        ready;

    modport master (
        output valid,
        output core_id,
        output data,
        input  ready
    );

    modport slave (
        input  valid,
        input  core_id,
        input  data,
        output ready
    );

endinterface

`endif // VX_CFG_EXT_DXA_GROUP_ENABLE
