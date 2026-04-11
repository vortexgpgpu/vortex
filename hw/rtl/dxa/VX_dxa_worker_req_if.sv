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

// DXA issue interface: carries request data + descriptor from dispatch to worker.
interface VX_dxa_worker_req_if import VX_gpu_pkg::*, VX_dxa_pkg::*; ();

    logic          valid;
    dxa_req_data_t req_data;
    dxa_desc_t     desc_data;
    logic          ready;

    modport master (
        output valid,
        output req_data,
        output desc_data,
        input  ready
    );

    modport slave (
        input  valid,
        input  req_data,
        input  desc_data,
        output ready
    );

endinterface
