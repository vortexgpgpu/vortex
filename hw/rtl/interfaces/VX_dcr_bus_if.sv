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

interface VX_dcr_bus_if import VX_gpu_pkg::*; ();

    wire       req_valid;
    dcr_req_t  req_data;

    wire       rsp_valid;
    dcr_rsp_t  rsp_data;

    modport master (
        output req_valid,
        output req_data,

        input  rsp_valid,
        input  rsp_data
    );

    modport slave (
        input  req_valid,
        input  req_data,

        output rsp_valid,
        output rsp_data
    );

endinterface
