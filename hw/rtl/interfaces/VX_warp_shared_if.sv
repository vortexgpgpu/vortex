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

// A core-wide warp-control request and its completion. A requester waits on at
// most one request with has_rsp set; rsp_valid completes it and carries the
// barrier phase. Other requests are posted and get no completion.
interface VX_warp_shared_if import VX_gpu_pkg::*; ();

    logic               req_valid;
    warp_shared_req_t   req_data;
    logic               req_ready;

    logic               rsp_valid;
    logic               rsp_phase;

    modport master (
        output req_valid,
        output req_data,
        input  req_ready,
        input  rsp_valid,
        input  rsp_phase
    );

    modport slave (
        input  req_valid,
        input  req_data,
        output req_ready,
        output rsp_valid,
        output rsp_phase
    );

endinterface
