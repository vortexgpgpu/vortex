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

// Narrow TLB miss/fill fabric: a request carries the missing VPN plus the
// access kind and AMO write intent; the response carries the translation,
// its page level, and a fault flag. Two orders of magnitude narrower than
// a VX_mem_bus_if lane, which keeps the socket/cluster arb trees cheap.
interface VX_tlb_bus_if import VX_tlb_pkg::*; #(
    parameter ID_WIDTH = 4
) ();

    typedef struct packed {
        logic [`UP(ID_WIDTH)-1:0]   id;
        tlb_access_e                access;
        logic                       amo;
        logic [TLB_VPN_WIDTH-1:0]   vpn;
    } req_data_t;

    typedef struct packed {
        logic [`UP(ID_WIDTH)-1:0]   id;
        logic                       fault;
        logic [TLB_LEVEL_WIDTH-1:0] level;
        logic [TLB_PPN_WIDTH-1:0]   ppn;
        logic [TLB_FLAGS_WIDTH-1:0] flags;
    } rsp_data_t;

    logic      req_valid;
    req_data_t req_data;
    logic      req_ready;

    logic      rsp_valid;
    rsp_data_t rsp_data;
    logic      rsp_ready;

    modport master (
        output req_valid,
        output req_data,
        input  req_ready,

        input  rsp_valid,
        input  rsp_data,
        output rsp_ready
    );

    modport slave (
        input  req_valid,
        input  req_data,
        output req_ready,

        output rsp_valid,
        output rsp_data,
        input  rsp_ready
    );

endinterface
