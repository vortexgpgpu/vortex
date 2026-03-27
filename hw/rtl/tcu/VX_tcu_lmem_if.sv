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

// Bank-parallel LMEM read port used by the TCU tile buffer.
// Master drives req_valid/req_data; slave drives req_ready/rsp_valid/rsp_data.
// Response arrives exactly one cycle after the request is accepted.

interface VX_tcu_lmem_if #(
    parameter NUM_BANKS       = 4,
    parameter BANK_ADDR_WIDTH = 12
) ();

    typedef struct packed {
        logic [BANK_ADDR_WIDTH-1:0] addr;
    } req_data_t;

    typedef struct packed {
        logic [NUM_BANKS-1:0][`XLEN-1:0] data;
    } rsp_data_t;

    logic       req_valid;
    req_data_t  req_data;
    logic       req_ready;

    logic       rsp_valid;
    rsp_data_t  rsp_data;

    modport master (
        output req_valid,
        output req_data,
        input  req_ready,
        input  rsp_valid,
        input  rsp_data
    );

    modport slave (
        input  req_valid,
        input  req_data,
        output req_ready,
        output rsp_valid,
        output rsp_data
    );

endinterface
