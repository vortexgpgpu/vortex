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

interface VX_lsu_mem_if import VX_gpu_pkg::*; #(
    parameter NUM_LANES  = 1,
    parameter DATA_SIZE  = 1,
    parameter TAG_WIDTH  = 1,
    parameter FLAGS_WIDTH = MEM_FLAGS_WIDTH,
    parameter MEM_ADDR_WIDTH = `MEM_ADDR_WIDTH,
    parameter ADDR_WIDTH = MEM_ADDR_WIDTH - `CLOG2(DATA_SIZE)
) ();

    typedef struct packed {
        logic [UUID_WIDTH-1:0]           uuid;
        logic [TAG_WIDTH-UUID_WIDTH-1:0] value;
    } tag_t;

    /* verilator lint_off UNUSEDSIGNAL */
    typedef struct packed {
        logic [NUM_LANES-1:0]                  mask;
        logic                                  rw;
        logic [NUM_LANES-1:0][ADDR_WIDTH-1:0]  addr;
        logic [NUM_LANES-1:0][DATA_SIZE*8-1:0] data;
        logic [NUM_LANES-1:0][DATA_SIZE-1:0]   byteen;
        logic [NUM_LANES-1:0][FLAGS_WIDTH-1:0] flags;
        tag_t                                  tag;
    `ifdef EXT_A_ENABLE
        // Per-lane AMO sideband. amo[i].valid == 1 marks lane i as an
        // AMO request; the bank's reservation table keys on
        // amo[i].hart_id = make_hart_id(cid, wid, lane). Plain loads
        // and stores leave amo defaulted to zero. The local-mem path
        // discards amo (LMEM-AMO is out of scope, proposal §6) so
        // some bits may end up driven-zero / unread — lint suppressed
        // at the typedef.
        amo_req_t [NUM_LANES-1:0] amo;
    `endif
    } req_data_t;
    /* verilator lint_on UNUSEDSIGNAL */

    typedef struct packed {
        logic [NUM_LANES-1:0]                  mask;
        logic [NUM_LANES-1:0][DATA_SIZE*8-1:0] data;
        tag_t                                  tag;
    } rsp_data_t;

    logic  req_valid;
    /* verilator lint_off UNUSEDSIGNAL */
    req_data_t req_data;
    /* verilator lint_on UNUSEDSIGNAL */
    logic  req_ready;

    logic  rsp_valid;
    rsp_data_t rsp_data;
    logic  rsp_ready;

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
