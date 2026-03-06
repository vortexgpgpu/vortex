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

// Write-only DXA shared-memory bank write interface.
// Shared valid/addr across all banks; per-bank data and byte-enables
// control which banks actually receive a write.

interface VX_dxa_bank_wr_if #(
    parameter NUM_BANKS       = 4,
    parameter BANK_ADDR_WIDTH = 12,
    parameter WORD_SIZE       = 4,
    parameter TAG_WIDTH       = 8
) ();

    localparam WORD_WIDTH = WORD_SIZE * 8;

    // Shared write control (same valid/addr for all banks)
    logic                                       wr_valid;   // write enable (shared)
    logic [BANK_ADDR_WIDTH-1:0]                 wr_addr;    // word address within bank (shared)
    logic [NUM_BANKS-1:0][WORD_WIDTH-1:0]       wr_data;    // per-bank write data
    logic [NUM_BANKS-1:0][WORD_SIZE-1:0]        wr_byteen;  // per-bank byte enables
    logic [TAG_WIDTH-1:0]                       wr_tag;     // completion metadata (shared)
    logic                                       wr_ready;   // backpressure (all-or-nothing)

    modport master (
        output wr_valid,
        output wr_addr,
        output wr_data,
        output wr_byteen,
        output wr_tag,
        input  wr_ready
    );

    modport slave (
        input  wr_valid,
        input  wr_addr,
        input  wr_data,
        input  wr_byteen,
        input  wr_tag,
        output wr_ready
    );

endinterface
