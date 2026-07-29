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

`ifdef VX_CFG_DXA_MBAR_ENABLE

module VX_dxa_mbar_completion import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_BANKS = 1,
    parameter ATTR_WIDTH = DXA_LMEM_ATTR_W
) (
    input wire                         clk,
    input wire                         reset,
    input wire [NUM_BANKS-1:0]         bank_wr_fire,
    input wire [ATTR_WIDTH-1:0]        bank_wr_attr,
    VX_mbar_completion_if.master       completion_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    wire terminal_write =
        (|bank_wr_fire) && bank_wr_attr[ATTR_WIDTH-1];
    wire [DXA_COMPLETION_REF_W-1:0] completion_ref =
        bank_wr_attr[DXA_COMPLETION_REF_W-1:0];

    localparam FIFO_DEPTH =
        (`VX_CFG_NUM_WARPS < 2) ? 2 : `VX_CFG_NUM_WARPS;
    wire [DXA_COMPLETION_REF_W-1:0] fifo_data;
    wire fifo_empty;
    wire fifo_full;
    wire fifo_pop = completion_if.valid && completion_if.ready;

    VX_fifo_queue #(
        .DATAW  (DXA_COMPLETION_REF_W),
        .DEPTH  (FIFO_DEPTH),
        .LUTRAM (1)
    ) completion_fifo (
        .clk        (clk),
        .reset      (reset),
        .push       (terminal_write),
        .pop        (fifo_pop),
        .data_in    (completion_ref),
        .data_out   (fifo_data),
        .empty      (fifo_empty),
        .full       (fifo_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    assign completion_if.valid = ~fifo_empty;
    assign completion_if.addr =
        fifo_data[MBAR_OBJECT_LG2 +: MBAR_ADDR_W];
    `UNUSED_VAR (fifo_data[MBAR_OBJECT_LG2-1:0])

    `RUNTIME_ASSERT(!(terminal_write && fifo_full),
        ("DXA mbarrier completion FIFO overflow"))
    `RUNTIME_ASSERT(!terminal_write
                 || (completion_ref[MBAR_OBJECT_LG2-1:0] == 0),
        ("DXA mbarrier completion reference is misaligned"))

endmodule

`endif
