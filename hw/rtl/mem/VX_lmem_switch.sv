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

module VX_lmem_switch import VX_gpu_pkg::*; #(
    parameter REQ0_OUT_BUF = 0,
    parameter REQ1_OUT_BUF = 0,
    parameter RSP_OUT_BUF  = 0,
    parameter `STRING ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,
    VX_lsu_mem_if.slave     lsu_in_if,
    VX_lsu_mem_if.master    global_out_if,
    VX_lsu_mem_if.master    local_out_if
);
    localparam REQ_DATAW = `NUM_LSU_LANES + 1 + `NUM_LSU_LANES * (LSU_WORD_SIZE + LSU_ADDR_WIDTH + `MEM_REQ_FLAGS_WIDTH + LSU_WORD_SIZE * 8) + LSU_TAG_WIDTH;
    localparam RSP_DATAW = `NUM_LSU_LANES + `NUM_LSU_LANES * (LSU_WORD_SIZE * 8) + LSU_TAG_WIDTH;

    wire [`NUM_LSU_LANES-1:0] is_addr_local_mask;
    wire req_global_ready;
    wire req_local_ready;

    for (genvar i = 0; i < `NUM_LSU_LANES; ++i) begin : g_is_addr_local_mask
        assign is_addr_local_mask[i] = lsu_in_if.req_data.flags[i][`MEM_REQ_FLAG_LOCAL];
    end

    wire is_addr_global = | (lsu_in_if.req_data.mask & ~is_addr_local_mask);
    wire is_addr_local  = | (lsu_in_if.req_data.mask & is_addr_local_mask);

    assign lsu_in_if.req_ready = (req_global_ready && is_addr_global)
                              || (req_local_ready && is_addr_local);

    VX_elastic_buffer #(
        .DATAW   (REQ_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(REQ0_OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(REQ0_OUT_BUF))
    ) req_global_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (lsu_in_if.req_valid && is_addr_global),
        .data_in   ({
            lsu_in_if.req_data.mask & ~is_addr_local_mask,
            lsu_in_if.req_data.rw,
            lsu_in_if.req_data.addr,
            lsu_in_if.req_data.data,
            lsu_in_if.req_data.byteen,
            lsu_in_if.req_data.flags,
            lsu_in_if.req_data.tag
        }),
        .ready_in  (req_global_ready),
        .valid_out (global_out_if.req_valid),
        .data_out  (global_out_if.req_data),
        .ready_out (global_out_if.req_ready)
    );

    VX_elastic_buffer #(
        .DATAW   (REQ_DATAW),
        .SIZE    (`TO_OUT_BUF_SIZE(REQ1_OUT_BUF)),
        .OUT_REG (`TO_OUT_BUF_REG(REQ1_OUT_BUF))
    ) req_local_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (lsu_in_if.req_valid && is_addr_local),
        .data_in   ({
            lsu_in_if.req_data.mask & is_addr_local_mask,
            lsu_in_if.req_data.rw,
            lsu_in_if.req_data.addr,
            lsu_in_if.req_data.data,
            lsu_in_if.req_data.byteen,
            lsu_in_if.req_data.flags,
            lsu_in_if.req_data.tag
        }),
        .ready_in  (req_local_ready),
        .valid_out (local_out_if.req_valid),
        .data_out  (local_out_if.req_data),
        .ready_out (local_out_if.req_ready)
    );

    VX_stream_arb #(
        .NUM_INPUTS (2),
        .DATAW      (RSP_DATAW),
        .ARBITER    (ARBITER),
        .OUT_BUF    (RSP_OUT_BUF)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  ({
            local_out_if.rsp_valid,
            global_out_if.rsp_valid
        }),
        .ready_in  ({
            local_out_if.rsp_ready,
            global_out_if.rsp_ready
        }),
        .data_in   ({
            local_out_if.rsp_data,
            global_out_if.rsp_data
        }),
        .data_out  (lsu_in_if.rsp_data),
        .valid_out (lsu_in_if.rsp_valid),
        .ready_out (lsu_in_if.rsp_ready),
        `UNUSED_PIN (sel_out)
    );

endmodule
