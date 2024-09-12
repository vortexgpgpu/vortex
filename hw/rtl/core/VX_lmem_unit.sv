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

module VX_lmem_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output cache_perf_t     cache_perf,
`endif

    VX_lsu_mem_if.slave     lsu_mem_in_if [`NUM_LSU_BLOCKS],
    VX_lsu_mem_if.master    lsu_mem_out_if [`NUM_LSU_BLOCKS]
);
    `STATIC_ASSERT(`IS_DIVISBLE((1 << `LMEM_LOG_SIZE), `MEM_BLOCK_SIZE), ("invalid parameter"))
    `STATIC_ASSERT(0 == (`LMEM_BASE_ADDR % (1 << `LMEM_LOG_SIZE)), ("invalid parameter"))

    localparam REQ_DATAW = `NUM_LSU_LANES + 1 + `NUM_LSU_LANES * (LSU_WORD_SIZE + LSU_ADDR_WIDTH + `ADDR_TYPE_WIDTH + LSU_WORD_SIZE * 8) + LSU_TAG_WIDTH;
    localparam RSP_DATAW = `NUM_LSU_LANES + `NUM_LSU_LANES * (LSU_WORD_SIZE * 8) + LSU_TAG_WIDTH;
    localparam LMEM_ADDR_WIDTH = `LMEM_LOG_SIZE - `CLOG2(LSU_WORD_SIZE);

     VX_lsu_mem_if #(
        .NUM_LANES (`NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_switch_if[`NUM_LSU_BLOCKS]();

    `RESET_RELAY_EX (block_reset, reset, `NUM_LSU_BLOCKS, 1);

    for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin

        wire [`NUM_LSU_LANES-1:0] is_addr_local_mask;
        for (genvar j = 0; j < `NUM_LSU_LANES; ++j) begin
            assign is_addr_local_mask[j] = lsu_mem_in_if[i].req_data.atype[j][`ADDR_TYPE_LOCAL];
        end

        wire is_addr_global = | (lsu_mem_in_if[i].req_data.mask & ~is_addr_local_mask);
        wire is_addr_local = | (lsu_mem_in_if[i].req_data.mask & is_addr_local_mask);

        wire req_global_ready;
        wire req_local_ready;

        VX_elastic_buffer #(
            .DATAW   (REQ_DATAW),
            .SIZE    (2),
            .OUT_REG (1)
        ) req_global_buf (
            .clk       (clk),
            .reset     (block_reset[i]),
            .valid_in  (lsu_mem_in_if[i].req_valid && is_addr_global),
            .data_in   ({
                lsu_mem_in_if[i].req_data.mask & ~is_addr_local_mask,
                lsu_mem_in_if[i].req_data.rw,
                lsu_mem_in_if[i].req_data.byteen,
                lsu_mem_in_if[i].req_data.addr,
                lsu_mem_in_if[i].req_data.atype,
                lsu_mem_in_if[i].req_data.data,
                lsu_mem_in_if[i].req_data.tag
            }),
            .ready_in  (req_global_ready),
            .valid_out (lsu_mem_out_if[i].req_valid),
            .data_out  ({
                lsu_mem_out_if[i].req_data.mask,
                lsu_mem_out_if[i].req_data.rw,
                lsu_mem_out_if[i].req_data.byteen,
                lsu_mem_out_if[i].req_data.addr,
                lsu_mem_out_if[i].req_data.atype,
                lsu_mem_out_if[i].req_data.data,
                lsu_mem_out_if[i].req_data.tag
            }),
            .ready_out (lsu_mem_out_if[i].req_ready)
        );

        VX_elastic_buffer #(
            .DATAW   (REQ_DATAW),
            .SIZE    (0),
            .OUT_REG (0)
        ) req_local_buf (
            .clk       (clk),
            .reset     (block_reset[i]),
            .valid_in  (lsu_mem_in_if[i].req_valid && is_addr_local),
            .data_in   ({
                lsu_mem_in_if[i].req_data.mask & is_addr_local_mask,
                lsu_mem_in_if[i].req_data.rw,
                lsu_mem_in_if[i].req_data.byteen,
                lsu_mem_in_if[i].req_data.addr,
                lsu_mem_in_if[i].req_data.atype,
                lsu_mem_in_if[i].req_data.data,
                lsu_mem_in_if[i].req_data.tag
            }),
            .ready_in  (req_local_ready),
            .valid_out (lsu_switch_if[i].req_valid),
            .data_out  ({
                lsu_switch_if[i].req_data.mask,
                lsu_switch_if[i].req_data.rw,
                lsu_switch_if[i].req_data.byteen,
                lsu_switch_if[i].req_data.addr,
                lsu_switch_if[i].req_data.atype,
                lsu_switch_if[i].req_data.data,
                lsu_switch_if[i].req_data.tag
            }),
            .ready_out (lsu_switch_if[i].req_ready)
        );

        assign lsu_mem_in_if[i].req_ready = (req_global_ready && is_addr_global)
                                         || (req_local_ready && is_addr_local);

        VX_stream_arb #(
            .NUM_INPUTS (2),
            .DATAW      (RSP_DATAW),
            .ARBITER    ("R"),
            .OUT_BUF    (1)
        ) rsp_arb (
            .clk       (clk),
            .reset     (block_reset[i]),
            .valid_in  ({
                lsu_switch_if[i].rsp_valid,
                lsu_mem_out_if[i].rsp_valid
            }),
            .ready_in  ({
                lsu_switch_if[i].rsp_ready,
                lsu_mem_out_if[i].rsp_ready
            }),
            .data_in   ({
                lsu_switch_if[i].rsp_data,
                lsu_mem_out_if[i].rsp_data
            }),
            .data_out  (lsu_mem_in_if[i].rsp_data),
            .valid_out (lsu_mem_in_if[i].rsp_valid),
            .ready_out (lsu_mem_in_if[i].rsp_ready),
            `UNUSED_PIN (sel_out)
        );
    end

    VX_mem_bus_if #(
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lmem_bus_if[LSU_NUM_REQS]();

    for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin
        VX_mem_bus_if #(
            .DATA_SIZE (LSU_WORD_SIZE),
            .TAG_WIDTH (LSU_TAG_WIDTH)
        ) lmem_bus_tmp_if[`NUM_LSU_LANES]();

        VX_lsu_adapter #(
            .NUM_LANES    (`NUM_LSU_LANES),
            .DATA_SIZE    (LSU_WORD_SIZE),
            .TAG_WIDTH    (LSU_TAG_WIDTH),
            .TAG_SEL_BITS (LSU_TAG_WIDTH - `UUID_WIDTH),
            .ARBITER      ("P"),
            .REQ_OUT_BUF  (3),
            .RSP_OUT_BUF  (0)
        ) lsu_adapter (
            .clk        (clk),
            .reset      (block_reset[i]),
            .lsu_mem_if (lsu_switch_if[i]),
            .mem_bus_if (lmem_bus_tmp_if)
        );

        for (genvar j = 0; j < `NUM_LSU_LANES; ++j) begin
            `ASSIGN_VX_MEM_BUS_IF (lmem_bus_if[i * `NUM_LSU_LANES + j], lmem_bus_tmp_if[j]);
        end
    end

    `RESET_RELAY (lmem_reset, reset);

    VX_local_mem #(
        .INSTANCE_ID($sformatf("%s-lmem", INSTANCE_ID)),
        .SIZE       (1 << `LMEM_LOG_SIZE),
        .NUM_REQS   (LSU_NUM_REQS),
        .NUM_BANKS  (`LMEM_NUM_BANKS),
        .WORD_SIZE  (LSU_WORD_SIZE),
        .ADDR_WIDTH (LMEM_ADDR_WIDTH),
        .UUID_WIDTH (`UUID_WIDTH),
        .TAG_WIDTH  (LSU_TAG_WIDTH),
        .OUT_BUF    (3)
    ) local_mem (
        .clk        (clk),
        .reset      (lmem_reset),
    `ifdef PERF_ENABLE
        .cache_perf (cache_perf),
    `endif
        .mem_bus_if (lmem_bus_if)
    );

endmodule
