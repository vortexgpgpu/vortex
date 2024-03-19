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
    parameter CORE_ID = 0
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

    localparam LMEM_ADDR_WIDTH = `LMEM_LOG_SIZE - `CLOG2(LSU_WORD_SIZE);

     VX_lsu_mem_if #(
        .NUM_LANES (`NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lmem_lsu_if[`NUM_LSU_BLOCKS]();

    for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin
        
        wire [`NUM_LSU_LANES-1:0] is_addr_local_mask;
        for (genvar j = 0; j < `NUM_LSU_LANES; ++j) begin
            assign is_addr_local_mask[j] = lsu_mem_in_if[i].req_data.mask[j] 
                                        && lsu_mem_in_if[i].req_data.atype[j][`ADDR_TYPE_LOCAL];
        end
        
        wire is_addr_local = | is_addr_local_mask;
        wire is_addr_global = | (~is_addr_local_mask);
        
        assign lsu_mem_out_if[i].req_valid      = lsu_mem_in_if[i].req_valid && is_addr_global;
        assign lsu_mem_out_if[i].req_data.mask  = lsu_mem_in_if[i].req_data.mask & ~is_addr_local_mask;
        assign lsu_mem_out_if[i].req_data.rw    = lsu_mem_in_if[i].req_data.rw;
        assign lsu_mem_out_if[i].req_data.byteen= lsu_mem_in_if[i].req_data.byteen;
        assign lsu_mem_out_if[i].req_data.addr  = lsu_mem_in_if[i].req_data.addr;
        assign lsu_mem_out_if[i].req_data.atype = lsu_mem_in_if[i].req_data.atype;
        assign lsu_mem_out_if[i].req_data.data  = lsu_mem_in_if[i].req_data.data;
        assign lsu_mem_out_if[i].req_data.tag   = lsu_mem_in_if[i].req_data.tag;

        assign lmem_lsu_if[i].req_valid         = lsu_mem_in_if[i].req_valid && is_addr_local;
        assign lmem_lsu_if[i].req_data.mask     = lsu_mem_in_if[i].req_data.mask & is_addr_local_mask;
        assign lmem_lsu_if[i].req_data.rw       = lsu_mem_in_if[i].req_data.rw;
        assign lmem_lsu_if[i].req_data.byteen   = lsu_mem_in_if[i].req_data.byteen;
        assign lmem_lsu_if[i].req_data.addr     = lsu_mem_in_if[i].req_data.addr;
        assign lmem_lsu_if[i].req_data.atype    = lsu_mem_in_if[i].req_data.atype;
        assign lmem_lsu_if[i].req_data.data     = lsu_mem_in_if[i].req_data.data;
        assign lmem_lsu_if[i].req_data.tag      = lsu_mem_in_if[i].req_data.tag;

        assign lsu_mem_in_if[i].req_ready = (lsu_mem_out_if[i].req_ready && is_addr_global) 
                                         || (lmem_lsu_if[i].req_ready && is_addr_local);
    end

    `RESET_RELAY (arb_reset, reset);

    for (genvar i = 0; i < `NUM_LSU_BLOCKS; ++i) begin
        
        wire rsp_arb_valid;
        wire rsp_arb_index;
        wire rsp_arb_ready;

        VX_generic_arbiter #(
            .NUM_REQS    (2),
            .LOCK_ENABLE (1),
            .TYPE        ("R")
        ) arbiter (
            .clk         (clk),
            .reset       (arb_reset),
            .requests    ({     
                lmem_lsu_if[i].rsp_valid,
                lsu_mem_out_if[i].rsp_valid
            }),
            .grant_valid (rsp_arb_valid),
            .grant_index (rsp_arb_index),
            `UNUSED_PIN (grant_onehot),
            .grant_unlock(rsp_arb_ready)
        );

        assign lsu_mem_in_if[i].rsp_valid = rsp_arb_valid;
        assign lsu_mem_in_if[i].rsp_data.mask = rsp_arb_index ? lmem_lsu_if[i].rsp_data.mask : lsu_mem_out_if[i].rsp_data.mask;
        assign lsu_mem_in_if[i].rsp_data.data = rsp_arb_index ? lmem_lsu_if[i].rsp_data.data : lsu_mem_out_if[i].rsp_data.data;
        assign lsu_mem_in_if[i].rsp_data.tag = rsp_arb_index ? lmem_lsu_if[i].rsp_data.tag : lsu_mem_out_if[i].rsp_data.tag;
        assign lsu_mem_out_if[i].rsp_ready = lsu_mem_in_if[i].rsp_ready && ~rsp_arb_index;
        assign lmem_lsu_if[i].rsp_ready = lsu_mem_in_if[i].rsp_ready && rsp_arb_index;        
        assign rsp_arb_ready = lsu_mem_in_if[i].rsp_ready;
    end

    VX_mem_bus_if #(
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lmem_bus_if[LSU_NUM_REQS]();

    `RESET_RELAY (adapter_reset, reset);

    for (genvar  i = 0; i < `NUM_LSU_BLOCKS; ++i) begin
        VX_mem_bus_if #(
            .DATA_SIZE (LSU_WORD_SIZE),
            .TAG_WIDTH (LSU_TAG_WIDTH)
        ) lmem_bus_tmp_if[`NUM_LSU_LANES]();

        VX_lsu_adapter #(
            .NUM_LANES    (`NUM_LSU_LANES),
            .DATA_SIZE    (LSU_WORD_SIZE), 
            .TAG_WIDTH    (LSU_TAG_WIDTH),
            .TAG_SEL_BITS (LSU_TAG_WIDTH - `UUID_WIDTH)
        ) lsu_adapter (
            .clk        (clk),
            .reset      (adapter_reset),
            .lsu_mem_if (lmem_lsu_if[i]),
            .mem_bus_if (lmem_bus_tmp_if)
        );

        for (genvar j = 0; j < `NUM_LSU_LANES; ++j) begin
            `ASSIGN_VX_MEM_BUS_IF (lmem_bus_if[i * `NUM_LSU_LANES + j], lmem_bus_tmp_if[j]);
        end
    end

    `RESET_RELAY (lmem_reset, reset);
    
    VX_local_mem #(
        .INSTANCE_ID($sformatf("core%0d-lmem", CORE_ID)),
        .SIZE       (1 << `LMEM_LOG_SIZE),
        .NUM_REQS   (LSU_NUM_REQS),
        .NUM_BANKS  (`LMEM_NUM_BANKS),
        .WORD_SIZE  (LSU_WORD_SIZE),
        .ADDR_WIDTH (LMEM_ADDR_WIDTH),
        .UUID_WIDTH (`UUID_WIDTH), 
        .TAG_WIDTH  (LSU_TAG_WIDTH)
    ) local_mem (        
        .clk        (clk),
        .reset      (lmem_reset),
    `ifdef PERF_ENABLE
        .cache_perf (cache_perf),
    `endif
        .mem_bus_if (lmem_bus_if)
    );

endmodule
