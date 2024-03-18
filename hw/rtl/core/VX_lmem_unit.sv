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

    VX_mem_bus_if.slave     dcache_bus_in_if [DCACHE_NUM_REQS],
    VX_mem_bus_if.master    dcache_bus_out_if [DCACHE_NUM_REQS]
);
    `STATIC_ASSERT(`IS_DIVISBLE((1 << `LMEM_LOG_SIZE), `MEM_BLOCK_SIZE), ("invalid parameter"))
    `STATIC_ASSERT(0 == (`LMEM_BASE_ADDR % (1 << `LMEM_LOG_SIZE)), ("invalid parameter"))

    localparam LMEM_ADDR_WIDTH = `LMEM_LOG_SIZE - `CLOG2(DCACHE_WORD_SIZE);

    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_TAG_WIDTH)
    ) lmem_bus_if[DCACHE_NUM_REQS]();

    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_TAG_WIDTH)
    ) switch_out_bus_if[2 * DCACHE_NUM_REQS]();

    `RESET_RELAY (switch_reset, reset);

    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin
        VX_mem_switch #(
            .NUM_REQS     (2),
            .DATA_SIZE    (DCACHE_WORD_SIZE),
            .TAG_WIDTH    (DCACHE_TAG_WIDTH),
            .ARBITER      ("P"),
            .REQ_OUT_BUF  (2),
            .RSP_OUT_BUF  (2)
        ) lmem_switch (
            .clk        (clk),
            .reset      (switch_reset),
            .bus_sel    (dcache_bus_in_if[i].req_data.atype[`ADDR_TYPE_LOCAL]),
            .bus_in_if  (dcache_bus_in_if[i]),
            .bus_out_if (switch_out_bus_if[i * 2 +: 2])
        );

        // output bus[0] goes to the dcache
        `ASSIGN_VX_MEM_BUS_IF (dcache_bus_out_if[i], switch_out_bus_if[i * 2 + 0]);

        // output bus[1] goes to the local memory
        `ASSIGN_VX_MEM_BUS_IF (lmem_bus_if[i], switch_out_bus_if[i * 2 + 1]);
    end

    `RESET_RELAY (lmem_reset, reset);
    
    VX_local_mem #(
        .INSTANCE_ID($sformatf("core%0d-lmem", CORE_ID)),
        .SIZE       (1 << `LMEM_LOG_SIZE),
        .NUM_REQS   (DCACHE_NUM_REQS),
        .NUM_BANKS  (`LMEM_NUM_BANKS),
        .WORD_SIZE  (DCACHE_WORD_SIZE),
        .ADDR_WIDTH (LMEM_ADDR_WIDTH),
        .UUID_WIDTH (`UUID_WIDTH), 
        .TAG_WIDTH  (DCACHE_TAG_WIDTH)
    ) local_mem (        
        .clk        (clk),
        .reset      (lmem_reset),

    `ifdef PERF_ENABLE
        .cache_perf (cache_perf),
    `endif
        .mem_bus_if (lmem_bus_if)
    );

endmodule
