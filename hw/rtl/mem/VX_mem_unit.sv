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

`define SMEM_ADDR_STACK_OPT

module VX_mem_unit import VX_gpu_pkg::*; #(
    parameter CLUSTER_ID = 0
) (
    input wire              clk,
    input wire              reset,
    
`ifdef PERF_ENABLE
    output cache_perf_t     perf_l2cache,
`ifdef EXT_RASTER_ENABLE
    output cache_perf_t     perf_rcache,
`endif
`ifdef EXT_TEX_ENABLE
    output cache_perf_t     perf_tcache,
`endif
`ifdef EXT_ROP_ENABLE
    output cache_perf_t     perf_ocache,
`endif
`endif    

    VX_mem_bus_if.slave     per_socket_icache_mem_bus_if [`NUM_SOCKETS],

    VX_mem_bus_if.slave     per_socket_dcache_mem_bus_if [`NUM_SOCKETS],

`ifdef EXT_TEX_ENABLE
    VX_mem_bus_if.slave     tcache_bus_if [`NUM_TEX_UNITS * TCACHE_NUM_REQS],
`endif

`ifdef EXT_RASTER_ENABLE
    VX_mem_bus_if.slave     rcache_bus_if [`NUM_RASTER_UNITS * RCACHE_NUM_REQS],
`endif 

`ifdef EXT_ROP_ENABLE
    VX_mem_bus_if.slave     ocache_bus_if [`NUM_ROP_UNITS * OCACHE_NUM_REQS],
`endif

    VX_mem_bus_if.master    mem_bus_if
);
    VX_mem_bus_if #(
        .DATA_SIZE (ICACHE_LINE_SIZE),
        .TAG_WIDTH (ICACHE_MEM_ARB_TAG_WIDTH)
    ) icache_mem_bus_if[1]();

    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_LINE_SIZE),
        .TAG_WIDTH (DCACHE_MEM_ARB_TAG_WIDTH)
    ) dcache_mem_bus_if[1]();

    `RESET_RELAY (l1_mem_arb_reset, reset);

    VX_mem_arb #(
        .NUM_INPUTS  (`NUM_SOCKETS),
        .DATA_SIZE   (ICACHE_LINE_SIZE),
        .TAG_WIDTH   (ICACHE_MEM_TAG_WIDTH),
        .TAG_SEL_IDX (1), // Skip 0 for NC flag
        .ARBITER     ("R"),
        .OUT_REG_REQ (2),
        .OUT_REG_RSP (2)
    ) icache_mem_arb (
        .clk        (clk),
        .reset      (l1_mem_arb_reset),
        .bus_in_if  (per_socket_icache_mem_bus_if),
        .bus_out_if (icache_mem_bus_if)
    );

    VX_mem_arb #(
        .NUM_INPUTS  (`NUM_SOCKETS),
        .DATA_SIZE   (DCACHE_LINE_SIZE),
        .TAG_WIDTH   (DCACHE_MEM_TAG_WIDTH),
        .TAG_SEL_IDX (1), // Skip 0 for NC flag
        .ARBITER     ("R"),
        .OUT_REG_REQ (2),
        .OUT_REG_RSP (2)
    ) dcache_mem_arb (
        .clk        (clk),
        .reset      (l1_mem_arb_reset),
        .bus_in_if  (per_socket_dcache_mem_bus_if),
        .bus_out_if (dcache_mem_bus_if)
    );

/////////////////////////////// T-Cache ///////////////////////////////////

`ifdef EXT_TEX_ENABLE

    VX_mem_bus_if #(
        .DATA_SIZE (TCACHE_LINE_SIZE),
        .TAG_WIDTH (TCACHE_MEM_TAG_WIDTH)
    ) tcache_mem_bus_if();

    `RESET_RELAY (tcache_reset, reset);

    VX_cache_cluster #(
        .INSTANCE_ID    ($sformatf("cluster%0d-tcache", CLUSTER_ID)),
        .NUM_UNITS      (`NUM_TCACHES),
        .NUM_INPUTS     (`NUM_TEX_UNITS),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`TCACHE_SIZE),
        .LINE_SIZE      (TCACHE_LINE_SIZE),
        .NUM_BANKS      (`TCACHE_NUM_BANKS),
        .NUM_WAYS       (`TCACHE_NUM_WAYS),
        .WORD_SIZE      (TCACHE_WORD_SIZE),
        .NUM_REQS       (TCACHE_NUM_REQS),
        .CRSQ_SIZE      (`TCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`TCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`TCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`TCACHE_MREQ_SIZE),
        .TAG_WIDTH      (TCACHE_TAG_WIDTH),
        .WRITE_ENABLE   (0),
        .UUID_WIDTH     (`UUID_WIDTH),
        .NC_ENABLE      (0),
        .CORE_OUT_REG   (2),
        .MEM_OUT_REG    (2)
    ) tcache (
    `ifdef PERF_ENABLE
        .cache_perf     (perf_tcache),
    `endif        
        .clk            (clk),
        .reset          (tcache_reset),
        .core_bus_if    (tcache_bus_if),
        .mem_bus_if     (tcache_mem_bus_if)
    );

`endif

/////////////////////////////// R-Cache ///////////////////////////////////

`ifdef EXT_RASTER_ENABLE

    VX_mem_bus_if #(
        .DATA_SIZE (RCACHE_LINE_SIZE),
        .TAG_WIDTH (RCACHE_MEM_TAG_WIDTH)
    ) rcache_mem_bus_if();

    `RESET_RELAY (rcache_reset, reset);

    VX_cache_cluster #(
        .INSTANCE_ID    ($sformatf("cluster%0d-rcache", CLUSTER_ID)),
        .NUM_UNITS      (`NUM_RCACHES),
        .NUM_INPUTS     (`NUM_RASTER_UNITS),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`RCACHE_SIZE),
        .LINE_SIZE      (RCACHE_LINE_SIZE),
        .NUM_BANKS      (`RCACHE_NUM_BANKS),
        .NUM_WAYS       (`RCACHE_NUM_WAYS),
        .WORD_SIZE      (RCACHE_WORD_SIZE),
        .NUM_REQS       (RCACHE_NUM_REQS),
        .CRSQ_SIZE      (`RCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`RCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`RCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`RCACHE_MREQ_SIZE),
        .TAG_WIDTH      (RCACHE_TAG_WIDTH),
        .WRITE_ENABLE   (0),
        .UUID_WIDTH     (0),        
        .NC_ENABLE      (0),
        .CORE_OUT_REG   (2),
        .MEM_OUT_REG    (2)
    ) rcache (
    `ifdef PERF_ENABLE
        .cache_perf     (perf_rcache),
    `endif        
        .clk            (clk),
        .reset          (rcache_reset),
        .core_bus_if    (rcache_bus_if),
        .mem_bus_if     (rcache_mem_bus_if)
    );

`endif

/////////////////////////////// O-Cache ///////////////////////////////////

`ifdef EXT_ROP_ENABLE

    VX_mem_bus_if #(
        .DATA_SIZE (OCACHE_LINE_SIZE),
        .TAG_WIDTH (OCACHE_MEM_TAG_WIDTH)
    ) ocache_mem_bus_if();

    `RESET_RELAY (ocache_reset, reset);

    VX_cache_cluster #(
        .INSTANCE_ID    ($sformatf("cluster%0d-ocache", CLUSTER_ID)),
        .NUM_UNITS      (`NUM_OCACHES),
        .NUM_INPUTS     (`NUM_ROP_UNITS),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`OCACHE_SIZE),
        .LINE_SIZE      (OCACHE_LINE_SIZE),
        .NUM_BANKS      (`OCACHE_NUM_BANKS),
        .NUM_WAYS       (`OCACHE_NUM_WAYS),
        .WORD_SIZE      (OCACHE_WORD_SIZE),
        .NUM_REQS       (OCACHE_NUM_REQS),
        .CRSQ_SIZE      (`OCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`OCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`OCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`OCACHE_MREQ_SIZE),
        .TAG_WIDTH      (OCACHE_TAG_WIDTH),
        .WRITE_ENABLE   (1),
        .UUID_WIDTH     (`UUID_WIDTH),
        .NC_ENABLE      (0),
        .CORE_OUT_REG   (2),
        .MEM_OUT_REG    (2)
    ) ocache (
    `ifdef PERF_ENABLE
        .cache_perf     (perf_ocache),
    `endif        
        .clk            (clk),
        .reset          (ocache_reset),

        .core_bus_if    (ocache_bus_if),
        .mem_bus_if     (ocache_mem_bus_if)
    );

`endif

/////////////////////////////// L2-Cache //////////////////////////////////

    VX_mem_bus_if #(
        .DATA_SIZE (L2_WORD_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) l2_mem_bus_if[L2_NUM_REQS]();

    localparam I_MEM_ARB_IDX = 0;
    localparam D_MEM_ARB_IDX = I_MEM_ARB_IDX + 1;
    localparam T_MEM_ARB_IDX = D_MEM_ARB_IDX + 1;
    localparam R_MEM_ARB_IDX = T_MEM_ARB_IDX + `EXT_TEX_ENABLED;
    localparam O_MEM_ARB_IDX = R_MEM_ARB_IDX + `EXT_RASTER_ENABLED;
    `UNUSED_PARAM (T_MEM_ARB_IDX)
    `UNUSED_PARAM (R_MEM_ARB_IDX)
    `UNUSED_PARAM (O_MEM_ARB_IDX)

    `ASSIGN_VX_MEM_BUS_IF_X (l2_mem_bus_if[I_MEM_ARB_IDX], icache_mem_bus_if[0], L1_MEM_TAG_WIDTH, ICACHE_MEM_ARB_TAG_WIDTH);
    `ASSIGN_VX_MEM_BUS_IF_X (l2_mem_bus_if[D_MEM_ARB_IDX], dcache_mem_bus_if[0], L1_MEM_TAG_WIDTH, DCACHE_MEM_ARB_TAG_WIDTH);

`ifdef EXT_TEX_ENABLE
    `ASSIGN_VX_MEM_BUS_IF_X (l2_mem_bus_if[T_MEM_ARB_IDX], tcache_mem_bus_if, L1_MEM_TAG_WIDTH, TCACHE_MEM_TAG_WIDTH);
`endif

`ifdef EXT_RASTER_ENABLE
    `ASSIGN_VX_MEM_BUS_IF_X (l2_mem_bus_if[R_MEM_ARB_IDX], rcache_mem_bus_if, L1_MEM_TAG_WIDTH, RCACHE_MEM_TAG_WIDTH);
`endif

`ifdef EXT_ROP_ENABLE
    `ASSIGN_VX_MEM_BUS_IF_X (l2_mem_bus_if[O_MEM_ARB_IDX], ocache_mem_bus_if, L1_MEM_TAG_WIDTH, OCACHE_MEM_TAG_WIDTH);
`endif

    `RESET_RELAY (l2_reset, reset);

    VX_cache_wrap #(
        .INSTANCE_ID    ($sformatf("cluster%0d-l2cache", CLUSTER_ID)),
        .CACHE_SIZE     (`L2_CACHE_SIZE),
        .LINE_SIZE      (`L2_LINE_SIZE),
        .NUM_BANKS      (`L2_NUM_BANKS),
        .NUM_WAYS       (`L2_NUM_WAYS),
        .WORD_SIZE      (L2_WORD_SIZE),
        .NUM_REQS       (L2_NUM_REQS),
        .CRSQ_SIZE      (`L2_CRSQ_SIZE),
        .MSHR_SIZE      (`L2_MSHR_SIZE),
        .MRSQ_SIZE      (`L2_MRSQ_SIZE),
        .MREQ_SIZE      (`L2_MREQ_SIZE),
        .TAG_WIDTH      (L1_MEM_TAG_WIDTH),
        .WRITE_ENABLE   (1),       
        .UUID_WIDTH     (`UUID_WIDTH),
        .CORE_OUT_REG   (2),
        .MEM_OUT_REG    (2),
        .NC_ENABLE      (1),
        .PASSTHRU       (!`L2_ENABLED)
    ) l2cache (            
        .clk            (clk),
        .reset          (l2_reset),
    `ifdef PERF_ENABLE
        .cache_perf     (perf_l2cache),
    `endif
        .core_bus_if    (l2_mem_bus_if),
        .mem_bus_if     (mem_bus_if)
    );
    
endmodule
