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
    VX_mem_perf_if.master   mem_perf_if,
`endif    

    VX_mem_bus_if.slave     icache_bus_if [`NUM_SOCKETS],

    VX_mem_bus_if.slave     dcache_bus_if [`NUM_SOCKETS * DCACHE_NUM_REQS],

    VX_mem_bus_if.master    mem_bus_if
);

`ifdef PERF_ENABLE
    VX_cache_perf_if perf_icache_if();
    VX_cache_perf_if perf_dcache_if();
    VX_cache_perf_if perf_l2cache_if();
`endif   

/////////////////////////////// I-Cache ///////////////////////////////////

    VX_mem_bus_if #(
        .DATA_SIZE (ICACHE_LINE_SIZE),
        .TAG_WIDTH (ICACHE_MEM_TAG_WIDTH)
    ) icache_mem_bus_if();
    
    `RESET_RELAY (icache_reset, reset);

    VX_cache_cluster #(
        .INSTANCE_ID    ($sformatf("cluster%0d-icache", CLUSTER_ID)),    
        .NUM_UNITS      (`NUM_ICACHES),
        .NUM_INPUTS     (`NUM_SOCKETS),
        .TAG_SEL_IDX    (0),
        .CACHE_SIZE     (`ICACHE_SIZE),
        .LINE_SIZE      (ICACHE_LINE_SIZE),
        .NUM_BANKS      (1),
        .NUM_WAYS       (`ICACHE_NUM_WAYS),
        .WORD_SIZE      (ICACHE_WORD_SIZE),
        .NUM_REQS       (1),
        .CRSQ_SIZE      (`ICACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`ICACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`ICACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`ICACHE_MREQ_SIZE),
        .TAG_WIDTH      (ICACHE_ARB_TAG_WIDTH),
        .UUID_WIDTH     (`UUID_WIDTH),
        .WRITE_ENABLE   (0),
        .CORE_OUT_REG   (2),
        .MEM_OUT_REG    (2)
    ) icache (
    `ifdef PERF_ENABLE
        .cache_perf_if  (perf_icache_if),
    `endif
        .clk            (clk),
        .reset          (icache_reset),
        .core_bus_if    (icache_bus_if),
        .mem_bus_if     (icache_mem_bus_if)
    );

/////////////////////////////// D-Cache ///////////////////////////////////

    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_LINE_SIZE),
        .TAG_WIDTH (DCACHE_MEM_TAG_WIDTH)
    ) dcache_mem_bus_if();

    `RESET_RELAY (dcache_reset, reset);

    VX_cache_cluster #(
        .INSTANCE_ID    ($sformatf("cluster%0d-dcache", CLUSTER_ID)),    
        .NUM_UNITS      (`NUM_DCACHES),
        .NUM_INPUTS     (`NUM_SOCKETS),
        .TAG_SEL_IDX    (1),
        .CACHE_SIZE     (`DCACHE_SIZE),
        .LINE_SIZE      (DCACHE_LINE_SIZE),
        .NUM_BANKS      (`DCACHE_NUM_BANKS),
        .NUM_WAYS       (`DCACHE_NUM_WAYS),
        .WORD_SIZE      (DCACHE_WORD_SIZE),
        .NUM_REQS       (DCACHE_NUM_REQS),
        .CRSQ_SIZE      (`DCACHE_CRSQ_SIZE),
        .MSHR_SIZE      (`DCACHE_MSHR_SIZE),
        .MRSQ_SIZE      (`DCACHE_MRSQ_SIZE),
        .MREQ_SIZE      (`DCACHE_MREQ_SIZE),
        .TAG_WIDTH      (DCACHE_ARB_TAG_WIDTH),
        .UUID_WIDTH     (`UUID_WIDTH),
        .WRITE_ENABLE   (1),        
        .NC_ENABLE      (1),
        .CORE_OUT_REG   (`SM_ENABLED ? 2 : 1),
        .MEM_OUT_REG    (2)
    ) dcache (
    `ifdef PERF_ENABLE
        .cache_perf_if  (perf_dcache_if),
    `endif
        
        .clk            (clk),
        .reset          (dcache_reset),        
        .core_bus_if    (dcache_bus_if),
        .mem_bus_if     (dcache_mem_bus_if)
    );

/////////////////////////////// L2-Cache //////////////////////////////////

    VX_mem_bus_if #(
        .DATA_SIZE (L2_WORD_SIZE),
        .TAG_WIDTH (L2_TAG_WIDTH)
    ) l2_mem_bus_if[L2_NUM_REQS]();

    localparam I_MEM_ARB_IDX = 0;
    localparam D_MEM_ARB_IDX = I_MEM_ARB_IDX + 1;

    `ASSIGN_VX_MEM_BUS_IF_X (l2_mem_bus_if[I_MEM_ARB_IDX], icache_mem_bus_if, L1_MEM_TAG_WIDTH, ICACHE_MEM_TAG_WIDTH);
    `ASSIGN_VX_MEM_BUS_IF_X (l2_mem_bus_if[D_MEM_ARB_IDX], dcache_mem_bus_if, L1_MEM_TAG_WIDTH, DCACHE_MEM_TAG_WIDTH);

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
        .cache_perf_if  (perf_l2cache_if),
    `endif
        .core_bus_if    (l2_mem_bus_if),
        .mem_bus_if     (mem_bus_if)
    );

`ifdef PERF_ENABLE
    
    `UNUSED_VAR (perf_dcache_if.mem_stalls)
    `UNUSED_VAR (perf_dcache_if.crsp_stalls)

    assign mem_perf_if.icache_reads       = perf_icache_if.reads;
    assign mem_perf_if.icache_read_misses = perf_icache_if.read_misses;
    
    assign mem_perf_if.dcache_reads       = perf_dcache_if.reads;
    assign mem_perf_if.dcache_writes      = perf_dcache_if.writes;
    assign mem_perf_if.dcache_read_misses = perf_dcache_if.read_misses;
    assign mem_perf_if.dcache_write_misses= perf_dcache_if.write_misses;
    assign mem_perf_if.dcache_bank_stalls = perf_dcache_if.bank_stalls;
    assign mem_perf_if.dcache_mshr_stalls = perf_dcache_if.mshr_stalls;

`ifdef L2_ENABLE
    assign mem_perf_if.l2cache_reads       = perf_l2cache_if.reads;
    assign mem_perf_if.l2cache_writes      = perf_l2cache_if.writes;
    assign mem_perf_if.l2cache_read_misses = perf_l2cache_if.read_misses;
    assign mem_perf_if.l2cache_write_misses= perf_l2cache_if.write_misses;
    assign mem_perf_if.l2cache_bank_stalls = perf_l2cache_if.bank_stalls;
    assign mem_perf_if.l2cache_mshr_stalls = perf_l2cache_if.mshr_stalls;
`else
    assign mem_perf_if.l2cache_reads       = '0;
    assign mem_perf_if.l2cache_writes      = '0;
    assign mem_perf_if.l2cache_read_misses = '0;
    assign mem_perf_if.l2cache_write_misses= '0;
    assign mem_perf_if.l2cache_bank_stalls = '0;
    assign mem_perf_if.l2cache_mshr_stalls = '0;
`endif
    
    assign mem_perf_if.l3cache_reads       = '0;
    assign mem_perf_if.l3cache_writes      = '0;
    assign mem_perf_if.l3cache_read_misses = '0;
    assign mem_perf_if.l3cache_write_misses= '0;
    assign mem_perf_if.l3cache_bank_stalls = '0;
    assign mem_perf_if.l3cache_mshr_stalls = '0;

    assign mem_perf_if.mem_reads   = '0;       
    assign mem_perf_if.mem_writes  = '0;
    assign mem_perf_if.mem_latency = '0;
    
`endif
    
endmodule
