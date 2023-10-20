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

module VX_smem_unit import VX_gpu_pkg::*; #(
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,
    
`ifdef PERF_ENABLE
    VX_mem_perf_if.slave    mem_perf_in_if,
    VX_mem_perf_if.master   mem_perf_out_if,
`endif

    VX_mem_bus_if.slave     dcache_bus_in_if [DCACHE_NUM_REQS],
    VX_mem_bus_if.master    dcache_bus_out_if [DCACHE_NUM_REQS]
);
    `UNUSED_PARAM (CORE_ID)

`ifdef SM_ENABLE
    localparam SMEM_ADDR_WIDTH = `SMEM_LOG_SIZE - `CLOG2(DCACHE_WORD_SIZE);

    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_NOSM_TAG_WIDTH)
    ) switch_out_bus_if[2 * DCACHE_NUM_REQS]();

`ifdef PERF_ENABLE
    VX_cache_perf_if perf_smem_if();
`endif   

    `RESET_RELAY (switch_reset, reset);

    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin
        VX_smem_switch #(
            .NUM_REQS     (2),
            .DATA_SIZE    (DCACHE_WORD_SIZE),
            .TAG_WIDTH    (DCACHE_TAG_WIDTH),
            .TAG_SEL_IDX  (0),
            .ARBITER      ("P"),
            .OUT_REG_REQ  (2),
            .OUT_REG_RSP  (2)
        ) smem_switch (
            .clk        (clk),
            .reset      (switch_reset),
            .bus_in_if  (dcache_bus_in_if[i]),
            .bus_out_if (switch_out_bus_if[i * 2 +: 2])
        );
    end

    // this bus goes to the dcache
    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin
        `ASSIGN_VX_MEM_BUS_IF (dcache_bus_out_if[i], switch_out_bus_if[i * 2]);
    end

    wire [DCACHE_NUM_REQS-1:0]                  smem_req_valid;
    wire [DCACHE_NUM_REQS-1:0]                  smem_req_rw;
    wire [DCACHE_NUM_REQS-1:0][SMEM_ADDR_WIDTH-1:0] smem_req_addr;
    wire [DCACHE_NUM_REQS-1:0][DCACHE_WORD_SIZE-1:0] smem_req_byteen;
    wire [DCACHE_NUM_REQS-1:0][DCACHE_WORD_SIZE*8-1:0] smem_req_data;
    wire [DCACHE_NUM_REQS-1:0][DCACHE_NOSM_TAG_WIDTH-1:0] smem_req_tag;
    wire [DCACHE_NUM_REQS-1:0]                  smem_req_ready;
    wire [DCACHE_NUM_REQS-1:0]                  smem_rsp_valid;
    wire [DCACHE_NUM_REQS-1:0][DCACHE_WORD_SIZE*8-1:0] smem_rsp_data;
    wire [DCACHE_NUM_REQS-1:0][DCACHE_NOSM_TAG_WIDTH-1:0] smem_rsp_tag;
    wire [DCACHE_NUM_REQS-1:0]                  smem_rsp_ready;

    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin

        assign smem_req_valid[i] = switch_out_bus_if[i * 2 + 1].req_valid;
        assign smem_req_rw[i] = switch_out_bus_if[i * 2 + 1].req_data.rw;
        assign smem_req_byteen[i] = switch_out_bus_if[i * 2 + 1].req_data.byteen;
        assign smem_req_data[i] = switch_out_bus_if[i * 2 + 1].req_data.data;
        assign smem_req_tag[i] = switch_out_bus_if[i * 2 + 1].req_data.tag;
        assign switch_out_bus_if[i * 2 + 1].req_ready = smem_req_ready[i];

        assign switch_out_bus_if[i * 2 + 1].rsp_valid = smem_rsp_valid[i];
        assign switch_out_bus_if[i * 2 + 1].rsp_data.data = smem_rsp_data[i];
        assign switch_out_bus_if[i * 2 + 1].rsp_data.tag = smem_rsp_tag[i];
        assign smem_rsp_ready[i] = switch_out_bus_if[i * 2 + 1].rsp_ready;

        assign smem_req_addr[i] = switch_out_bus_if[i * 2 + 1].req_data.addr[SMEM_ADDR_WIDTH-1:0];          
    end

    `RESET_RELAY (smem_reset, reset);
    
    VX_shared_mem #(
        .INSTANCE_ID($sformatf("core%0d-smem", CORE_ID)),
        .SIZE       (1 << `SMEM_LOG_SIZE),
        .NUM_REQS   (DCACHE_NUM_REQS),
        .NUM_BANKS  (`SMEM_NUM_BANKS),
        .WORD_SIZE  (DCACHE_WORD_SIZE),
        .ADDR_WIDTH (SMEM_ADDR_WIDTH),
        .UUID_WIDTH (`UUID_WIDTH), 
        .TAG_WIDTH  (DCACHE_NOSM_TAG_WIDTH)
    ) shared_mem (        
        .clk        (clk),
        .reset      (smem_reset),

    `ifdef PERF_ENABLE
        .cache_perf_if(perf_smem_if),
    `endif

        // Core request
        .req_valid  (smem_req_valid),
        .req_rw     (smem_req_rw),
        .req_byteen (smem_req_byteen),
        .req_addr   (smem_req_addr),
        .req_data   (smem_req_data),        
        .req_tag    (smem_req_tag),
        .req_ready  (smem_req_ready),

        // Core response
        .rsp_valid  (smem_rsp_valid),
        .rsp_data   (smem_rsp_data),
        .rsp_tag    (smem_rsp_tag),
        .rsp_ready  (smem_rsp_ready)
    );   

`else

    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin
        `ASSIGN_VX_MEM_BUS_IF (dcache_bus_out_if[i], dcache_bus_in_if[i]);
    end

    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

`endif

`ifdef PERF_ENABLE

    assign mem_perf_out_if.icache_reads         = mem_perf_in_if.icache_reads;
    assign mem_perf_out_if.icache_read_misses   = mem_perf_in_if.icache_read_misses;

    assign mem_perf_out_if.dcache_reads         = mem_perf_in_if.dcache_reads;
    assign mem_perf_out_if.dcache_writes        = mem_perf_in_if.dcache_writes;
    assign mem_perf_out_if.dcache_read_misses   = mem_perf_in_if.dcache_read_misses;
    assign mem_perf_out_if.dcache_write_misses  = mem_perf_in_if.dcache_write_misses;
    assign mem_perf_out_if.dcache_bank_stalls   = mem_perf_in_if.dcache_bank_stalls;
    assign mem_perf_out_if.dcache_mshr_stalls   = mem_perf_in_if.dcache_mshr_stalls;

    assign mem_perf_out_if.l2cache_reads        = mem_perf_in_if.l2cache_reads;
    assign mem_perf_out_if.l2cache_writes       = mem_perf_in_if.l2cache_writes;
    assign mem_perf_out_if.l2cache_read_misses  = mem_perf_in_if.l2cache_read_misses;
    assign mem_perf_out_if.l2cache_write_misses = mem_perf_in_if.l2cache_write_misses;
    assign mem_perf_out_if.l2cache_bank_stalls  = mem_perf_in_if.l2cache_bank_stalls;
    assign mem_perf_out_if.l2cache_mshr_stalls  = mem_perf_in_if.l2cache_mshr_stalls;

    assign mem_perf_out_if.l3cache_reads        = mem_perf_in_if.l3cache_reads;
    assign mem_perf_out_if.l3cache_writes       = mem_perf_in_if.l3cache_writes;
    assign mem_perf_out_if.l3cache_read_misses  = mem_perf_in_if.l3cache_read_misses;
    assign mem_perf_out_if.l3cache_write_misses = mem_perf_in_if.l3cache_write_misses;
    assign mem_perf_out_if.l3cache_bank_stalls  = mem_perf_in_if.l3cache_bank_stalls;
    assign mem_perf_out_if.l3cache_mshr_stalls  = mem_perf_in_if.l3cache_mshr_stalls;

    assign mem_perf_out_if.mem_reads            = mem_perf_in_if.mem_reads;
    assign mem_perf_out_if.mem_writes           = mem_perf_in_if.mem_writes;
    assign mem_perf_out_if.mem_latency          = mem_perf_in_if.mem_latency;

`ifdef SM_ENABLE
    assign mem_perf_out_if.smem_reads           = perf_smem_if.reads;
    assign mem_perf_out_if.smem_writes          = perf_smem_if.writes;
    assign mem_perf_out_if.smem_bank_stalls     = perf_smem_if.bank_stalls;
`else
    assign mem_perf_out_if.smem_reads           = '0;
    assign mem_perf_out_if.smem_writes          = '0;
    assign mem_perf_out_if.smem_bank_stalls     = '0;
`endif
    
`endif

endmodule
