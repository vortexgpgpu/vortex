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

module VX_socket import VX_gpu_pkg::*; #( 
    parameter SOCKET_ID = 0
) (        
    `SCOPE_IO_DECL
    
    // Clock
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    VX_mem_perf_if.slave    mem_perf_if,
`endif

    // DCRs
    VX_dcr_bus_if.slave     dcr_bus_if,

    // Memory
    VX_mem_bus_if.master    mem_bus_if,

`ifdef GBAR_ENABLE
    // Barrier
    VX_gbar_bus_if.master   gbar_bus_if,
`endif

    // simulation helper signals
    output wire             sim_ebreak,
    output wire [`NUM_REGS-1:0][`XLEN-1:0] sim_wb_value,

    // Status
    output wire             busy
);

`ifdef GBAR_ENABLE
    VX_gbar_bus_if per_core_gbar_bus_if[`SOCKET_SIZE]();

    `RESET_RELAY (gbar_arb_reset, reset);

    VX_gbar_arb #(
        .NUM_REQS (`SOCKET_SIZE),
        .OUT_REG  ((`SOCKET_SIZE > 1) ? 2 : 0)
    ) gbar_arb (
        .clk        (clk),
        .reset      (gbar_arb_reset),
        .bus_in_if  (per_core_gbar_bus_if),
        .bus_out_if (gbar_bus_if)
    );
`endif

    ///////////////////////////////////////////////////////////////////////////

`ifdef PERF_ENABLE
    VX_mem_perf_if mem_perf_tmp_if();
    assign mem_perf_tmp_if.l2cache = mem_perf_if.l2cache;
    assign mem_perf_tmp_if.l3cache = mem_perf_if.l3cache;
    assign mem_perf_tmp_if.smem = 'x;
    assign mem_perf_tmp_if.mem = mem_perf_if.mem;
`endif

    ///////////////////////////////////////////////////////////////////////////

    VX_mem_bus_if #(
        .DATA_SIZE (ICACHE_WORD_SIZE), 
        .TAG_WIDTH (ICACHE_TAG_WIDTH)
    ) per_core_icache_bus_if[`SOCKET_SIZE]();

    VX_mem_bus_if #(
        .DATA_SIZE (ICACHE_LINE_SIZE),
        .TAG_WIDTH (ICACHE_MEM_TAG_WIDTH)
    ) icache_mem_bus_if();

    `RESET_RELAY (icache_reset, reset);

    VX_cache_cluster #(
        .INSTANCE_ID    ($sformatf("socket%0d-icache", SOCKET_ID)),    
        .NUM_UNITS      (`NUM_ICACHES),
        .NUM_INPUTS     (`SOCKET_SIZE),
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
        .TAG_WIDTH      (ICACHE_TAG_WIDTH),
        .UUID_WIDTH     (`UUID_WIDTH),
        .WRITE_ENABLE   (0),
        .CORE_OUT_REG   (2),
        .MEM_OUT_REG    (2)
    ) icache (
    `ifdef PERF_ENABLE
        .cache_perf     (mem_perf_tmp_if.icache),
    `endif
        .clk            (clk),
        .reset          (icache_reset),
        .core_bus_if    (per_core_icache_bus_if),
        .mem_bus_if     (icache_mem_bus_if)
    );

    ///////////////////////////////////////////////////////////////////////////

    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_NOSM_TAG_WIDTH)
    ) per_core_dcache_bus_if[`SOCKET_SIZE * DCACHE_NUM_REQS]();
    
    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_LINE_SIZE),
        .TAG_WIDTH (DCACHE_MEM_TAG_WIDTH)
    ) dcache_mem_bus_if();

    `RESET_RELAY (dcache_reset, reset);

    VX_cache_cluster #(
        .INSTANCE_ID    ($sformatf("socket%0d-dcache", SOCKET_ID)),    
        .NUM_UNITS      (`NUM_DCACHES),
        .NUM_INPUTS     (`SOCKET_SIZE),
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
        .TAG_WIDTH      (DCACHE_NOSM_TAG_WIDTH),
        .UUID_WIDTH     (`UUID_WIDTH),
        .WRITE_ENABLE   (1),        
        .NC_ENABLE      (1),
        .CORE_OUT_REG   (`SM_ENABLED ? 2 : 1),
        .MEM_OUT_REG    (2)
    ) dcache (
    `ifdef PERF_ENABLE
        .cache_perf     (mem_perf_tmp_if.dcache),
    `endif        
        .clk            (clk),
        .reset          (dcache_reset),        
        .core_bus_if    (per_core_dcache_bus_if),
        .mem_bus_if     (dcache_mem_bus_if)
    );

    ///////////////////////////////////////////////////////////////////////////  

    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L1_MEM_TAG_WIDTH)
    ) l1_mem_bus_if[2]();

    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
    ) l1_mem_arb_bus_if[1]();

    `ASSIGN_VX_MEM_BUS_IF_X (l1_mem_bus_if[0], icache_mem_bus_if, L1_MEM_TAG_WIDTH, ICACHE_MEM_TAG_WIDTH);
    `ASSIGN_VX_MEM_BUS_IF_X (l1_mem_bus_if[1], dcache_mem_bus_if, L1_MEM_TAG_WIDTH, DCACHE_MEM_TAG_WIDTH);

    `RESET_RELAY (mem_arb_reset, reset);

    VX_mem_arb #(
        .NUM_INPUTS   (2),
        .DATA_SIZE    (`L1_LINE_SIZE),
        .TAG_WIDTH    (L1_MEM_TAG_WIDTH),
        .TAG_SEL_IDX  (1), // Skip 0 for NC flag
        .ARBITER      ("R"),
        .OUT_REG_REQ  (2),
        .OUT_REG_RSP  (2)
    ) mem_arb (
        .clk        (clk),
        .reset      (mem_arb_reset),
        .bus_in_if  (l1_mem_bus_if),
        .bus_out_if (l1_mem_arb_bus_if)
    );

    `ASSIGN_VX_MEM_BUS_IF (mem_bus_if, l1_mem_arb_bus_if[0]);

    ///////////////////////////////////////////////////////////////////////////

    wire [`SOCKET_SIZE-1:0] per_core_sim_ebreak;
    wire [`SOCKET_SIZE-1:0][`NUM_REGS-1:0][`XLEN-1:0] per_core_sim_wb_value;
    assign sim_ebreak = per_core_sim_ebreak[0];
    assign sim_wb_value = per_core_sim_wb_value[0];
    `UNUSED_VAR (per_core_sim_ebreak)
    `UNUSED_VAR (per_core_sim_wb_value)

    wire [`SOCKET_SIZE-1:0] per_core_busy;

    `BUFFER_DCR_BUS_IF (core_dcr_bus_if, dcr_bus_if, (`SOCKET_SIZE > 1));

    `SCOPE_IO_SWITCH (`SOCKET_SIZE)

    // Generate all cores
    for (genvar i = 0; i < `SOCKET_SIZE; ++i) begin

        `RESET_RELAY (core_reset, reset);

        VX_core #(
            .CORE_ID ((SOCKET_ID * `SOCKET_SIZE) + i)
        ) core (
            `SCOPE_IO_BIND  (i)

            .clk            (clk),
            .reset          (core_reset),

        `ifdef PERF_ENABLE
            .mem_perf_if    (mem_perf_tmp_if),
        `endif
            
            .dcr_bus_if     (core_dcr_bus_if),

            .dcache_bus_if  (per_core_dcache_bus_if[i * DCACHE_NUM_REQS +: DCACHE_NUM_REQS]),

            .icache_bus_if  (per_core_icache_bus_if[i]),

        `ifdef GBAR_ENABLE
            .gbar_bus_if    (per_core_gbar_bus_if[i]),
        `endif

            .sim_ebreak     (per_core_sim_ebreak[i]),
            .sim_wb_value   (per_core_sim_wb_value[i]),
            .busy           (per_core_busy[i])
        );
    end

    `BUFFER_EX(busy, (| per_core_busy), 1'b1, (`SOCKET_SIZE > 1));
    
endmodule
