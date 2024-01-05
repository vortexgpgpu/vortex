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

module VX_cluster import VX_gpu_pkg::*; #(
    parameter CLUSTER_ID = 0
) ( 
    `SCOPE_IO_DECL

    // Clock
    input  wire                 clk,
    input  wire                 reset,

`ifdef PERF_ENABLE
    VX_mem_perf_if.slave        mem_perf_if,
`endif

    // DCRs
    VX_dcr_bus_if.slave         dcr_bus_if,

    // Memory
    VX_mem_bus_if.master        mem_bus_if,

    // simulation helper signals
    output wire                 sim_ebreak,
    output wire [`NUM_REGS-1:0][`XLEN-1:0] sim_wb_value,

    // Status
    output wire                 busy
);

`ifdef SCOPE
    localparam scope_socket = 0;
    `SCOPE_IO_SWITCH (scope_socket + `NUM_SOCKETS);
`endif    

`ifdef PERF_ENABLE
    VX_mem_perf_if mem_perf_tmp_if();    
    assign mem_perf_tmp_if.icache  = 'x;
    assign mem_perf_tmp_if.dcache  = 'x;
    assign mem_perf_tmp_if.l3cache = mem_perf_if.l3cache;
    assign mem_perf_tmp_if.smem    = 'x;
    assign mem_perf_tmp_if.mem     = mem_perf_if.mem;
`endif    

`ifdef GBAR_ENABLE

    VX_gbar_bus_if per_socket_gbar_bus_if[`NUM_SOCKETS]();
    VX_gbar_bus_if gbar_bus_if();

    `RESET_RELAY (gbar_reset, reset);

    VX_gbar_arb #(
        .NUM_REQS (`NUM_SOCKETS),
        .OUT_REG  ((`NUM_SOCKETS > 2) ? 1 : 0) // bgar_unit has no backpressure
    ) gbar_arb (
        .clk        (clk),
        .reset      (gbar_reset),
        .bus_in_if  (per_socket_gbar_bus_if),
        .bus_out_if (gbar_bus_if)
    );

    VX_gbar_unit #(
        .INSTANCE_ID ($sformatf("gbar%0d", CLUSTER_ID))
    ) gbar_unit (
        .clk         (clk),
        .reset       (gbar_reset),
        .gbar_bus_if (gbar_bus_if)
    );

`endif

    VX_mem_bus_if #(
        .DATA_SIZE (`L1_LINE_SIZE),
        .TAG_WIDTH (L1_MEM_ARB_TAG_WIDTH)
    ) per_socket_mem_bus_if[`NUM_SOCKETS]();

    `RESET_RELAY (l2_reset, reset);

    VX_cache_wrap #(
        .INSTANCE_ID    ("l2cache"),
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
        .TAG_WIDTH      (L2_TAG_WIDTH),
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
        .cache_perf     (mem_perf_tmp_if.l2cache),
    `endif
        .core_bus_if    (per_socket_mem_bus_if),
        .mem_bus_if     (mem_bus_if)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire [`NUM_SOCKETS-1:0] per_socket_sim_ebreak;
    wire [`NUM_SOCKETS-1:0][`NUM_REGS-1:0][`XLEN-1:0] per_socket_sim_wb_value;
    assign sim_ebreak = per_socket_sim_ebreak[0];
    assign sim_wb_value = per_socket_sim_wb_value[0];
    `UNUSED_VAR (per_socket_sim_ebreak)
    `UNUSED_VAR (per_socket_sim_wb_value)

    VX_dcr_bus_if socket_dcr_bus_tmp_if();
    assign socket_dcr_bus_tmp_if.write_valid = dcr_bus_if.write_valid && (dcr_bus_if.write_addr >= `VX_DCR_BASE_STATE_BEGIN && dcr_bus_if.write_addr < `VX_DCR_BASE_STATE_END);
    assign socket_dcr_bus_tmp_if.write_addr  = dcr_bus_if.write_addr;
    assign socket_dcr_bus_tmp_if.write_data  = dcr_bus_if.write_data;

    wire [`NUM_SOCKETS-1:0] per_socket_busy;

    `BUFFER_DCR_BUS_IF (socket_dcr_bus_if, socket_dcr_bus_tmp_if, (`NUM_SOCKETS > 1));

    // Generate all sockets
    for (genvar i = 0; i < `NUM_SOCKETS; ++i) begin

        `RESET_RELAY (socket_reset, reset);

        VX_socket #(
            .SOCKET_ID ((CLUSTER_ID * `NUM_SOCKETS) + i)
        ) socket (
            `SCOPE_IO_BIND  (scope_socket+i)

            .clk            (clk),
            .reset          (socket_reset),

        `ifdef PERF_ENABLE
            .mem_perf_if    (mem_perf_tmp_if),
        `endif
            
            .dcr_bus_if     (socket_dcr_bus_if),

            .mem_bus_if     (per_socket_mem_bus_if[i]),

        `ifdef GBAR_ENABLE
            .gbar_bus_if    (per_socket_gbar_bus_if[i]),
        `endif

            .sim_ebreak     (per_socket_sim_ebreak[i]),
            .sim_wb_value   (per_socket_sim_wb_value[i]),
            .busy           (per_socket_busy[i])
        );
    end

    `BUFFER_EX(busy, (| per_socket_busy), 1'b1, (`NUM_SOCKETS > 1));

endmodule
