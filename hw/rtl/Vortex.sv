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

module Vortex import VX_gpu_pkg::*; (
    `SCOPE_IO_DECL

    // Clock
    input  wire                             clk,
    input  wire                             reset,

    // Memory request
    output wire                             mem_req_valid,
    output wire                             mem_req_rw,
    output wire [`VX_MEM_BYTEEN_WIDTH-1:0]  mem_req_byteen,
    output wire [`VX_MEM_ADDR_WIDTH-1:0]    mem_req_addr,
    output wire [`VX_MEM_DATA_WIDTH-1:0]    mem_req_data,
    output wire [`VX_MEM_TAG_WIDTH-1:0]     mem_req_tag,
    input  wire                             mem_req_ready,

    // Memory response    
    input wire                              mem_rsp_valid,    
    input wire [`VX_MEM_DATA_WIDTH-1:0]     mem_rsp_data,
    input wire [`VX_MEM_TAG_WIDTH-1:0]      mem_rsp_tag,
    output wire                             mem_rsp_ready,

    // DCR write request
    input  wire                             dcr_wr_valid,
    input  wire [`VX_DCR_ADDR_WIDTH-1:0]    dcr_wr_addr,
    input  wire [`VX_DCR_DATA_WIDTH-1:0]    dcr_wr_data,

    // Status
    output wire                             busy
);

`ifdef PERF_ENABLE
    VX_mem_perf_if mem_perf_if();
    assign mem_perf_if.icache  = 'x;
    assign mem_perf_if.dcache  = 'x;
    assign mem_perf_if.l2cache = 'x;
`endif

    VX_mem_bus_if #(
        .DATA_SIZE (`L2_LINE_SIZE),
        .TAG_WIDTH (L2_MEM_TAG_WIDTH)
    ) per_cluster_mem_bus_if[`NUM_CLUSTERS]();

    VX_mem_bus_if #(
        .DATA_SIZE (`L3_LINE_SIZE),
        .TAG_WIDTH (L3_MEM_TAG_WIDTH)
    ) mem_bus_if();

    `RESET_RELAY (l3_reset, reset);

    VX_cache_wrap #(
        .INSTANCE_ID    ("l3cache"),
        .CACHE_SIZE     (`L3_CACHE_SIZE),
        .LINE_SIZE      (`L3_LINE_SIZE),
        .NUM_BANKS      (`L3_NUM_BANKS),
        .NUM_WAYS       (`L3_NUM_WAYS),
        .WORD_SIZE      (L3_WORD_SIZE),
        .NUM_REQS       (L3_NUM_REQS),
        .CRSQ_SIZE      (`L3_CRSQ_SIZE),
        .MSHR_SIZE      (`L3_MSHR_SIZE),
        .MRSQ_SIZE      (`L3_MRSQ_SIZE),
        .MREQ_SIZE      (`L3_MREQ_SIZE),
        .TAG_WIDTH      (L2_MEM_TAG_WIDTH),
        .WRITE_ENABLE   (1),
        .UUID_WIDTH     (`UUID_WIDTH),  
        .CORE_OUT_REG   (2),
        .MEM_OUT_REG    (2),
        .NC_ENABLE      (1),
        .PASSTHRU       (!`L3_ENABLED)
    ) l3cache (
        .clk            (clk),
        .reset          (l3_reset),

    `ifdef PERF_ENABLE
        .cache_perf     (mem_perf_if.l3cache),
    `endif

        .core_bus_if    (per_cluster_mem_bus_if),
        .mem_bus_if     (mem_bus_if)
    );

    assign mem_req_valid = mem_bus_if.req_valid;
    assign mem_req_rw    = mem_bus_if.req_data.rw;
    assign mem_req_byteen= mem_bus_if.req_data.byteen;
    assign mem_req_addr  = mem_bus_if.req_data.addr;
    assign mem_req_data  = mem_bus_if.req_data.data;
    assign mem_req_tag   = mem_bus_if.req_data.tag;
    assign mem_bus_if.req_ready = mem_req_ready;

    assign mem_bus_if.rsp_valid = mem_rsp_valid;
    assign mem_bus_if.rsp_data.data  = mem_rsp_data;
    assign mem_bus_if.rsp_data.tag   = mem_rsp_tag;
    assign mem_rsp_ready = mem_bus_if.rsp_ready;

    wire mem_req_fire = mem_req_valid && mem_req_ready;
    wire mem_rsp_fire = mem_rsp_valid && mem_rsp_ready;
    `UNUSED_VAR (mem_req_fire)
    `UNUSED_VAR (mem_rsp_fire)

    wire sim_ebreak /* verilator public */;
    wire [`NUM_REGS-1:0][`XLEN-1:0] sim_wb_value /* verilator public */;    
    wire [`NUM_CLUSTERS-1:0] per_cluster_sim_ebreak;
    wire [`NUM_CLUSTERS-1:0][`NUM_REGS-1:0][`XLEN-1:0] per_cluster_sim_wb_value;
    assign sim_ebreak = per_cluster_sim_ebreak[0];
    assign sim_wb_value = per_cluster_sim_wb_value[0];
    `UNUSED_VAR (per_cluster_sim_ebreak)
    `UNUSED_VAR (per_cluster_sim_wb_value)        

    VX_dcr_bus_if dcr_bus_if();
    assign dcr_bus_if.write_valid = dcr_wr_valid;
    assign dcr_bus_if.write_addr  = dcr_wr_addr;
    assign dcr_bus_if.write_data  = dcr_wr_data;

    wire [`NUM_CLUSTERS-1:0] per_cluster_busy;

    `SCOPE_IO_SWITCH (`NUM_CLUSTERS)

    // Generate all clusters
    for (genvar i = 0; i < `NUM_CLUSTERS; ++i) begin

        `RESET_RELAY (cluster_reset, reset);

        `BUFFER_DCR_BUS_IF (cluster_dcr_bus_if, dcr_bus_if, (`NUM_CLUSTERS > 1));

        VX_cluster #(
            .CLUSTER_ID (i)
        ) cluster (
            `SCOPE_IO_BIND (i)

            .clk                (clk),
            .reset              (cluster_reset),

        `ifdef PERF_ENABLE
            .mem_perf_if        (mem_perf_if),
        `endif
            
            .dcr_bus_if         (cluster_dcr_bus_if),

            .mem_bus_if         (per_cluster_mem_bus_if[i]),

            .sim_ebreak         (per_cluster_sim_ebreak[i]),
            .sim_wb_value       (per_cluster_sim_wb_value[i]),

            .busy               (per_cluster_busy[i])
        );
    end

    `BUFFER_EX(busy, (| per_cluster_busy), 1'b1, (`NUM_CLUSTERS > 1));

`ifdef PERF_ENABLE

    reg [`PERF_CTR_BITS-1:0] perf_mem_pending_reads;    
    mem_perf_t mem_perf;

    always @(posedge clk) begin
        if (reset) begin
            perf_mem_pending_reads <= '0;
        end else begin
            perf_mem_pending_reads <= $signed(perf_mem_pending_reads) + 
                `PERF_CTR_BITS'($signed(2'(mem_req_fire && ~mem_bus_if.req_data.rw) - 2'(mem_rsp_fire)));
        end
    end

    wire mem_rd_req_fire = mem_req_fire && ~mem_bus_if.req_data.rw;
    wire mem_wr_req_fire = mem_req_fire && mem_bus_if.req_data.rw;

    always @(posedge clk) begin
        if (reset) begin       
            mem_perf <= '0;
        end else begin
            mem_perf.reads <= mem_perf.reads + `PERF_CTR_BITS'(mem_rd_req_fire);
            mem_perf.writes <= mem_perf.writes + `PERF_CTR_BITS'(mem_wr_req_fire);
            mem_perf.latency <= mem_perf.latency + perf_mem_pending_reads;
        end
    end
    assign mem_perf_if.mem = mem_perf;
    
`endif

`ifdef DBG_TRACE_CORE_MEM
    always @(posedge clk) begin
        if (mem_req_fire) begin
            if (mem_req_rw)
                `TRACE(1, ("%d: MEM Wr Req: addr=0x%0h, tag=0x%0h, byteen=0x%0h data=0x%0h\n", $time, `TO_FULL_ADDR(mem_req_addr), mem_req_tag, mem_req_byteen, mem_req_data));
            else
                `TRACE(1, ("%d: MEM Rd Req: addr=0x%0h, tag=0x%0h, byteen=0x%0h\n", $time, `TO_FULL_ADDR(mem_req_addr), mem_req_tag, mem_req_byteen));
        end
        if (mem_rsp_fire) begin
            `TRACE(1, ("%d: MEM Rsp: tag=0x%0h, data=0x%0h\n", $time, mem_rsp_tag, mem_rsp_data));
        end
    end
`endif

`ifdef SIMULATION
    always @(posedge clk) begin
        $fflush(); // flush stdout buffer
    end
`endif

endmodule
