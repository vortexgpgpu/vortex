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

`ifdef EXT_F_ENABLE
`include "VX_fpu_define.vh"
`endif

module VX_core_top import VX_gpu_pkg::*; #(
    parameter CORE_ID = 0
) (
    // Clock
    input wire                              clk,
    input wire                              reset,

    input wire                              dcr_write_valid,
    input wire [`VX_DCR_ADDR_WIDTH-1:0]     dcr_write_addr,
    input wire [`VX_DCR_DATA_WIDTH-1:0]     dcr_write_data,

    output wire [DCACHE_NUM_REQS-1:0]       dcache_req_valid,
    output wire [DCACHE_NUM_REQS-1:0]       dcache_req_rw,
    output wire [DCACHE_NUM_REQS-1:0][DCACHE_WORD_SIZE-1:0] dcache_req_byteen,
    output wire [DCACHE_NUM_REQS-1:0][DCACHE_ADDR_WIDTH-1:0] dcache_req_addr,
    output wire [DCACHE_NUM_REQS-1:0][`MEM_REQ_FLAGS_WIDTH-1:0] dcache_req_flags,
    output wire [DCACHE_NUM_REQS-1:0][DCACHE_WORD_SIZE*8-1:0] dcache_req_data,
    output wire [DCACHE_NUM_REQS-1:0][DCACHE_TAG_WIDTH-1:0] dcache_req_tag,
    input  wire [DCACHE_NUM_REQS-1:0]       dcache_req_ready,

    input wire  [DCACHE_NUM_REQS-1:0]       dcache_rsp_valid,
    input wire  [DCACHE_NUM_REQS-1:0][DCACHE_WORD_SIZE*8-1:0] dcache_rsp_data,
    input wire  [DCACHE_NUM_REQS-1:0][DCACHE_TAG_WIDTH-1:0] dcache_rsp_tag,
    output wire [DCACHE_NUM_REQS-1:0]       dcache_rsp_ready,

    output wire                             icache_req_valid,
    output wire                             icache_req_rw,
    output wire [ICACHE_WORD_SIZE-1:0]      icache_req_byteen,
    output wire [ICACHE_ADDR_WIDTH-1:0]     icache_req_addr,
    output wire [ICACHE_WORD_SIZE*8-1:0]    icache_req_data,
    output wire [ICACHE_TAG_WIDTH-1:0]      icache_req_tag,
    input  wire                             icache_req_ready,

    input wire                              icache_rsp_valid,
    input wire  [ICACHE_WORD_SIZE*8-1:0]    icache_rsp_data,
    input wire  [ICACHE_TAG_WIDTH-1:0]      icache_rsp_tag,
    output wire                             icache_rsp_ready,

`ifdef GBAR_ENABLE
    output wire                             gbar_req_valid,
    output wire [`NB_WIDTH-1:0]             gbar_req_id,
    output wire [`NC_WIDTH-1:0]             gbar_req_size_m1,
    output wire [`NC_WIDTH-1:0]             gbar_req_core_id,
    input wire                              gbar_req_ready,
    input wire                              gbar_rsp_valid,
    input wire [`NB_WIDTH-1:0]              gbar_rsp_id,
`endif
    // Status
    output wire                             busy
);

`ifdef GBAR_ENABLE
    VX_gbar_bus_if gbar_bus_if();

    assign gbar_req_valid = gbar_bus_if.req_valid;
    assign gbar_req_id = gbar_bus_if.req_id;
    assign gbar_req_size_m1 = gbar_bus_if.req_size_m1;
    assign gbar_req_core_id =  gbar_bus_if.req_core_id;
    assign gbar_bus_if.req_ready = gbar_req_ready;
    assign gbar_bus_if.rsp_valid = gbar_rsp_valid;
    assign gbar_bus_if.rsp_id = gbar_rsp_id;
`endif

    VX_dcr_bus_if dcr_bus_if();

    assign dcr_bus_if.write_valid = dcr_write_valid;
    assign dcr_bus_if.write_addr = dcr_write_addr;
    assign dcr_bus_if.write_data = dcr_write_data;

    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_TAG_WIDTH)
    ) dcache_bus_if[DCACHE_NUM_REQS]();

    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin
        assign dcache_req_valid[i] = dcache_bus_if[i].req_valid;
        assign dcache_req_rw[i] = dcache_bus_if[i].req_data.rw;
        assign dcache_req_byteen[i] = dcache_bus_if[i].req_data.byteen;
        assign dcache_req_addr[i] = dcache_bus_if[i].req_data.addr;
        assign dcache_req_flags[i] = dcache_bus_if[i].req_data.flags;
        assign dcache_req_data[i] = dcache_bus_if[i].req_data.data;
        assign dcache_req_tag[i] = dcache_bus_if[i].req_data.tag;
        assign dcache_bus_if[i].req_ready = dcache_req_ready[i];

        assign dcache_bus_if[i].rsp_valid = dcache_rsp_valid[i];
        assign dcache_bus_if[i].rsp_data.tag = dcache_rsp_tag[i];
        assign dcache_bus_if[i].rsp_data.data = dcache_rsp_data[i];
        assign dcache_rsp_ready[i] = dcache_bus_if[i].rsp_ready;
    end

    VX_mem_bus_if #(
        .DATA_SIZE (ICACHE_WORD_SIZE),
        .TAG_WIDTH (ICACHE_TAG_WIDTH)
    ) icache_bus_if();

    assign icache_req_valid = icache_bus_if.req_valid;
    assign icache_req_rw = icache_bus_if.req_data.rw;
    assign icache_req_byteen = icache_bus_if.req_data.byteen;
    assign icache_req_addr = icache_bus_if.req_data.addr;
    assign icache_req_data = icache_bus_if.req_data.data;
    assign icache_req_tag = icache_bus_if.req_data.tag;
    assign icache_bus_if.req_ready = icache_req_ready;
    `UNUSED_VAR (icache_bus_if.req_data.flags)

    assign icache_bus_if.rsp_valid = icache_rsp_valid;
    assign icache_bus_if.rsp_data.tag = icache_rsp_tag;
    assign icache_bus_if.rsp_data.data = icache_rsp_data;
    assign icache_rsp_ready = icache_bus_if.rsp_ready;

`ifdef PERF_ENABLE
    sysmem_perf_t mem_perf;
    assign mem_perf.icache  = '0;
    assign mem_perf.dcache  = '0;
    assign mem_perf.l2cache = '0;
    assign mem_perf.l3cache = '0;
    assign mem_perf.lmem    = '0;
    assign mem_perf.mem     = '0;
`endif

`ifdef SCOPE
    wire [0:0] scope_reset_w = 1'b0;
    wire [0:0] scope_bus_in_w = 1'b0;
    wire [0:0] scope_bus_out_w;
    `UNUSED_VAR (scope_bus_out_w)
`endif

    VX_core #(
        .INSTANCE_ID (`SFORMATF(("core"))),
        .CORE_ID (CORE_ID)
    ) core (
        `SCOPE_IO_BIND (0)
        .clk            (clk),
        .reset          (reset),

    `ifdef PERF_ENABLE
        .sysmem_perf    (sysmem_perf),
    `endif

        .dcr_bus_if     (dcr_bus_if),

        .dcache_bus_if  (dcache_bus_if),

        .icache_bus_if  (icache_bus_if),

    `ifdef GBAR_ENABLE
        .gbar_bus_if    (gbar_bus_if),
    `endif

        .busy           (busy)
    );

endmodule
