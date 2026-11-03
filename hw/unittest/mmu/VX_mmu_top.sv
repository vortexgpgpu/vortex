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

// VX_mmu with its buses flattened for the C++ bench: NUM_REQS request lanes,
// the walker link and the flush/fault sidebands.
module VX_mmu_top import VX_gpu_pkg::*, VX_tlb_pkg::*; #(
    parameter NUM_REQS   = 4,
    parameter TLB_SIZE   = 8,
    parameter MSHR_SIZE  = 4,
    parameter DATA_SIZE  = 8,
    parameter TAG_WIDTH  = UUID_WIDTH + 16,
    parameter ATTR_WIDTH = MEM_ATTR_WIDTH,
    parameter ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(DATA_SIZE),
    parameter ID_WIDTH   = `CLOG2(MSHR_SIZE)
) (
    input  wire clk,
    input  wire reset,
    input  wire vm_active,

    // client side
    input  wire [NUM_REQS-1:0]                  in_valid,
    input  wire [NUM_REQS-1:0]                  in_rw,
    input  wire [NUM_REQS-1:0][ADDR_WIDTH-1:0]  in_addr,
    input  wire [NUM_REQS-1:0]                  in_io,     // attr: IO (bypass)
    input  wire [NUM_REQS-1:0]                  in_amo,    // attr: AMO
    input  wire [NUM_REQS-1:0][TAG_WIDTH-1:0]   in_tag,
    output wire [NUM_REQS-1:0]                  in_ready,
    output wire [NUM_REQS-1:0]                  crsp_valid,
    output wire [NUM_REQS-1:0][TAG_WIDTH-1:0]   crsp_tag,
    output wire [NUM_REQS-1:0][DATA_SIZE*8-1:0] crsp_data,
    input  wire [NUM_REQS-1:0]                  crsp_ready,

    // memory side
    output wire [NUM_REQS-1:0]                  out_valid,
    output wire [NUM_REQS-1:0]                  out_rw,
    output wire [NUM_REQS-1:0][ADDR_WIDTH-1:0]  out_addr,
    output wire [NUM_REQS-1:0]                  out_io,
    output wire [NUM_REQS-1:0]                  out_amo,
    output wire [NUM_REQS-1:0][TAG_WIDTH-1:0]   out_tag,
    input  wire [NUM_REQS-1:0]                  out_ready,

    // walker link
    output wire                                 walk_valid,
    output wire [`UP(ID_WIDTH)-1:0]             walk_id,
    output wire [1:0]                           walk_access,
    output wire                                 walk_amo,
    output wire [TLB_VPN_WIDTH-1:0]             walk_vpn,
    input  wire                                 walk_ready,
    input  wire                                 fill_valid,
    input  wire [`UP(ID_WIDTH)-1:0]             fill_id,
    input  wire                                 fill_fault,
    input  wire [TLB_LEVEL_WIDTH-1:0]           fill_level,
    input  wire [TLB_PPN_WIDTH-1:0]             fill_ppn,
    input  wire [TLB_FLAGS_WIDTH-1:0]           fill_flags,
    output wire                                 fill_ready,

    input  wire                                 flush_req,
    output wire                                 flush_done,
    output wire                                 fault_valid,
    output wire                                 empty
);
    VX_mem_bus_if #(
        .DATA_SIZE  (DATA_SIZE),
        .ATTR_WIDTH (ATTR_WIDTH),
        .TAG_WIDTH  (TAG_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH)
    ) core_bus_if[NUM_REQS](), mem_bus_if[NUM_REQS]();

    VX_tlb_bus_if #(.ID_WIDTH (ID_WIDTH)) tlb_bus_if();
    VX_tlb_flush_if flush_if();
    VX_mmu_fault_if fault_if();

    for (genvar l = 0; l < NUM_REQS; ++l) begin : g_lanes
        assign core_bus_if[l].req_valid        = in_valid[l];
        assign core_bus_if[l].req_data.rw      = in_rw[l];
        assign core_bus_if[l].req_data.addr    = in_addr[l];
        assign core_bus_if[l].req_data.data    = (DATA_SIZE*8)'(in_tag[l]);
        assign core_bus_if[l].req_data.byteen  = '1;
        always @(*) begin
            core_bus_if[l].req_data.attr = '0;
            core_bus_if[l].req_data.attr[MEM_ATTR_IO_OFFS]  = in_io[l];
            core_bus_if[l].req_data.attr[MEM_ATTR_AMO_OFFS] = in_amo[l];
        end
        assign core_bus_if[l].req_data.tag     = in_tag[l];
        assign in_ready[l]                     = core_bus_if[l].req_ready;
        assign crsp_valid[l]                   = core_bus_if[l].rsp_valid;
        assign crsp_tag[l]                     = core_bus_if[l].rsp_data.tag;
        assign crsp_data[l]                    = core_bus_if[l].rsp_data.data;
        assign core_bus_if[l].rsp_ready        = crsp_ready[l];

        assign out_valid[l]                    = mem_bus_if[l].req_valid;
        assign out_rw[l]                       = mem_bus_if[l].req_data.rw;
        assign out_addr[l]                     = mem_bus_if[l].req_data.addr;
        assign out_io[l]                       = mem_bus_if[l].req_data.attr[MEM_ATTR_IO_OFFS];
        assign out_amo[l]                      = mem_bus_if[l].req_data.attr[MEM_ATTR_AMO_OFFS];
        assign out_tag[l]                      = mem_bus_if[l].req_data.tag;
        assign mem_bus_if[l].req_ready         = out_ready[l];
        // no memory responses: the bench checks the requests and the
        // teardown responses the MMU makes itself
        assign mem_bus_if[l].rsp_valid         = 1'b0;
        assign mem_bus_if[l].rsp_data          = '0;
        `UNUSED_VAR ({mem_bus_if[l].req_data.data, mem_bus_if[l].req_data.byteen, mem_bus_if[l].rsp_ready, mem_bus_if[l].req_data.attr})
    end

    assign walk_valid  = tlb_bus_if.req_valid;
    assign walk_id     = tlb_bus_if.req_data.id;
    assign walk_access = tlb_bus_if.req_data.access;
    assign walk_amo    = tlb_bus_if.req_data.amo;
    assign walk_vpn    = tlb_bus_if.req_data.vpn;
    assign tlb_bus_if.req_ready = walk_ready;
    assign tlb_bus_if.rsp_valid = fill_valid;
    assign tlb_bus_if.rsp_data  = '{id: fill_id, fault: fill_fault, level: fill_level, ppn: fill_ppn, flags: fill_flags};
    assign fill_ready = tlb_bus_if.rsp_ready;

    assign flush_if.req = flush_req;
    assign flush_done   = flush_if.done;
    assign fault_valid  = fault_if.valid;
    `UNUSED_VAR ({fault_if.va, fault_if.access, fault_if.amo})

    VX_mmu #(
        .INSTANCE_ID ("mmu"),
        .NUM_REQS    (NUM_REQS),
        .TLB_SIZE    (TLB_SIZE),
        .MSHR_SIZE   (MSHR_SIZE),
        .DATA_SIZE   (DATA_SIZE),
        .TAG_WIDTH   (TAG_WIDTH),
        .ADDR_WIDTH  (ADDR_WIDTH),
        .ATTR_WIDTH  (ATTR_WIDTH),
        .ID_WIDTH    (ID_WIDTH)
    ) mmu (
        .clk         (clk),
        .reset       (reset),
        .vm_active   (vm_active),
        .core_bus_if (core_bus_if),
        .mem_bus_if  (mem_bus_if),
        .tlb_bus_if  (tlb_bus_if),
        .flush_if    (flush_if),
        .fault_if    (fault_if),
        .empty       (empty)
    );

endmodule
