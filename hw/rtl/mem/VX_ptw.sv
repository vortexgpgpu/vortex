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

// Page-table walker complex. This per-core instance drives a single walker
// whose PTE fetches ride the core's dcache port.
module VX_ptw import VX_gpu_pkg::*, VX_tlb_pkg::*; #(
    parameter PT_LEVELS  = `VX_VM_PT_LEVEL,
    parameter DATA_SIZE  = DCACHE_WORD_SIZE,
    parameter TAG_WIDTH  = DCACHE_TAG_WIDTH_BASE,
    parameter ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(DATA_SIZE),
    parameter ATTR_WIDTH = MEM_ATTR_WIDTH,
    parameter ID_WIDTH   = 4
) (
    input wire clk,
    input wire reset,

    input wire [`VX_CFG_XLEN-1:0] satp,

    VX_tlb_bus_if.slave   miss_if,
    VX_mem_bus_if.master  mem_bus_if,
    VX_tlb_flush_if.slave flush_if,

    output wire           empty
);
    wire                       w_active;
    wire [`UP(ID_WIDTH)-1:0]   w_rsp_id;
    wire                       w_rsp_fault;
    wire [TLB_LEVEL_WIDTH-1:0] w_rsp_level;
    wire [TLB_PPN_WIDTH-1:0]   w_rsp_ppn;
    wire [TLB_FLAGS_WIDTH-1:0] w_rsp_flags;

    VX_ptw_walker #(
        .PT_LEVELS  (PT_LEVELS),
        .DATA_SIZE  (DATA_SIZE),
        .TAG_WIDTH  (TAG_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH),
        .ATTR_WIDTH (ATTR_WIDTH),
        .ID_WIDTH   (ID_WIDTH)
    ) walker (
        .clk        (clk),
        .reset      (reset),
        .satp       (satp),
        .req_valid  (miss_if.req_valid),
        .req_id     (miss_if.req_data.id),
        .req_access (miss_if.req_data.access),
        .req_amo    (miss_if.req_data.amo),
        .req_vpn    (miss_if.req_data.vpn),
        .req_ready  (miss_if.req_ready),
        .rsp_valid  (miss_if.rsp_valid),
        .rsp_id     (w_rsp_id),
        .rsp_fault  (w_rsp_fault),
        .rsp_level  (w_rsp_level),
        .rsp_ppn    (w_rsp_ppn),
        .rsp_flags  (w_rsp_flags),
        .rsp_ready  (miss_if.rsp_ready),
        .active     (w_active),
        .mem_bus_if (mem_bus_if)
    );

    assign miss_if.rsp_data = '{
        id:    w_rsp_id,
        fault: w_rsp_fault,
        level: w_rsp_level,
        ppn:   w_rsp_ppn,
        flags: w_rsp_flags
    };

    assign empty = ~w_active;

    // Flush completes once no walk is in flight (installs are dropped there).
    assign flush_if.done = flush_if.req && ~w_active;

endmodule
